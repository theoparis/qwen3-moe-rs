//! S.2: incremental detokenization + stop-string holdback.
//!
//! Bounded-tail decode-and-diff with UTF-8 holdback; stop-string scan with
//! cross-chunk boundary handling (hold back the longest stop prefix).
//! Emission is GATED behind this module. See docs/SERVE_PLAN.md S.2.
//!
//! # Two holdback invariants (both GATE SSE emission — nothing bypasses them)
//!
//! 1. **UTF-8 holdback.** Text returned by [`IncrementalDetok::push`] is ALWAYS
//!    complete, valid UTF-8. A multi-byte codepoint split across several tokens
//!    (byte-level BPE emits `U+FFFD` for the partial bytes) is held back until
//!    the final byte arrives, at which point the whole codepoint is emitted
//!    exactly once — never a `U+FFFD` mojibake, never a half char. If a codepoint
//!    never completes (end-of-generation), its bytes are DROPPED in
//!    [`IncrementalDetok::finish`] (documented policy: drop, do not emit `U+FFFD`).
//!    The holdback is BOUNDED: a real split completes within
//!    `MAX_UTF8_HOLDBACK_TOKENS` (4), so a longer `U+FFFD`-terminated run is a
//!    literal replacement-char token (or invalid-byte run) the model actually
//!    emitted — it is force-committed as content, never held forever.
//!
//! 2. **Stop-string holdback.** Text that either IS a stop string or is a nonempty
//!    trailing suffix that could still grow into a stop string is NEVER emitted.
//!    The LONGEST such suffix is held across chunk boundaries. When a stop string
//!    fully matches, the stop and everything after it are excluded from the emitted
//!    text and the hit is reported (the earliest match position wins).
//!
//! # Bounded-window complexity
//!
//! `push` decodes only a bounded TAIL window `tokens[prefix_offset..len]` and
//! diffs it against the previous decode of `tokens[prefix_offset..read_offset]`.
//! After every successful commit `prefix_offset` jumps forward to the old
//! `read_offset`, so the window spans `(read_offset - prefix_offset) + (len -
//! read_offset)` tokens: the first term is the size of the LAST commit and the
//! second is the current uncommitted run. The uncommitted run is capped because
//! a still-`U+FFFD`-terminated run is force-committed once it exceeds
//! `MAX_UTF8_HOLDBACK_TOKENS` (4) — so it reaches at most 5 tokens before it
//! commits, and every commit's run is likewise ≤ 5. The window is therefore
//! bounded by `2 · (MAX_UTF8_HOLDBACK_TOKENS + 1) = 10` tokens (in the common
//! single-token-commit case just `1 + K`, `K ≤ 4`), and — crucially —
//! INDEPENDENT of total generated length: work is O(1) per token, O(n) overall
//! UNCONDITIONALLY (before the force-commit guard a sustained `U+FFFD` run
//! stalled `read_offset` and grew the window to O(n), an O(n²) head-of-line
//! stall). The full growing sequence is NEVER re-decoded.

use tokenizers::Tokenizer;

/// Upper bound on how many tokens a genuinely-split multi-byte codepoint can
/// occupy. A UTF-8 codepoint is at most 4 bytes, and byte-level BPE emits at
/// least 1 byte per token, so any real split completes within 4 tokens. An
/// uncommitted run that is STILL `U+FFFD`-terminated after more than this many
/// tokens therefore cannot be an incomplete codepoint — it is the model
/// emitting the literal `U+FFFD` vocab token (id 5691 in the 30B vocab)
/// repeatedly, or a sustained invalid-UTF-8 byte run. Past this bound we
/// FORCE-COMMIT the tail so `read_offset` advances, keeping the decode window
/// bounded and avoiding a permanent head-of-line stall (see [`IncrementalDetok::push`]).
const MAX_UTF8_HOLDBACK_TOKENS: usize = 4;

/// Outcome of pushing one token id into the incremental detokenizer.
///
/// `text` is the newly emittable text — already past BOTH holdbacks, therefore
/// safe to hand to the SSE layer verbatim. `stop` is `Some(matched)` on the push
/// that completes a stop string (the stop itself is excluded from `text`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PushResult {
    /// Text safe to emit now (complete UTF-8, no stop-string prefix).
    pub text: String,
    /// Set once, naming the stop string that matched. When set, generation
    /// should stop; the stop string is NOT included in `text`.
    pub stop: Option<String>,
}

/// Incremental detokenizer with UTF-8 and stop-string holdback.
///
/// One instance per generation. Feed generated token ids in order via [`push`];
/// call [`finish`] once at end-of-generation to flush held-back text.
///
/// [`push`]: IncrementalDetok::push
/// [`finish`]: IncrementalDetok::finish
pub struct IncrementalDetok {
    tokenizer: Tokenizer,
    /// Stop strings; empty strings are filtered out at construction (an empty
    /// stop would match everywhere and is meaningless).
    stops: Vec<String>,
    /// All generated token ids, in order. Only a bounded tail is ever decoded.
    tokens: Vec<u32>,
    /// Left edge of the decode window. Lags `read_offset` by one commit so the
    /// boundary token always decodes with enough left context for the byte-level
    /// decoder's leading-space normalization to be stable across the seam.
    prefix_offset: usize,
    /// Right edge of the already-committed prefix. `tokens[prefix_offset..read_offset]`
    /// is the text we diffed away last time; the new suffix is what we emit.
    read_offset: usize,
    /// Text that has been decoded (complete UTF-8) but not yet released to the
    /// client because a trailing suffix might still grow into a stop string.
    pending: String,
    /// Latched once a stop string matches; further pushes are inert.
    stopped: bool,
}

impl IncrementalDetok {
    /// Construct from a `tokenizers::Tokenizer` and the request's stop strings.
    ///
    /// The tokenizer is used with `skip_special_tokens = false` so that content
    /// tokens like `<think>`/`</think>` pass through byte-exact (they are NOT
    /// stop strings and must reach the reasoning/content split downstream).
    pub fn new(tokenizer: Tokenizer, stops: Vec<String>) -> Self {
        let stops = stops.into_iter().filter(|s| !s.is_empty()).collect();
        Self {
            tokenizer,
            stops,
            tokens: Vec::new(),
            prefix_offset: 0,
            read_offset: 0,
            pending: String::new(),
            stopped: false,
        }
    }

    /// Decode a bounded id slice, never the full sequence. Errors decode to `""`
    /// (the caller re-tries on the next token via the growing tail window).
    fn decode(&self, ids: &[u32]) -> String {
        self.tokenizer.decode(ids, false).unwrap_or_default()
    }

    /// Push one generated token id; return any newly emittable text plus a stop
    /// hit indicator. See the module-level invariants.
    pub fn push(&mut self, token_id: u32) -> PushResult {
        if self.stopped {
            return PushResult::default();
        }

        self.tokens.push(token_id);
        let n = self.tokens.len();

        // --- Invariant 1: bounded-tail decode-and-diff with UTF-8 holdback. ---
        let prefix_text = self.decode(&self.tokens[self.prefix_offset..self.read_offset]);
        let new_text = self.decode(&self.tokens[self.prefix_offset..n]);

        // Normally commit only when the tail grew, is a clean continuation of
        // what we already emitted, and does NOT end mid-codepoint (trailing
        // U+FFFD ⇒ a multi-byte char is split across tokens — hold every byte
        // back until it completes, so the whole codepoint emits exactly once).
        let grew_clean = new_text.len() > prefix_text.len() && new_text.starts_with(&prefix_text);
        let ends_incomplete = new_text.ends_with(char::REPLACEMENT_CHARACTER);

        // FORCE-COMMIT guard: a genuine split completes within
        // MAX_UTF8_HOLDBACK_TOKENS (see its docs). If the uncommitted run
        // `n - read_offset` EXCEEDS that while still U+FFFD-terminated, the
        // U+FFFD is real content (the model emitted the literal replacement-char
        // token, or an unrecoverable byte run) — NOT a split. Emit it anyway,
        // matching what a batch decode of these ids produces, so read_offset
        // advances and the window can never grow without bound (no O(n²) stall).
        let force = ends_incomplete && (n - self.read_offset) > MAX_UTF8_HOLDBACK_TOKENS;

        if grew_clean && (!ends_incomplete || force) {
            let committed = new_text[prefix_text.len()..].to_string();
            // Advance the window: the just-committed tokens become the new left
            // context, keeping the decode window bounded (see module docs).
            self.prefix_offset = self.read_offset;
            self.read_offset = n;
            self.pending.push_str(&committed);
        }
        // else: incomplete/normalization-unstable tail — hold back, retry next push.

        // --- Invariant 2: stop-string scan over the pending buffer. ---
        self.scan_and_release()
    }

    /// Run the stop scan over `self.pending`, mutate holdback state, and return
    /// what is safe to emit now.
    fn scan_and_release(&mut self) -> PushResult {
        match scan_stops(&self.pending, &self.stops) {
            ScanOutcome::Hit { emit_len, stop } => {
                let text = self.pending[..emit_len].to_string();
                self.pending.clear();
                self.stopped = true;
                PushResult {
                    text,
                    stop: Some(stop),
                }
            }
            ScanOutcome::Partial { emit_len } => {
                let text = self.pending[..emit_len].to_string();
                // Retain the trailing suffix (a possible stop prefix) for the
                // next chunk; drop only the safe prefix we are emitting.
                self.pending.drain(..emit_len);
                PushResult { text, stop: None }
            }
        }
    }

    /// Flush held-back text at end-of-generation and return it.
    ///
    /// Policy: the stop-prefix holdback is RELEASED (no stop completed, so the
    /// text is genuine output). Any UTF-8-incomplete trailing bytes (a codepoint
    /// that never received its final byte) are DROPPED — never emitted as
    /// `U+FFFD`. Idempotent-ish: after a stop hit, returns `""` (pending already
    /// cleared).
    ///
    /// Scope note: with the force-commit guard in [`push`], the uncommitted tail
    /// reaching `finish` is at most `MAX_UTF8_HOLDBACK_TOKENS` (4) tokens — only
    /// a genuine trailing incomplete codepoint survives to be dropped here.
    ///
    /// ACCEPTED EDGE (flagged independently by two reviewers): a LEGITIMATE
    /// model-emitted `U+FFFD` as the very last character(s) of generation is
    /// indistinguishable, at the `String` level, from the dangling bytes of a
    /// split codepoint, so it is dropped too. vLLM's replacement-char heuristic
    /// has the same behavior; disambiguating would require byte-level decoder
    /// access that `tokenizers` does not expose. (Mid-stream, the force-commit
    /// guard already emits sustained `U+FFFD` runs correctly — only a run of
    /// ≤ 4 tokens landing exactly at end-of-generation is affected.)
    ///
    /// [`push`]: IncrementalDetok::push
    pub fn finish(&mut self) -> String {
        if self.stopped {
            return String::new();
        }

        // Final UTF-8 flush: decode whatever tail was held for completeness and
        // strip any trailing replacement char(s) from a codepoint that never
        // completed (drop policy).
        let n = self.tokens.len();
        let prefix_text = self.decode(&self.tokens[self.prefix_offset..self.read_offset]);
        let new_text = self.decode(&self.tokens[self.prefix_offset..n]);
        if new_text.len() > prefix_text.len() && new_text.starts_with(&prefix_text) {
            let mut tail = new_text[prefix_text.len()..].to_string();
            while tail.ends_with(char::REPLACEMENT_CHARACTER) {
                tail.pop();
            }
            self.pending.push_str(&tail);
            self.prefix_offset = self.read_offset;
            self.read_offset = n;
        }

        std::mem::take(&mut self.pending)
    }
}

/// Result of scanning a pending buffer against the stop strings. Pure logic —
/// exercised directly by unit tests.
#[derive(Debug, PartialEq, Eq)]
enum ScanOutcome {
    /// A stop string fully matched. `emit_len` bytes precede the match and are
    /// safe to emit; the stop (`stop`) and everything after it are discarded.
    Hit { emit_len: usize, stop: String },
    /// No full match. `emit_len` bytes are safe to release now; the remaining
    /// `pending[emit_len..]` suffix is held back (a possible stop prefix).
    Partial { emit_len: usize },
}

/// Scan `pending` for stop strings.
///
/// - If any stop occurs, the EARLIEST-starting occurrence wins (ties broken by
///   stop-list order — probe-confirmed with `["</s", "</s>"]`). The tie only
///   changes the REPORTED stop identity; the emitted/truncated text is identical
///   either way (both cut at the same earliest position).
/// - Otherwise returns [`ScanOutcome::Partial`] whose `emit_len` withholds the
///   LONGEST proper trailing suffix of `pending` that is a prefix of some stop
///   string (cross-chunk boundary handling). Suffix candidates are taken only at
///   char boundaries, which — because both `pending` and every stop are valid
///   UTF-8 — is exactly the set of suffixes that can equal a stop prefix.
fn scan_stops(pending: &str, stops: &[String]) -> ScanOutcome {
    // Earliest full match across all stops.
    let mut best: Option<(usize, &str)> = None;
    for s in stops {
        if let Some(pos) = pending.find(s.as_str()) {
            match best {
                Some((bp, _)) if bp <= pos => {}
                _ => best = Some((pos, s.as_str())),
            }
        }
    }
    if let Some((pos, s)) = best {
        return ScanOutcome::Hit {
            emit_len: pos,
            stop: s.to_string(),
        };
    }

    // No full match: hold back the longest proper suffix that prefixes some stop.
    let len = pending.len();
    for start in 0..len {
        if !pending.is_char_boundary(start) {
            continue;
        }
        let suffix = &pending[start..];
        if stops
            .iter()
            .any(|st| st.len() > suffix.len() && st.starts_with(suffix))
        {
            return ScanOutcome::Partial { emit_len: start };
        }
    }
    ScanOutcome::Partial { emit_len: len }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Real in-repo Qwen3-30B tokenizer (byte-level BPE) — realistic split cases.
    fn tokenizer() -> Tokenizer {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("models/qwen3-30b-a3b-instruct-2507/tokenizer.json");
        Tokenizer::from_file(&path)
            .unwrap_or_else(|e| panic!("load tokenizer {}: {e}", path.display()))
    }

    /// Encode without special tokens, returning the raw id sequence.
    fn enc(tk: &Tokenizer, s: &str) -> Vec<u32> {
        tk.encode(s, false).unwrap().get_ids().to_vec()
    }

    /// Feed every id, then finish; return the fully concatenated emitted text and
    /// the first stop hit (if any).
    fn run(mut d: IncrementalDetok, ids: &[u32]) -> (String, Option<String>) {
        let mut out = String::new();
        let mut hit = None;
        for &id in ids {
            let r = d.push(id);
            out.push_str(&r.text);
            if r.stop.is_some() && hit.is_none() {
                hit = r.stop.clone();
            }
        }
        out.push_str(&d.finish());
        (out, hit)
    }

    // (a) A multi-byte codepoint split across two tokens must not mojibake and
    //     must arrive exactly once, when complete.
    #[test]
    fn multibyte_split_across_tokens_no_mojibake() {
        let tk = tokenizer();
        // 🥲 is byte-level BPE'd into >= 2 tokens; the first decodes to U+FFFD.
        let ids = enc(&tk, "🥲");
        assert!(
            ids.len() >= 2,
            "test needs a >=2-token emoji; got {ids:?} for 🥲"
        );

        let mut d = IncrementalDetok::new(tokenizer(), vec!["STOP".to_string()]);
        // First token: only the incomplete-codepoint half — nothing emitted.
        let r0 = d.push(ids[0]);
        assert_eq!(r0.text, "", "partial codepoint must be held back");
        assert!(
            !r0.text.contains('\u{FFFD}'),
            "no replacement char may leak"
        );
        // Feed the remaining tokens: the whole emoji arrives exactly once.
        let mut rest = String::new();
        for &id in &ids[1..] {
            rest.push_str(&d.push(id).text);
        }
        rest.push_str(&d.finish());
        assert_eq!(rest, "🥲");
    }

    // Same holdback logic must handle a 3-token codepoint (two U+FFFD stages).
    #[test]
    fn multibyte_split_three_tokens() {
        let tk = tokenizer();
        let ids = enc(&tk, "🫠");
        assert!(ids.len() >= 3, "expected 3-token emoji; got {ids:?}");
        let d = IncrementalDetok::new(tokenizer(), vec![]);
        let (out, hit) = run(d, &ids);
        assert_eq!(out, "🫠");
        assert_eq!(hit, None);
    }

    // (b) A stop string spanning multiple chunks: the first token ends with the
    //     stop's prefix. Nothing of the stop may leak; the stop is reported.
    #[test]
    fn stop_string_split_across_chunks() {
        let tk = tokenizer();
        // "\n\nHuman:" tokenizes to ["\n\n", "Human", ":"] — a genuine 3-chunk
        // stop. Precede it with real content to prove the split point.
        let stop = "\n\nHuman:";
        let ids = enc(&tk, &format!("Hello{stop}"));
        let d = IncrementalDetok::new(tokenizer(), vec![stop.to_string()]);
        let (out, hit) = run(d, &ids);
        assert_eq!(out, "Hello", "content before the stop, nothing of the stop");
        assert!(!out.contains("Human"), "no stop fragment may leak");
        assert_eq!(hit.as_deref(), Some(stop));
    }

    // (c) `<think>`/`</think>` are NOT stops and must pass through byte-exact,
    //     even though a stop ("<|im_end|>") shares their leading '<'.
    #[test]
    fn think_tags_pass_through_byte_exact() {
        let tk = tokenizer();
        let text = "<think>reasoning</think>answer";
        let ids = enc(&tk, text);
        let d = IncrementalDetok::new(tokenizer(), vec!["<|im_end|>".to_string()]);
        let (out, hit) = run(d, &ids);
        assert_eq!(out, text);
        assert_eq!(hit, None);
    }

    // (d) A stop prefix is held back, then RELEASED when the next token diverges
    //     from the stop (the held "\n\n" is emitted once "Hello" proves no match).
    #[test]
    fn stop_prefix_released_on_divergence() {
        let tk = tokenizer();
        let stop = "\n\nHuman:";
        // "\n\nHello": "\n\n" is a stop prefix, then diverges into "Hello".
        let ids = enc(&tk, "\n\nHello");
        let mut d = IncrementalDetok::new(tokenizer(), vec![stop.to_string()]);

        let mut emitted = String::new();
        let mut saw_holdback = false;
        for (i, &id) in ids.iter().enumerate() {
            let r = d.push(id);
            if i == 0 {
                // First chunk is the "\n\n" stop prefix — held back.
                assert_eq!(r.text, "", "stop prefix must be held");
                saw_holdback = true;
            }
            assert_eq!(r.stop, None, "must not falsely report a stop");
            emitted.push_str(&r.text);
        }
        emitted.push_str(&d.finish());
        assert!(saw_holdback);
        assert_eq!(emitted, "\n\nHello", "held prefix released on divergence");
    }

    // (e) finish() flushes a held-back stop prefix (no stop completed).
    #[test]
    fn finish_flushes_stop_prefix_holdback() {
        let tk = tokenizer();
        let stop = "\n\nHuman:";
        // "Hi\n\n": "Hi" emits, "\n\n" is held as a stop prefix until finish.
        let ids = enc(&tk, "Hi\n\n");
        let mut d = IncrementalDetok::new(tokenizer(), vec![stop.to_string()]);

        let mut streamed = String::new();
        for &id in &ids {
            streamed.push_str(&d.push(id).text);
        }
        assert!(
            !streamed.ends_with("\n\n"),
            "trailing stop prefix must be withheld while streaming, got {streamed:?}"
        );
        let flushed = d.finish();
        assert_eq!(flushed, "\n\n", "finish releases the held stop prefix");
        assert_eq!(format!("{streamed}{flushed}"), "Hi\n\n");
    }

    // finish() drops an incomplete multi-byte codepoint (drop policy).
    #[test]
    fn finish_drops_incomplete_utf8() {
        let tk = tokenizer();
        let ids = enc(&tk, "🥲");
        assert!(ids.len() >= 2);
        let mut d = IncrementalDetok::new(tokenizer(), vec![]);
        // Feed ONLY the first (partial-codepoint) token, then finish.
        let r = d.push(ids[0]);
        assert_eq!(r.text, "");
        let flushed = d.finish();
        assert_eq!(flushed, "", "incomplete codepoint dropped, never U+FFFD");
        assert!(!flushed.contains('\u{FFFD}'));
    }

    // Streaming a full emoji byte-for-byte equals the whole-string decode
    // (no leaked replacement chars anywhere in the stream).
    #[test]
    fn emoji_stream_matches_batch_decode() {
        let tk = tokenizer();
        let text = "done 🥲 ok";
        let ids = enc(&tk, text);
        let d = IncrementalDetok::new(tokenizer(), vec![]);
        let (out, _) = run(d, &ids);
        assert_eq!(out, text);
        assert!(!out.contains('\u{FFFD}'));
    }

    /// The single vocab token that decodes to the literal replacement char.
    /// Found by encoding "\u{FFFD}" (id 5691 in the 30B vocab); asserted to be a
    /// single token that round-trips to the replacement char.
    fn replacement_token(tk: &Tokenizer) -> u32 {
        let ids = enc(tk, "\u{FFFD}");
        assert_eq!(
            ids.len(),
            1,
            "expected a single U+FFFD vocab token, got {ids:?}"
        );
        let id = ids[0];
        assert_eq!(
            tk.decode(&[id], false).unwrap(),
            "\u{FFFD}",
            "token id {id} must decode to the replacement char"
        );
        id
    }

    // (Fix 8a) A SUSTAINED run of the literal U+FFFD vocab token must NOT stall
    // forever. Pre-fix, read_offset never advanced (window grew O(n), no emit —
    // head-of-line stall). The force-commit guard emits it as content (matching a
    // batch decode of the same ids) and keeps the window bounded.
    #[test]
    fn literal_replacement_token_force_commits_no_stall() {
        let tk = tokenizer();
        let fffd = replacement_token(&tk);

        let mut d = IncrementalDetok::new(tokenizer(), vec![]);
        let n = 8usize;
        let mut streamed = String::new();
        for _ in 0..n {
            streamed.push_str(&d.push(fffd).text);
        }

        // Window stayed bounded: read_offset advanced past the early tokens
        // instead of stalling at 0.
        assert!(
            d.read_offset > 0,
            "read_offset must advance (force-commit), else permanent O(n^2) stall"
        );
        assert!(
            d.tokens.len() - d.prefix_offset <= 2 * (MAX_UTF8_HOLDBACK_TOKENS + 1),
            "decode window must stay bounded (<= 2*(MAX+1)); got {}",
            d.tokens.len() - d.prefix_offset
        );

        let flushed = d.finish();
        let out = format!("{streamed}{flushed}");

        // Text WAS emitted (no permanent stall).
        assert!(!out.is_empty(), "force-commit must emit content, not stall");
        assert!(out.chars().all(|c| c == '\u{FFFD}'));
        // At least the first force-committed run (MAX+1 tokens) emitted.
        assert!(
            out.chars().count() >= MAX_UTF8_HOLDBACK_TOKENS + 1,
            "expected >= {} committed chars, got {}",
            MAX_UTF8_HOLDBACK_TOKENS + 1,
            out.chars().count()
        );
        // The emitted stream matches the batch decode of the same ids for the
        // force-committed prefix (a <=4-token U+FFFD tail may be dropped by
        // finish per the documented end-of-generation edge).
        let batch = tk.decode(&vec![fffd; n], false).unwrap();
        assert!(
            batch.starts_with(&out),
            "stream {out:?} must prefix batch decode {batch:?}"
        );
    }

    // (Fix 8b) A GENUINE multi-byte split (<= MAX_UTF8_HOLDBACK_TOKENS tokens)
    // must still hold back below the force-commit threshold: no premature commit,
    // no U+FFFD leak, and the whole codepoint arrives exactly once.
    #[test]
    fn genuine_split_holds_back_below_force_threshold() {
        let tk = tokenizer();
        let ids = enc(&tk, "🫠"); // 3-token (4-byte) emoji: within the threshold.
        assert!(
            ids.len() >= 3 && ids.len() <= MAX_UTF8_HOLDBACK_TOKENS,
            "need a genuine <= {}-token split; got {ids:?}",
            MAX_UTF8_HOLDBACK_TOKENS
        );

        let mut d = IncrementalDetok::new(tokenizer(), vec![]);
        // Every intermediate push holds back (no U+FFFD leaks, force NOT fired).
        for &id in &ids[..ids.len() - 1] {
            let r = d.push(id);
            assert_eq!(
                r.text, "",
                "partial codepoint must be held, not force-committed"
            );
            assert!(!r.text.contains('\u{FFFD}'));
        }
        // The final byte completes the codepoint — emitted exactly once, clean.
        let last = d.push(*ids.last().unwrap());
        let out = format!("{}{}", last.text, d.finish());
        assert_eq!(out, "🫠");
        assert!(!out.contains('\u{FFFD}'));
    }

    // ---- Pure-logic stop-scan tests (no tokenizer). ----

    #[test]
    fn scan_full_match_cuts_before_stop_earliest_wins() {
        let stops = vec!["END".to_string(), "STOP".to_string()];
        // Both present; "STOP" starts earlier ⇒ earliest wins.
        match scan_stops("abc STOP def END", &stops) {
            ScanOutcome::Hit { emit_len, stop } => {
                assert_eq!(&"abc STOP def END"[..emit_len], "abc ");
                assert_eq!(stop, "STOP");
            }
            o => panic!("expected Hit, got {o:?}"),
        }
    }

    #[test]
    fn scan_holds_longest_stop_prefix_suffix() {
        let stops = vec!["world".to_string()];
        // Trailing "wor" is the longest suffix that prefixes "world".
        match scan_stops("hello wor", &stops) {
            ScanOutcome::Partial { emit_len } => {
                assert_eq!(&"hello wor"[..emit_len], "hello ");
            }
            o => panic!("expected Partial, got {o:?}"),
        }
    }

    #[test]
    fn scan_no_holdback_when_no_prefix_overlap() {
        let stops = vec!["world".to_string()];
        match scan_stops("hello there", &stops) {
            ScanOutcome::Partial { emit_len } => assert_eq!(emit_len, "hello there".len()),
            o => panic!("expected Partial, got {o:?}"),
        }
    }

    #[test]
    fn scan_empty_stops_emits_everything() {
        match scan_stops("anything", &[]) {
            ScanOutcome::Partial { emit_len } => assert_eq!(emit_len, "anything".len()),
            o => panic!("expected Partial, got {o:?}"),
        }
    }

    #[test]
    fn scan_multibyte_stop_prefix_respects_char_boundary() {
        // Stop contains a multi-byte char; a trailing "café" prefixes "caféX".
        let stops = vec!["caféX".to_string()];
        match scan_stops("say café", &stops) {
            ScanOutcome::Partial { emit_len } => {
                assert_eq!(&"say café"[..emit_len], "say ");
            }
            o => panic!("expected Partial, got {o:?}"),
        }
    }

    #[test]
    fn empty_stop_strings_are_filtered() {
        let d = IncrementalDetok::new(tokenizer(), vec!["".to_string(), "x".to_string()]);
        assert_eq!(d.stops, vec!["x".to_string()]);
    }
}
