#![cfg(feature = "serve")]
//! S.1 GATE: minijinja chat-template rendering must be BYTE-IDENTICAL to HF
//! transformers `apply_chat_template` for both served models.
//!
//! Fixtures were captured with transformers 5.12.1 / python 3.12.3 (see
//! tests/fixtures/template/meta.json). For each model in
//! {qwen3-30b-a3b-instruct-2507, qwen3.6-35b-a3b} and each case in
//! tests/fixtures/template/inputs.json, render with add_generation_prompt=true
//! and compare bytes against tests/fixtures/template/<model>/<name>.txt.
//!
//! Run: `cargo test --features serve --test template_parity`

use std::path::PathBuf;

use qwen3_burn::serve::template::{ChatTemplate, OrderedJson};
use serde::Deserialize;

/// One parity case as stored in inputs.json. BOTH `messages` and `tools` use the
/// order-preserving `OrderedJson` so object key insertion order survives into
/// `tojson` (byte parity) — the 35B template `tojson`s message subtrees on the
/// tool-replay path (`tool_call.arguments | tojson`), so `messages` key order is
/// load-bearing, not just `tools`.
#[derive(Deserialize)]
struct Case {
    name: String,
    messages: OrderedJson,
    #[serde(default)]
    enable_thinking: Option<bool>,
    #[serde(default)]
    tools: Option<OrderedJson>,
}

const MODELS: [&str; 2] = ["qwen3-30b-a3b-instruct-2507", "qwen3.6-35b-a3b"];

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Print the first differing byte offset and the surrounding context of both
/// strings, then return a failure message.
fn first_diff_report(model: &str, name: &str, got: &str, want: &str) -> String {
    let gb = got.as_bytes();
    let wb = want.as_bytes();
    let mut off = 0;
    while off < gb.len() && off < wb.len() && gb[off] == wb[off] {
        off += 1;
    }
    let ctx = |s: &str, at: usize| {
        let start = at.saturating_sub(80);
        let end = (at + 80).min(s.len());
        // Snap to char boundaries so slicing never panics on multi-byte input.
        let start = (start..=at).find(|&i| s.is_char_boundary(i)).unwrap_or(at);
        let end = (at..=end)
            .rev()
            .find(|&i| s.is_char_boundary(i))
            .unwrap_or(at);
        s[start..end].to_string()
    };
    format!(
        "\n[{model} / {name}] MISMATCH at byte offset {off} \
         (got {} bytes, want {} bytes)\n\
         --- GOT  (…{off}…) ---\n{:?}\n\
         --- WANT (…{off}…) ---\n{:?}\n",
        gb.len(),
        wb.len(),
        ctx(got, off),
        ctx(want, off),
    )
}

#[test]
fn template_parity_all_models_all_cases() {
    let root = manifest_dir();
    let inputs_path = root.join("tests/fixtures/template/inputs.json");
    let inputs_text = std::fs::read_to_string(&inputs_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", inputs_path.display()));
    // Deserialize DIRECTLY from text (not via serde_json::Value) so OrderedJson
    // preserves tool-def key insertion order.
    let cases: Vec<Case> = serde_json::from_str(&inputs_text)
        .unwrap_or_else(|e| panic!("parse {}: {e}", inputs_path.display()));

    let mut failures: Vec<String> = Vec::new();
    let mut passed = 0usize;

    for model in MODELS {
        let model_dir = root.join("models").join(model);
        let tmpl = match ChatTemplate::from_model_dir(&model_dir) {
            Ok(t) => t,
            Err(e) => {
                failures.push(format!(
                    "[{model}] failed to load template from {}: {e}",
                    model_dir.display()
                ));
                continue;
            }
        };

        for case in &cases {
            let want_path = root
                .join("tests/fixtures/template")
                .join(model)
                .join(format!("{}.txt", case.name));
            let want = std::fs::read_to_string(&want_path)
                .unwrap_or_else(|e| panic!("read {}: {e}", want_path.display()));

            let got = match tmpl.render(
                &case.messages,
                case.tools.as_ref(),
                true, // add_generation_prompt
                case.enable_thinking,
            ) {
                Ok(s) => s,
                Err(e) => {
                    failures.push(format!("[{model} / {}] render error: {e}", case.name));
                    continue;
                }
            };

            if got == want {
                passed += 1;
                println!("PASS {model} / {}", case.name);
            } else {
                failures.push(first_diff_report(model, &case.name, &got, &want));
            }
        }
    }

    let total = MODELS.len() * cases.len();
    println!("\n{passed}/{total} cases byte-identical");
    if !failures.is_empty() {
        panic!(
            "template parity FAILED ({}/{total} passed):\n{}",
            passed,
            failures.join("\n")
        );
    }
}
