//! Verifiable Manim reward — the Rust side.
//!
//! DRY by design: the reward LOGIC (markdown extraction, static-AST safety gate, dense staged
//! scoring, sandboxing) lives ONCE in the tested Python harness `a0/manim_reward.py` (unit-tested
//! by `a0/test_reward.py`). This Rust side shells out to it in `--score-only` mode and parses a
//! bare float, instead of reimplementing a Python AST analyzer in Rust. `manim` is a Python tool
//! anyway, so the reward path is inherently a subprocess; reusing the validated harness avoids a
//! second, divergent implementation.
//!
//! FAIL-SAFE: any failure (python missing, non-zero exit, timeout, unparseable output) yields a
//! reward of `0.0` — a reward function must never panic the training loop.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// A reward function over a batch of completion strings → scalar rewards (one per completion).
pub trait RewardFn: Send + Sync {
    /// Score each completion. Length of the result equals `completions.len()`.
    fn score_batch(&self, completions: &[String]) -> Vec<f32>;
}

/// Verifiable Manim reward backed by `a0/manim_reward.py`.
#[derive(Clone, Debug)]
pub struct ManimReward {
    python: String,
    script: PathBuf,
    /// Run the (slow, sandboxed) `manim --dry_run` render stage. Off by default (static scoring
    /// alone already gives dense partial credit and intra-group variance).
    allow_render: bool,
    /// Wall-clock budget for ONE completion's subprocess. On timeout the child is killed and the
    /// reward is `0.0`. Guards against a generated infinite loop or a hung harness freezing training.
    timeout: Duration,
}

impl Default for ManimReward {
    fn default() -> Self {
        ManimReward {
            python: "python3".to_string(),
            // default: repo-relative; override with `with_script` for a deployed binary. NOTE: a
            // wrong/missing path makes EVERY reward 0.0 (fail-safe) and trains KL-only — training
            // code should pass an ABSOLUTE path via `with_script`.
            script: PathBuf::from("a0/manim_reward.py"),
            allow_render: false,
            timeout: Duration::from_secs(10),
        }
    }
}

impl ManimReward {
    pub fn new() -> Self {
        Self::default()
    }

    /// Point at a specific `manim_reward.py` (e.g. an absolute path for a deployed binary).
    pub fn with_script<P: AsRef<Path>>(mut self, script: P) -> Self {
        self.script = script.as_ref().to_path_buf();
        self
    }

    /// Override the python interpreter (default `python3`).
    pub fn with_python(mut self, python: impl Into<String>) -> Self {
        self.python = python.into();
        self
    }

    /// Enable the sandboxed `manim --dry_run` render stage (slow; needs `manim` installed).
    ///
    /// WARNING: render EXECUTES the model-generated code. The Python harness has a static AST gate,
    /// but that is not a sandbox — for adversarial RL outputs, run the harness under OS-level
    /// isolation (container / seccomp / firejail, process-group kill, mem/CPU/no-network limits).
    pub fn with_render(mut self, allow: bool) -> Self {
        self.allow_render = allow;
        self
    }

    /// Per-completion subprocess timeout (default 10s). Raise it if `with_render` is enabled.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Score a single completion. Returns `0.0` on ANY error — spawn failure, non-zero exit,
    /// unparseable output, OR timeout (the child is killed). Never panics, never hangs.
    pub fn score_one(&self, completion: &str) -> f32 {
        let mut cmd = Command::new(&self.python);
        cmd.arg(&self.script).arg("--score-only");
        if !self.allow_render {
            cmd.arg("--no-render");
        }
        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());

        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(_) => return 0.0, // python missing / not executable
        };
        let mut stdout = child.stdout.take();
        // Write the completion to the harness's stdin, then close the pipe (drop). Completions are
        // short (<= a few KB for a 256-token gen), so this won't deadlock against the 64KB pipe
        // buffer even if the child stalls before reading.
        if let Some(mut stdin) = child.stdin.take() {
            let _ = stdin.write_all(completion.as_bytes());
        }
        // Poll for exit with a deadline; kill on timeout so a generated infinite loop or a hung
        // harness can never freeze the training step. (`--score-only` output is tiny, so reading
        // stdout after exit cannot deadlock.)
        let deadline = Instant::now() + self.timeout;
        loop {
            match child.try_wait() {
                Ok(Some(status)) => {
                    if !status.success() {
                        return 0.0;
                    }
                    let mut buf = String::new();
                    if let Some(mut so) = stdout.take() {
                        let _ = so.read_to_string(&mut buf);
                    }
                    return buf
                        .trim()
                        .parse::<f32>()
                        .map(|s| s.clamp(0.0, 1.0))
                        .unwrap_or(0.0);
                }
                Ok(None) => {
                    if Instant::now() >= deadline {
                        let _ = child.kill();
                        let _ = child.wait();
                        return 0.0; // timeout fail-safe
                    }
                    std::thread::sleep(Duration::from_millis(5));
                }
                Err(_) => return 0.0,
            }
        }
    }
}

impl RewardFn for ManimReward {
    fn score_batch(&self, completions: &[String]) -> Vec<f32> {
        // Sequential reference impl. Each call is a fast static check (no render by default).
        // PERF (deferred): bound a worker pool across the P*G batch — see docs/GRPO_PLAN.md §2b.
        completions.iter().map(|c| self.score_one(c)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn python_available() -> bool {
        Command::new("python3")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    fn harness() -> ManimReward {
        ManimReward::new()
            .with_script(concat!(env!("CARGO_MANIFEST_DIR"), "/a0/manim_reward.py"))
            .with_render(false)
    }

    const VALID: &str = "from manim import *\nclass S(Scene):\n    def construct(self):\n        self.play(Create(Square()))\n";
    const MALICIOUS: &str =
        "import os\nos.system('rm -rf /tmp/x')\nclass S(Scene):\n    def construct(self): pass\n";
    const GARBAGE: &str = "this is not python at all !!! (((";

    #[test]
    fn scores_match_python_harness() {
        if !python_available() {
            eprintln!("skip: python3 unavailable");
            return;
        }
        let r = harness();
        let scores = r.score_batch(&[VALID.into(), MALICIOUS.into(), GARBAGE.into()]);
        assert_eq!(scores.len(), 3);
        assert!(
            scores[0] >= 0.6,
            "valid scene should score high, got {}",
            scores[0]
        );
        assert_eq!(
            scores[1], 0.0,
            "malicious code must score 0 (safety gate), got {}",
            scores[1]
        );
        assert!(
            scores[2] < 0.05,
            "garbage should score ~0, got {}",
            scores[2]
        );
        // intra-group variance (GRPO needs this)
        assert!(scores[0] > scores[2], "rewards must spread across a group");
    }

    #[test]
    fn hung_subprocess_times_out_to_zero() {
        if !python_available() {
            eprintln!("skip: python3 unavailable");
            return;
        }
        // a fake harness that ignores stdin and sleeps far longer than the timeout
        let script = std::env::temp_dir().join("grpo_hang_reward_test.py");
        std::fs::write(&script, "import time\ntime.sleep(30)\n").unwrap();
        let r = ManimReward::new()
            .with_script(&script)
            .with_timeout(Duration::from_millis(300));
        let t = Instant::now();
        let s = r.score_one("anything");
        let _ = std::fs::remove_file(&script);
        assert_eq!(s, 0.0, "a hung subprocess must score 0.0");
        assert!(
            t.elapsed() < Duration::from_secs(5),
            "must return promptly after timeout, took {:?}",
            t.elapsed()
        );
    }

    #[test]
    fn missing_python_is_failsafe_zero() {
        // a non-existent interpreter must yield 0.0, never panic
        let r = ManimReward::new()
            .with_python("definitely-not-a-real-python-xyz")
            .with_script(concat!(env!("CARGO_MANIFEST_DIR"), "/a0/manim_reward.py"));
        assert_eq!(r.score_one(VALID), 0.0);
    }
}
