# Release Verification — qwen3-burn v0.3.0

**Verdict: READY-WITH-FIXES** — build/layout/honesty/figures all pass; one blocker remains
in the server curl examples (the in-parallel fix was applied but is still off by the quant
suffix), plus two minor cross-doc friction points.

Checks run: (1) `cargo check --lib` (2) layout (3) doc-facts-vs-code (4) honesty greps
(5) beginner walkthrough (6) cross-doc consistency. Evidence below.

## Findings

### 1. [BLOCKER] curl `"model"` field still exact-mismatches the loaded id → 404
- **Files:** `README.md:67,70-72`; `docs/GETTING_STARTED.md:172-173,182,187`
- **Code truth:** `src/serve/engine.rs:481` builds the 35B id as
  `format!("qwen3.6-35b-a3b-{quant_tag}")` (quant_tag ∈ bf16|fp8|nvfp4); 30B id is
  `"qwen3-30b-a3b"` (engine.rs:442). `validate_chat`/`validate_completion`
  (`handlers.rs:260,306`) do an EXACT `req.model != loaded_model_id` → 404 `model_not_found`.
- **Evidence:** The parallel fix changed the curl body to `"model":"qwen3.6-35b-a3b"` — but
  that literal (no quant suffix) is a value `model_id()` NEVER returns. README starts the
  server with `QUANT=bf16` → real id `qwen3.6-35b-a3b-bf16`; GETTING_STARTED starts with
  `QUANT=nvfp4` → real id `qwen3.6-35b-a3b-nvfp4`. Both curl bodies still 404 verbatim. The
  prose "e.g. `qwen3.6-35b-a3b`" (README:71, GS:173) cites an id that does not exist in code.
- **Fix:** Use the suffixed id matching each doc's launch command — README `qwen3.6-35b-a3b-bf16`,
  GETTING_STARTED `qwen3.6-35b-a3b-nvfp4` — and correct the "e.g." prose to the suffixed form.
  (The added "first curl /v1/models" guidance is correct and is the working escape hatch.)

### 2. [MINOR] GETTING_STARTED id explanation cites the wrong quant for its own server cmd
- **File:** `docs/GETTING_STARTED.md:156` vs `172-173`
- **Evidence:** The server is launched with `QUANT=nvfp4` (line 156), but the id note says
  "for `MODEL=qwen3.6-35b QUANT=bf16` that id is `qwen3.6-35b-a3b`" — wrong quant AND wrong
  (unsuffixed) id for the shown run; the actual `/v1/models` id there is `qwen3.6-35b-a3b-nvfp4`.
- **Fix:** Make the id note reference the `nvfp4` launch it actually shows: `qwen3.6-35b-a3b-nvfp4`.

### 3. [MINOR] CPU quick-start checkpoint path differs across the two docs
- **Files:** `README.md:49-52` vs `docs/GETTING_STARTED.md:41-42,94-98`
- **Evidence:** README's generate example reads `models/model.safetensors` /
  `models/tokenizer.json` (the fallback root path), while GETTING_STARTED downloads the 0.6B to
  `models/qwen3-0.6b/` and calls `--model models/qwen3-0.6b/model.safetensors`. A beginner who
  downloads per GETTING_STARTED then pastes the README command hits "file not found".
- **Fix:** Point README's quick-start at `models/qwen3-0.6b/model.safetensors` (and tokenizer)
  to match GETTING_STARTED's download target.

### 4. [NOTE] Build passes with one unused-import warning
- **Evidence:** `cargo check --lib` exit 0; `Finished dev profile`. One warning only:
  `unused_imports: Qwen3_5HybridCache, Qwen3_5HybridLayerCache` (lib). Cosmetic, not a blocker.
- **Fix (optional):** drop the unused import or `cargo fix --lib`.

## Passed checks (evidence)

- **BUILD (item 1):** `cargo check --lib` exit 0 (1 cosmetic warning, above).
- **LAYOUT (item 2):** `LICENSE` present; `models/` holds only `.gitkeep`; `vendor/cubecl` +
  `vendor/cubek` present; grep for `qwen3-burn-manin-grpo` across the four docs → 0 hits.
- **DOC FACTS (item 3):**
  - qwen-serve env table (GETTING_STARTED:141-149) matches `src/bin/qwen_serve.rs` exactly:
    HOST=0.0.0.0, PORT=8000, MODEL=qwen3.6-35b, QUANT=bf16, MODEL_DIR per-model default,
    T_MAX=4096, QUEUE_DEPTH=2, and the 30B default dir = `...-instruct-2507`.
  - Feature flags (README:125-131) match `Cargo.toml [features]`: `default=[]`, `cuda`, `train`,
    `serve` — names and gating correct.
  - HF repo names consistent across README & GETTING_STARTED: `Qwen/Qwen3-30B-A3B-Instruct-2507`,
    `Qwen/Qwen3.6-35B-A3B`, `nvidia/Qwen3.6-35B-A3B-NVFP4`, `Qwen/Qwen3-0.6B`.
  - All referenced examples exist: `generate`, `vllm_infer`, `qwen35_generate`, `grpo_train`,
    `grpo_cuda`; `scripts/serve_gates/*.py` and `a0/{grpo_reference,manim_reward}.py` exist.
  - Figures spot-checked (>6) against cited docs, all found: 0.73 / 19.38 / 21.03 / ≈45 tok/s
    (PERF_80TOKS_PLAN); 0.91/4.85/8.96/11.78 tok/s, 22.5GB, 89.9%/0.0374/97.8%
    (QUANT_FLASH_SPEC_PLAN); 12/12, 20.9, 7.5/9.9 tok/s, 72 tests (SERVE_PLAN); cosine 0.9999
    (MOE_PLAN); 5.85 tok/s (perf-gap-vs-prod).
- **HONESTY (item 4):** every "zero Python"/"end-to-end Rust" hit is an explicit negation
  (ARCHITECTURE:225, README:148); no hits for revolutionary/blazing/comprehensive/robust. The
  GRPO novelty claim is phrased as absence-of-evidence in all four docs.
- **CROSS-DOC (item 6):** version 0.3.0 consistent (Cargo.toml = RELEASE_NOTES "v0.3.0"); no doc
  claims a different release version. 30B captured tok/s handled honestly and each cites source:
  README "19.38 (re-run 21.03)"; RELEASE_NOTES "19.38→21.03 (notes SERVE_PLAN cites 20.9)";
  ARCHITECTURE "≈20.9 (PERF reports 19.38→21.03)"; GETTING_STARTED "20.9". No contradiction —
  the 19.38/21.03-vs-20.9 gap is explicitly reconciled, not hidden.

---

## Resolution (coordinator, post-verification)

- Finding 1 (BLOCKER): FIXED — curl `"model"` values now quant-suffixed per engine.rs model_id():
  README uses `qwen3.6-35b-a3b-bf16` (matches its QUANT=bf16 launch), GETTING_STARTED uses
  `qwen3.6-35b-a3b-nvfp4` (matches its QUANT=nvfp4 launch); both docs explain the
  /v1/models-first rule and the 30B id `qwen3-30b-a3b`. Verified by grep: zero unsuffixed ids remain.
- Finding 2 (minor): FIXED — GETTING_STARTED id note now cites the nvfp4 case it launches.
- Finding 3 (minor): FIXED — README CPU quick-start paths harmonized to models/qwen3-0.6b/.
- Finding 4 (note): accepted — pre-existing cosmetic unused-import warning in src/capture.rs.

VERDICT AFTER FIXES: READY
