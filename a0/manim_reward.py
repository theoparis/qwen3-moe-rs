#!/usr/bin/env python3
"""
A0 verifiable Manim reward harness (reference of record for the Burn port).

Design (docs/GRPO_PLAN.md §2b fixes f/g/h):
  1. extract code from markdown fences
  2. STATIC analysis only (ast.parse) for the bulk of the score — NEVER execute model
     code to score structure. Reject dangerous imports/calls up front.
  3. DENSE staged partial credit (syntax -> manim import -> Scene subclass -> construct ->
     animation calls -> optional render) so a group of completions gets DIFFERENT scores
     and the GRPO advantage never collapses to zero.
  4. The ONLY stage that executes code is the optional `manim --dry_run` render, run in a
     hardened sandbox (new session/process-group, rlimits, scrubbed env, tmp cwd, network
     namespace via `unshare -n` when available, process-group kill on timeout). It degrades
     gracefully to a static score when manim is absent.

Pure stdlib (ast, subprocess, resource, tempfile). No torch/manim needed to score structure.
"""
import ast
import hashlib
import os
import re
import resource
import shutil
import signal
import subprocess
import tempfile

# Imports/among-calls that must never run, even sandboxed. Static reject -> reward 0.
_FORBIDDEN_MODULES = {
    "os", "sys", "subprocess", "shutil", "socket", "ctypes", "multiprocessing",
    "pty", "pickle", "marshal", "importlib", "builtins", "code", "pdb",
    "requests", "urllib", "http", "ftplib", "smtplib", "asyncio",
}
_ALLOWED_FROM_OS = set()  # nothing from os
_FORBIDDEN_CALLS = {"eval", "exec", "compile", "__import__", "open", "input", "globals", "locals", "getattr", "setattr"}

_FENCE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text):
    """Pull python out of markdown fences; fall back to the whole text."""
    blocks = _FENCE.findall(text)
    if blocks:
        # take the longest fenced block (usually the real program)
        return max(blocks, key=len).strip()
    return text.strip()


def static_analysis(code):
    """Parse + feature-extract WITHOUT executing. Returns (features dict, reject_reason|None)."""
    feats = dict(parses=False, imports_manim=False, has_scene=False, has_construct=False,
                 has_anim_calls=False, n_code_lines=0, n_classes=0)
    feats["n_code_lines"] = sum(1 for ln in code.splitlines() if ln.strip() and not ln.strip().startswith("#"))
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return feats, None  # doesn't parse -> not rejected, just low score
    feats["parses"] = True

    reject = None
    anim_methods = {"play", "add", "wait", "remove"}
    for node in ast.walk(tree):
        # forbidden imports
        if isinstance(node, ast.Import):
            for a in node.names:
                top = a.name.split(".")[0]
                if top in _FORBIDDEN_MODULES:
                    reject = f"forbidden import: {a.name}"
                if top == "manim":
                    feats["imports_manim"] = True
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".")[0]
            if top in _FORBIDDEN_MODULES and not (top == "os" and set(n.name for n in node.names) <= _ALLOWED_FROM_OS):
                reject = f"forbidden from-import: {node.module}"
            if top == "manim":
                feats["imports_manim"] = True
        # forbidden builtins / dunder access
        elif isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id in _FORBIDDEN_CALLS:
                reject = f"forbidden call: {f.id}"
            if isinstance(f, ast.Attribute) and f.attr in anim_methods:
                feats["has_anim_calls"] = True
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                reject = f"dunder access: {node.attr}"
        # Scene subclass + construct
        elif isinstance(node, ast.ClassDef):
            feats["n_classes"] += 1
            base_names = [b.id for b in node.bases if isinstance(b, ast.Name)]
            base_names += [b.attr for b in node.bases if isinstance(b, ast.Attribute)]
            if any("Scene" in b for b in base_names):
                feats["has_scene"] = True
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "construct":
                        feats["has_construct"] = True
    return feats, reject


def _preexec_limits(cpu_s=20, mem_bytes=2 * 1024**3, fsize=64 * 1024**2, nofile=64):
    """Run in the child before exec: new session + resource limits (fix g)."""
    os.setsid()  # new process group so we can killpg the whole tree (manim spawns latex/ffmpeg)
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
    resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))
    resource.setrlimit(resource.RLIMIT_NOFILE, (nofile, nofile))


def run_sandboxed(argv, timeout=30):
    """Execute argv in a hardened sandbox. Returns (rc, stdout, stderr). fix g."""
    workdir = tempfile.mkdtemp(prefix="manim_sbx_")
    env = {"PATH": "/usr/bin:/bin", "HOME": workdir, "TMPDIR": workdir, "PYTHONDONTWRITEBYTECODE": "1"}
    # network isolation when available (no firejail here; unshare -n is)
    if shutil.which("unshare"):
        argv = ["unshare", "-n", "--"] + list(argv)
    try:
        p = subprocess.Popen(argv, cwd=workdir, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, preexec_fn=_preexec_limits, text=True)
        try:
            out, err = p.communicate(timeout=timeout)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)  # kill the whole tree, no orphans
            except ProcessLookupError:
                pass
            p.communicate()
            return (-9, "", "TIMEOUT")
        return (rc, out[:8192], err[:8192])
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def manim_available():
    return shutil.which("manim") is not None


def staged_reward(text, allow_render=True):
    """Dense staged reward in [0,1]. Returns (score, detail dict). fix h: dense -> variance."""
    code = extract_code(text)
    feats, reject = static_analysis(code)
    detail = {"reject": reject, **feats, "rendered": False, "code_sha": hashlib.sha256(code.encode()).hexdigest()[:12]}

    if reject is not None:                       # safety gate: malicious -> hard 0 (fix f)
        return 0.0, detail
    if not feats["parses"]:                      # garbage / syntax error -> tiny
        return 0.02, detail

    # dense structural credit (sums to 0.7 without render) — guarantees intra-group spread
    score = 0.10                                  # parses
    if feats["imports_manim"]:   score += 0.10
    if feats["has_scene"]:       score += 0.20
    if feats["has_construct"]:   score += 0.20
    if feats["has_anim_calls"]:  score += 0.10
    score += min(feats["n_code_lines"], 10) * 0.001  # tiny length credit, capped

    # optional render stage (the only code execution) — gated + sandboxed
    if allow_render and feats["has_construct"] and manim_available():
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(code)
            scene_file = f.name
        try:
            rc, _o, _e = run_sandboxed(["python3", "-m", "manim", "--dry_run", scene_file], timeout=45)
            if rc == 0:
                score = min(1.0, score + 0.30)
                detail["rendered"] = True
        finally:
            os.unlink(scene_file)

    return round(min(score, 1.0), 4), detail


if __name__ == "__main__":
    import json
    import sys
    txt = sys.stdin.read() if not sys.stdin.isatty() else "no input"
    allow_render = "--no-render" not in sys.argv
    s, d = staged_reward(txt, allow_render=allow_render)
    if "--score-only" in sys.argv:
        # bare float on stdout, for the Rust reward harness (src/grpo/reward.rs)
        print(s)
    else:
        print(json.dumps({"score": s, "detail": d}, indent=2))
