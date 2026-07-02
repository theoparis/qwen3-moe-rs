#!/usr/bin/env python3
"""S.5(g)+(h): sustained mixed smoke + error paths + FIFO concurrency.

(g) 20 sequential mixed requests (greedy/sampled/cancel-mid-stream, short+long
    prompts): stable completion tok/s, server RSS flat across requests (the
    fresh-cache leak check — unified memory on GB10 makes RSS the observable).
(h) error paths: 400 length overflow, 400 template failure (malformed tool
    schema), 400 unsupported params (n>1, logprobs), 429 queue full (fired
    while a long request decodes), 2 concurrent clients (FIFO order, both
    complete).

Usage: sustained_and_errors.py <server_pid> [base]   (base default http://127.0.0.1:8000)
"""

import concurrent.futures as cf
import json
import sys
import time
import urllib.error
import urllib.request

PID = int(sys.argv[1])
BASE = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8000"


def rss_kb() -> int:
    with open(f"/proc/{PID}/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    return 0


def post(path: str, body: dict, timeout: float = 600.0, read_bytes: int | None = None):
    """POST json; returns (status, body_text). read_bytes truncates (cancel case)."""
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if read_bytes is not None:
                data = r.read(read_bytes)  # then close early = cancel-mid-stream
                return r.status, data.decode(errors="replace")
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def check(name: str, cond: bool, detail: str = ""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        sys.exit(1)


def model_id() -> str:
    with urllib.request.urlopen(BASE + "/v1/models") as r:
        return json.loads(r.read())["data"][0]["id"]


def main() -> None:
    import os
    model = model_id()
    skip_sustained = os.environ.get("SKIP_SUSTAINED") == "1"
    long_prompt = "Explain, step by step and in detail, how a transformer language model works. " * 8
    cases = []
    for i in range(20):
        kind = i % 4
        if kind == 0:
            cases.append(("greedy-short", {"messages": [{"role": "user", "content": "Count: 1 2 3"}], "temperature": 0, "max_tokens": 24}, None))
        elif kind == 1:
            cases.append(("sampled-short", {"messages": [{"role": "user", "content": "Name a color."}], "temperature": 0.8, "seed": i, "max_tokens": 24}, None))
        elif kind == 2:
            cases.append(("greedy-long", {"messages": [{"role": "user", "content": long_prompt}], "temperature": 0, "max_tokens": 48}, None))
        else:
            cases.append(("cancel-mid-stream", {"messages": [{"role": "user", "content": "Write a long story."}], "temperature": 0, "max_tokens": 512, "stream": True}, 400))

    if skip_sustained:
        print("(g) SKIPPED (SKIP_SUSTAINED=1)")
        cases = []
    print(f"(g) sustained smoke: {len(cases)} mixed requests against {model}")
    rss0 = rss_kb()
    rates = []
    rss_tail = []
    for n, (name, body, read_bytes) in enumerate(cases):
        body["model"] = model
        t0 = time.time()
        status, text = post("/v1/chat/completions", body, read_bytes=read_bytes)
        dt = time.time() - t0
        assert status == 200, f"{name}: HTTP {status}: {text[:200]}"
        if read_bytes is None and not body.get("stream"):
            u = json.loads(text)["usage"]
            rate = u["completion_tokens"] / dt if dt > 0 else 0
            rates.append(rate)
            rss_tail.append(rss_kb())
            print(f"  {n:2d} {name:16s} {u['completion_tokens']:3d} tok in {dt:5.1f}s = {rate:5.2f} tok/s rss={rss_kb()} kB")
        else:
            print(f"  {n:2d} {name:16s} (stream cancelled after {read_bytes}B) {dt:5.1f}s rss={rss_kb()} kB")
        time.sleep(0.3)  # let a cancelled engine request wind down
    rss1 = rss_kb()
    if skip_sustained:
        rss0 = rss1  # no sustained data to assert on
    # Stability is PER CLASS — greedy/sampled/long have different inherent rates
    # (greedy device-argmax vs host-sampled differ ~7x on the 35B eager path).
    from collections import defaultdict
    by_class = defaultdict(list)
    for (name, _, rb), r in zip([c for c in cases if c[2] is None and not c[1].get("stream")], rates):
        by_class[name].append(r)
    for name, rs in by_class.items():
        steady = rs[1:] if len(rs) > 2 else rs  # first request may be cache-cold
        med = sorted(steady)[len(steady) // 2]
        check(f"tok/s stable within class {name} (steady within 20% of median {med:.2f})",
              all(abs(r - med) / med < 0.2 for r in steady),
              f"rates={['%.2f' % r for r in rs]}")
    # Leak check: steady-state RSS flat. The first cycle(s) of request shapes
    # legitimately grow RSS (CubeCL per-shape JIT kernels + autotune caches —
    # one-time, verified to plateau); a per-request leak keeps climbing across
    # REPEATED identical shapes. So assert growth over the SECOND HALF < 2%.
    if rss_tail:
        mid = rss_tail[len(rss_tail) // 2]
        growth = (rss1 - mid) / max(mid, 1)
        check(
            f"steady-state RSS flat (mid-run {mid} kB → end {rss1} kB; cold-start {rss0} kB excluded as per-shape warmup)",
            growth < 0.02,
            f"steady growth {growth * 100:.1f}%",
        )
    if rss_tail:
        tail = rss_tail[-5:]
        check("RSS plateaus over the last 5 requests (<1% spread)",
              (max(tail) - min(tail)) / max(min(tail), 1) < 0.01, f"tail={tail}")

    print("(h) error paths:")
    st, tx = post("/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": "hi " * 6000}], "max_tokens": 4096})
    check("length overflow → 400 + envelope", st == 400 and "error" in json.loads(tx))
    st, tx = post("/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": "hi"}], "tools": [{"type": "function"}]})
    check("malformed tool schema → 400 (template failure)", st == 400, f"got {st}")
    st, tx = post("/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": "hi"}], "n": 2})
    check("n>1 → 400", st == 400)
    st, tx = post("/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": "hi"}], "logprobs": True})
    check("logprobs → 400", st == 400)
    st, tx = post("/v1/chat/completions", {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]})
    check("wrong model → 404 model_not_found", st == 404 and "model_not_found" in tx)

    # 429: saturate — one long decode + queue_depth fillers, then one more.
    print("  probing 429 (queue full under a long decode)...")
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(post, "/v1/chat/completions",
                          {"model": model, "messages": [{"role": "user", "content": f"Write a paragraph about topic {i}."}], "temperature": 0, "max_tokens": 256})
                for i in range(6)]
        time.sleep(1.0)
        statuses = [f.result()[0] for f in futs]
    check("saturation yields ≥1 429 and ≥1 200", 429 in statuses and 200 in statuses, f"statuses={statuses}")

    # FIFO: two concurrent clients; both complete; first-submitted finishes first.
    def timed(tag: str):
        t0 = time.time()
        st, tx = post("/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": f"Say only the word {tag}."}], "temperature": 0, "max_tokens": 8})
        return tag, st, time.time() - t0, time.time()

    with cf.ThreadPoolExecutor(max_workers=2) as ex:
        f1 = ex.submit(timed, "alpha")
        time.sleep(0.2)
        f2 = ex.submit(timed, "beta")
        r1, r2 = f1.result(), f2.result()
    check("both concurrent clients complete 200", r1[1] == 200 and r2[1] == 200)
    check("FIFO order preserved (first submitted finished first)", r1[3] < r2[3],
          f"alpha done@{r1[3]:.2f} beta done@{r2[3]:.2f}")
    print("(g)+(h): ALL PASS")


if __name__ == "__main__":
    main()
