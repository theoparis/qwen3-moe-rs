#!/usr/bin/env python3
"""S.5(d): E2E greedy parity — server output must EQUAL the proven example's.

The reference is the greedy continuation of the shared raw prompt, produced by
the repo's proven example binaries (qwen35_generate for the 35B, vllm_infer for
the 30B). Asserts non-stream text == reference AND streamed-concatenated ==
non-stream (byte-identical).

Usage: greedy_parity.py <base_url> <max_tokens> <reference_file>
  reference_file holds the exact expected continuation text (raw bytes).
"""

import json
import sys
import urllib.request

BASE, MAX_TOKENS, REF_FILE = sys.argv[1], int(sys.argv[2]), sys.argv[3]
PROMPT = "The capital of France is"
reference = open(REF_FILE, "rb").read().decode()


def post(path: str, body: dict) -> tuple[int, str]:
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.status, r.read().decode()


def check(name: str, cond: bool, detail: str = ""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"\n    {detail}" if detail and not cond else ""))
    if not cond:
        sys.exit(1)


def main() -> None:
    with urllib.request.urlopen(BASE + "/v1/models") as r:
        model = json.loads(r.read())["data"][0]["id"]
    print(f"greedy parity gate: model={model} prompt={PROMPT!r} max_tokens={MAX_TOKENS}")

    body = {"model": model, "prompt": PROMPT, "temperature": 0, "max_tokens": MAX_TOKENS}
    st, tx = post("/v1/completions", body)
    check("non-stream 200", st == 200)
    non_stream = json.loads(tx)["choices"][0]["text"]
    check(
        "non-stream text == example reference",
        non_stream == reference,
        f"server={non_stream!r}\n    reference={reference!r}",
    )

    st, tx = post("/v1/completions", {**body, "stream": True})
    check("stream 200", st == 200)
    streamed = ""
    for block in tx.split("\n\n"):
        block = block.strip("\n")
        if not block.startswith("data: ") or block == "data: [DONE]":
            continue
        payload = json.loads(block[len("data: "):])
        if payload["choices"]:
            streamed += payload["choices"][0]["text"]
    check(
        "streamed-concatenated == non-stream",
        streamed == non_stream,
        f"streamed={streamed!r}\n    nonstream={non_stream!r}",
    )
    print("greedy parity gate: ALL PASS")


if __name__ == "__main__":
    main()
