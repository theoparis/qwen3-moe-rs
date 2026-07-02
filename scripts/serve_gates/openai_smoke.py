#!/usr/bin/env python3
"""S.5(f): python `openai` client smoke against a live qwen-serve.

chat (non-stream + stream), legacy completions, /v1/models — through the real
SDK, which exercises header/shape strictness a hand-rolled client would miss.
Also asserts stream-concat == non-stream greedy text (S.5(d) consistency leg).

Usage: openai_smoke.py [base_url]   (default http://127.0.0.1:8000/v1)
"""

import sys

from openai import OpenAI

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000/v1"
client = OpenAI(base_url=BASE, api_key="unused")


def check(name: str, cond: bool, detail: str = ""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        sys.exit(1)


def main() -> None:
    models = client.models.list()
    model = models.data[0].id
    print(f"openai client smoke against model={model}")
    check("models.list returns exactly the loaded model", len(models.data) == 1)

    msgs = [{"role": "user", "content": "In one short sentence: why is the sky blue?"}]

    r = client.chat.completions.create(model=model, messages=msgs, temperature=0, max_tokens=48)
    check("non-stream chat returns a choice", len(r.choices) == 1)
    nonstream_content = r.choices[0].message.content or ""
    nonstream_reasoning = getattr(r.choices[0].message, "reasoning_content", None) or ""
    check("finish_reason set", r.choices[0].finish_reason in ("stop", "length"))
    check("usage totals add up", r.usage.total_tokens == r.usage.prompt_tokens + r.usage.completion_tokens)
    print(f"    content: {nonstream_content[:80]!r}")

    stream = client.chat.completions.create(
        model=model, messages=msgs, temperature=0, max_tokens=48,
        stream=True, stream_options={"include_usage": True},
    )
    s_content, s_reasoning, saw_role, saw_usage, finish = "", "", False, False, None
    for chunk in stream:
        if not chunk.choices:
            saw_usage = chunk.usage is not None
            continue
        d = chunk.choices[0].delta
        if d.role == "assistant":
            saw_role = True
        if d.content:
            s_content += d.content
        rc = getattr(d, "reasoning_content", None)
        if rc:
            s_reasoning += rc
        if chunk.choices[0].finish_reason:
            finish = chunk.choices[0].finish_reason
    check("stream first-chunk role seen", saw_role)
    check("stream finish_reason set", finish in ("stop", "length"))
    check("stream usage chunk seen (include_usage)", saw_usage)
    check("greedy stream content == non-stream content", s_content == nonstream_content,
          f"stream={s_content[:60]!r} nonstream={nonstream_content[:60]!r}")
    check("greedy stream reasoning == non-stream reasoning", s_reasoning == nonstream_reasoning)

    c = client.completions.create(model=model, prompt="The capital of France is", temperature=0, max_tokens=16)
    check("legacy completions returns text", len(c.choices[0].text) > 0)
    print(f"    completion: {c.choices[0].text[:80]!r}")

    print("openai client smoke: ALL PASS")


if __name__ == "__main__":
    main()
