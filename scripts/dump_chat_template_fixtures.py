#!/usr/bin/env python3
"""S.1 parity-gate golden dumps (docs/SERVE_PLAN.md).

Renders HF `apply_chat_template` for the gate matrix on BOTH served models and
writes the byte-exact outputs under tests/fixtures/template/. The Rust side
(src/serve/template.rs) must reproduce every dump byte-identically from the
same inputs (tests/fixtures/template/inputs.json).

Run:  <venv-with-transformers>/bin/python scripts/dump_chat_template_fixtures.py
"""

import json
import sys
from pathlib import Path

from transformers import AutoTokenizer
import transformers

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "tests" / "fixtures" / "template"

MODELS = {
    "qwen3-30b-a3b-instruct-2507": REPO / "models" / "qwen3-30b-a3b-instruct-2507",
    "qwen3.6-35b-a3b": REPO / "models" / "qwen3.6-35b-a3b",
}

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    },
}

# The gate matrix from docs/SERVE_PLAN.md S.1. `enable_thinking: None` means
# "do not pass the kwarg" (template default). All cases use
# add_generation_prompt=True (the server always appends the assistant stub).
CASES = [
    {
        "name": "single_user",
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "enable_thinking": None,
        "tools": None,
    },
    {
        "name": "system_multiturn",
        "messages": [
            {"role": "system", "content": "You are a terse assistant. Answer in one sentence."},
            {"role": "user", "content": "Name a prime number."},
            {"role": "assistant", "content": "Seven."},
            {"role": "user", "content": "And one bigger than that?"},
        ],
        "enable_thinking": None,
        "tools": None,
    },
    {
        "name": "thinking_on",
        "messages": [{"role": "user", "content": "How many r's are in strawberry?"}],
        "enable_thinking": True,
        "tools": None,
    },
    {
        "name": "thinking_off",
        "messages": [{"role": "user", "content": "How many r's are in strawberry?"}],
        "enable_thinking": False,
        "tools": None,
    },
    {
        # Exercises the [::-1] reverse-scan path: think blocks must be stripped
        # from all but the newest assistant turn.
        "name": "assistant_think_history",
        "messages": [
            {"role": "user", "content": "Is 91 prime?"},
            {
                "role": "assistant",
                "content": "<think>\n91 = 7 * 13, so no.\n</think>\n\nNo, 91 = 7 x 13.",
            },
            {"role": "user", "content": "What about 97?"},
        ],
        "enable_thinking": None,
        "tools": None,
    },
    {
        "name": "tool_defs",
        "messages": [{"role": "user", "content": "What's the weather in Paris?"}],
        "enable_thinking": None,
        "tools": [WEATHER_TOOL],
    },
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "inputs.json").write_text(json.dumps(CASES, indent=2, ensure_ascii=False) + "\n")

    meta = {"transformers": transformers.__version__, "python": sys.version.split()[0]}
    for model_name, model_dir in MODELS.items():
        tok = AutoTokenizer.from_pretrained(str(model_dir))
        mdir = OUT / model_name
        mdir.mkdir(exist_ok=True)
        for case in CASES:
            kwargs = {"tokenize": False, "add_generation_prompt": True}
            if case["enable_thinking"] is not None:
                kwargs["enable_thinking"] = case["enable_thinking"]
            if case["tools"] is not None:
                kwargs["tools"] = case["tools"]
            rendered = tok.apply_chat_template(case["messages"], **kwargs)
            path = mdir / f"{case['name']}.txt"
            path.write_bytes(rendered.encode("utf-8"))
            print(f"{model_name}/{case['name']}: {len(rendered)} chars")
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("meta:", meta)


if __name__ == "__main__":
    main()
