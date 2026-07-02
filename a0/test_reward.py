#!/usr/bin/env python3
"""Unit tests for the A0 Manim reward harness. Pure stdlib; run: python3 a0/test_reward.py"""
import numpy as np
from manim_reward import staged_reward, extract_code, static_analysis

VALID = """
from manim import *

class SquareToCircle(Scene):
    def construct(self):
        sq = Square()
        self.play(Create(sq))
        self.play(Transform(sq, Circle()))
        self.wait()
"""

EMPTY_SCENE = """
from manim import Scene

class Boring(Scene):
    def construct(self):
        pass
"""

NO_MANIM = """
def helper(x):
    return x + 1

result = helper(41)
"""

SYNTAX_ERR = "class Broken(Scene)\n    def construct(self) self.play("

MALICIOUS_OS = "import os\nfrom manim import *\nos.system('rm -rf /tmp/x')\nclass S(Scene):\n    def construct(self): pass"
MALICIOUS_EXEC = "from manim import *\nexec('print(1)')\nclass S(Scene):\n    def construct(self): self.play(1)"
MALICIOUS_DUNDER = "from manim import *\nclass S(Scene):\n    def construct(self):\n        ().__class__.__bases__"

MARKDOWN = "Here is the animation:\n```python\n" + VALID + "\n```\nHope it helps!"


def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


def main():
    print("static / safety gate:")
    s_mal_os, d = staged_reward(MALICIOUS_OS, allow_render=False)
    check("malicious os.system -> 0.0 (rejected)", s_mal_os == 0.0 and d["reject"])
    s_exec, d = staged_reward(MALICIOUS_EXEC, allow_render=False)
    check("exec() -> 0.0 (rejected)", s_exec == 0.0 and d["reject"])
    s_dun, d = staged_reward(MALICIOUS_DUNDER, allow_render=False)
    check("dunder escape -> 0.0 (rejected)", s_dun == 0.0 and d["reject"])

    print("syntax / structure:")
    s_syn, _ = staged_reward(SYNTAX_ERR, allow_render=False)
    check("syntax error -> tiny (<0.05)", s_syn < 0.05)
    s_valid, dv = staged_reward(VALID, allow_render=False)
    check("valid scene -> high (>=0.6) without render", s_valid >= 0.6 and dv["has_construct"] and dv["has_anim_calls"])
    s_empty, de = staged_reward(EMPTY_SCENE, allow_render=False)
    check("empty scene -> mid, below valid (no anim-call credit)",
          0.4 <= s_empty < s_valid and not de["has_anim_calls"])
    s_nomanim, _ = staged_reward(NO_MANIM, allow_render=False)
    check("parses but no manim/scene -> low (<0.2)", s_nomanim < 0.2)

    print("markdown extraction:")
    code = extract_code(MARKDOWN)
    check("extracts fenced python", "class SquareToCircle" in code and "Hope it helps" not in code)
    s_md, _ = staged_reward(MARKDOWN, allow_render=False)
    check("markdown-wrapped scores same as raw", abs(s_md - s_valid) < 1e-9)

    print("GRPO variance guarantee (the collapse fix):")
    group = [VALID, EMPTY_SCENE, NO_MANIM, SYNTAX_ERR]   # a plausible G=4 rollout group
    rewards = np.array([staged_reward(t, allow_render=False)[0] for t in group])
    std = rewards.std(ddof=1)
    print(f"    group rewards = {rewards.tolist()}  std = {std:.4f}")
    check("varied completions -> nonzero reward std (advantage != 0)", std > 0.1)

    print("\nALL REWARD TESTS PASSED")


if __name__ == "__main__":
    main()
