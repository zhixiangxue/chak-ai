"""Parallel Skill Calling Demo (mixed SkillBase + ClaudeSkill)

Demonstrates that multiple skills — both code-defined ``SkillBase`` and
file-defined ``ClaudeSkill`` — can be activated and exercised inside a
single parallel ``tool_calls`` batch without their per-skill state
colliding inside ``ToolManager``.

Background
----------
``ToolManager`` previously tracked the active skill in a single slot
(``_active_skill``) and a flat ``_selected_methods`` list. When the LLM
emitted a parallel batch that activated two different skills, the two
coroutines wrote ``self._active_skill = skill_tool`` in turn, the last
writer won, and the next round failed Stage-3 dispatch with
``Tool not found: <selected_method>``.

This example exercises the post-fix behaviour where state is per-skill
(``_active_skills``, ``_selected_methods``, ``_method_to_skill``) and
verifies that:
  * 2 ``SkillBase`` skills (A, B) — multi-stage progressive disclosure
  * 2 ``ClaudeSkill`` skills (C, D) — single-shot SKILL.md activation
can be activated in any combination within a single tool_calls batch.
"""

import asyncio
import os
import tempfile
import time
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

import dotenv

dotenv.load_dotenv()

import chak
from chak import Conversation
from chak.tools import SkillBase
from chak.tools.skills import ClaudeSkill


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


# ============================================================
# SkillBase variants (code-defined, multi-method)
# ============================================================

class SkillA(SkillBase):
    """Skill A: numeric helpers."""

    name = "skill_a"
    description = "Skill A — numeric helpers (square, double)"

    async def square(self, x: float) -> str:
        """Return x * x.

        Args:
            x: input number
        """
        print(f"[skill_a.square]  started  at {_ts()}  x={x}")
        await asyncio.sleep(1.2)
        print(f"[skill_a.square]  finished at {_ts()}")
        return f"[skill_a] square({x}) = {x * x}"

    async def double(self, x: float) -> str:
        """Return x * 2.

        Args:
            x: input number
        """
        print(f"[skill_a.double]  started  at {_ts()}  x={x}")
        await asyncio.sleep(1.2)
        print(f"[skill_a.double]  finished at {_ts()}")
        return f"[skill_a] double({x}) = {x * 2}"


class SkillB(SkillBase):
    """Skill B: string helpers."""

    name = "skill_b"
    description = "Skill B — string helpers (upper, reverse)"

    async def upper(self, text: str) -> str:
        """Uppercase the input string.

        Args:
            text: input text
        """
        print(f"[skill_b.upper]   started  at {_ts()}  text={text!r}")
        await asyncio.sleep(1.2)
        print(f"[skill_b.upper]   finished at {_ts()}")
        return f"[skill_b] upper({text!r}) = {text.upper()!r}"

    async def reverse(self, text: str) -> str:
        """Reverse the input string.

        Args:
            text: input text
        """
        print(f"[skill_b.reverse] started  at {_ts()}  text={text!r}")
        await asyncio.sleep(1.2)
        print(f"[skill_b.reverse] finished at {_ts()}")
        return f"[skill_b] reverse({text!r}) = {text[::-1]!r}"


# ============================================================
# ClaudeSkill variants (file-defined, SKILL.md format)
# ============================================================

# SKILL.md bodies are crafted so their effect is *visible in the final
# answer*: each one forces the assistant to emit a specific literal marker.
# If skill_c / skill_d failed to activate (or their state got clobbered by
# the SkillBase entries), the markers below would be missing — making the
# parallel-mixed-skill outcome trivially verifiable by eye.
SKILL_C_MD = """\
---
name: skill_c
description: Skill C — output framer. Wraps the final answer in fixed BEGIN/END markers so the caller can verify this skill was active.
---

# Skill C — Output Framer

When this skill is active, the assistant MUST wrap its FINAL answer
between these two literal marker lines, exactly as shown:

    ===== BEGIN [skill_c frame] =====
    <the body of the final answer goes here>
    ===== END   [skill_c frame] =====

Both marker lines are mandatory and MUST appear verbatim, on their own
lines, with no extra prose before BEGIN or after END.
"""

SKILL_D_MD = """\
---
name: skill_d
description: Skill D — audit signer. Appends a fixed audit signature line to the final answer listing all skills that participated.
---

# Skill D — Audit Signer

When this skill is active, the assistant MUST append, as the VERY LAST
line of the final answer (after any skill_c END marker), the literal
string:

    [skill_d audit] signed-by=skill_d · participants=skill_a,skill_b,skill_c,skill_d

This line MUST appear exactly once, verbatim, at the very end.
"""


def _materialise_claude_skill(stack: ExitStack, skill_md: str) -> ClaudeSkill:
    """Write a SKILL.md into a temp directory and load it as a ClaudeSkill."""
    tmp_dir = stack.enter_context(tempfile.TemporaryDirectory())
    (Path(tmp_dir) / "SKILL.md").write_text(skill_md, encoding="utf-8")
    return ClaudeSkill(tmp_dir)


# ============================================================
# Demo
# ============================================================

PROMPT = (
    "You have access to four skills: skill_a, skill_b, skill_c, skill_d.\n"
    "STEP 1 — In one parallel tool_calls batch, ACTIVATE ALL FOUR skills by\n"
    "  calling each one with NO arguments (empty object {}).\n"
    "STEP 2 — After they are activated, call:\n"
    "  * skill_a with method='skill_a-square' on x=5\n"
    "  * skill_a with method='skill_a-double' on x=7\n"
    "  * skill_b with method='skill_b-upper' on text='chak'\n"
    "  * skill_b with method='skill_b-reverse' on text='parallel'\n"
    "STEP 3 — Once Stage-3 method tools are exposed, call all four selected\n"
    "  methods (skill_a-square, skill_a-double, skill_b-upper, skill_b-reverse)\n"
    "  IN ONE PARALLEL tool_calls BATCH.\n"
    "STEP 4 — Produce the final answer. You MUST follow BOTH:\n"
    "  * skill_c — wrap the answer body in the BEGIN/END frame markers.\n"
    "  * skill_d — append the audit signature line as the very last line.\n"
    "  Inside the frame, list the four computed results on separate lines."
)


async def main() -> None:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("Error: DASHSCOPE_API_KEY not set")
        return

    with ExitStack() as stack:
        skill_a = SkillA()
        skill_b = SkillB()
        skill_c = _materialise_claude_skill(stack, SKILL_C_MD)
        skill_d = _materialise_claude_skill(stack, SKILL_D_MD)

        # Conversation auto-wraps SkillBase instances internally; pass them as-is.
        tools = [skill_a, skill_b, skill_c, skill_d]

        print("=" * 72)
        print("PARALLEL SKILLS DEMO (2x SkillBase + 2x ClaudeSkill)")
        print("=" * 72)
        print("Skills loaded:")
        print(f"  - SkillBase   {skill_a.name}")
        print(f"  - SkillBase   {skill_b.name}")
        print(f"  - ClaudeSkill {skill_c.name}")
        print(f"  - ClaudeSkill {skill_d.name}")
        print("-" * 72)
        print("Prompt:")
        print(PROMPT)
        print("-" * 72)

        conv = Conversation(
            provider_uri="qwen",
            model_uri="bailian/qwen-plus",
            api_key=api_key,
            tools=tools,
            # All skill methods here are async I/O — default ASYNCIO runner is fine.
            tool_executor=chak.ToolExecutor.ASYNCIO,
        )

        start = time.time()
        response = await conv.asend(PROMPT)
        elapsed = time.time() - start

        print("-" * 72)
        print(f"\nAssistant:\n{response.content}")
        print()
        print("=" * 72)
        print(f"Total wall time: {elapsed:.2f}s (LLM rounds + tool execution)")
        print("=" * 72)
        print(
            "Inspect the [skill.method] timestamps above. When skills are\n"
            "activated in the same tool_calls batch, their start timestamps\n"
            "should be within milliseconds of each other — proving that\n"
            "ToolManager dispatches them concurrently and that per-skill\n"
            "state (_active_skills / _selected_methods) does not collide\n"
            "across SkillBase and ClaudeSkill instances."
        )


if __name__ == "__main__":
    asyncio.run(main())
