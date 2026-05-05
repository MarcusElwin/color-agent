"""Tier 4 reflection: a second model pass that critiques and re-ranks the
initial agent output. Defaults to same-model Sonnet; pass reviewer='claude-opus-4-7'
to A/B the stronger-reviewer trick."""

from __future__ import annotations

from typing import Any

from anthropic import Anthropic

from color_agent.agent import RETURN_HEX_LIST_TOOL, get_client
from color_agent.prompts import REFLECT_SYSTEM

DEFAULT_REVIEWER = "claude-sonnet-4-6"
MAX_TOKENS = 1024


def reflect(query: str, initial: dict[str, Any],
            reviewer: str = DEFAULT_REVIEWER,
            client: Anthropic | None = None) -> dict[str, Any]:
    """Single critique pass. Same payload shape as agent.call_agent.
    The reviewer must return >=5 candidates again — even if it agrees with all."""
    cli = client or get_client()
    cands_lines = "\n".join(
        f"  {i+1}. {c['hex']} ({c.get('name','')}) — {c.get('rationale','')}"
        for i, c in enumerate(initial.get("candidates", []))
    )
    user = (
        f"Description: {query}\n"
        f"Proposed candidates:\n{cands_lines}\n\n"
        "Review and return the final ranked list via return_hex_list."
    )

    resp = cli.messages.create(
        model=reviewer,
        max_tokens=MAX_TOKENS,
        system=REFLECT_SYSTEM,
        tools=[RETURN_HEX_LIST_TOOL],
        tool_choice={"type": "tool", "name": "return_hex_list"},
        messages=[{"role": "user", "content": user}],
    )
    for b in resp.content:
        if getattr(b, "type", None) == "tool_use" and b.name == "return_hex_list":
            return dict(b.input)
    raise RuntimeError("Reflection did not call return_hex_list")
