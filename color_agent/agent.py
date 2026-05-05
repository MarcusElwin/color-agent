"""Tier 4 base agent. Two-step Anthropic call:
  1. tool_choice=auto so the model can call web_search server-side.
  2. tool_choice forced to return_hex_list to extract structured candidates.

Forcing return_hex_list in step 1 would prefill the assistant turn and prevent
web_search from running first — that's the single biggest gotcha. Don't merge
these two calls.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from anthropic import Anthropic

from color_agent.prompts import BASE_AGENT_SYSTEM, CONSISTENCY_USER_TEMPLATE
from color_agent.types import Candidate

DEFAULT_MODEL = "claude-sonnet-4-6"
FAST_MODEL = "claude-haiku-4-5"
MAX_TOKENS = 1024
MAX_AGENT_ITERATIONS = 6  # safety cap on the auto-tool loop


@dataclass(frozen=True)
class ModelConfig:
    model: str
    web_search_tool: dict[str, Any]
    requires_code_execution: bool


def _web_search_basic(max_uses: int) -> dict[str, Any]:
    return {
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": max_uses,
    }


def _web_search_advanced(max_uses: int) -> dict[str, Any]:
    return {
        "type": "web_search_20260209",
        "name": "web_search",
        "max_uses": max_uses,
    }


def model_config(model: str, max_uses: int = 3) -> ModelConfig:
    """Pick the right web_search version for a given model id.
    Sonnet 4.6 / Opus 4.7 use the dynamic-filter version (which auto-injects
    code_execution server-side — don't add it to the tools array, the API
    rejects duplicate names). Haiku stays on basic."""
    if model.startswith("claude-haiku"):
        return ModelConfig(
            model=model,
            web_search_tool=_web_search_basic(max_uses),
            requires_code_execution=False,
        )
    return ModelConfig(
        model=model,
        web_search_tool=_web_search_advanced(max_uses),
        requires_code_execution=False,
    )


RETURN_HEX_LIST_TOOL: dict[str, Any] = {
    "name": "return_hex_list",
    "description": "Return ranked color candidates for the user's description. "
                   "First candidate is the best guess; remaining candidates "
                   "span plausible alternatives.",
    "input_schema": {
        "type": "object",
        "required": ["candidates", "overall_confidence", "source"],
        "properties": {
            "candidates": {
                "type": "array",
                "minItems": 5,
                "maxItems": 8,
                "items": {
                    "type": "object",
                    "required": ["hex", "name", "confidence", "rationale"],
                    "properties": {
                        "hex": {
                            "type": "string",
                            "pattern": "^#[0-9A-Fa-f]{6}$",
                        },
                        "name": {"type": "string"},
                        "confidence": {"enum": ["high", "medium", "low"]},
                        "rationale": {"type": "string", "maxLength": 200},
                    },
                },
            },
            "overall_confidence": {"enum": ["high", "medium", "low"]},
            "source": {"enum": ["knowledge", "web_search"]},
        },
    },
}


CONFIDENCE_TO_SCORE = {"high": 0.9, "medium": 0.7, "low": 0.5}


_client: Anthropic | None = None
_dotenv_loaded = False


def _ensure_dotenv_loaded() -> None:
    """Load .env from cwd (or any parent) on first client construction.
    No-op if python-dotenv isn't installed or no .env exists. Existing env
    vars take precedence — we don't override what the shell already set."""
    global _dotenv_loaded
    if _dotenv_loaded:
        return
    _dotenv_loaded = True
    try:
        from dotenv import find_dotenv, load_dotenv
    except ImportError:
        return
    path = find_dotenv(usecwd=True)
    if path:
        load_dotenv(path, override=False)


def get_client() -> Anthropic:
    global _client
    if _client is None:
        _ensure_dotenv_loaded()
        _client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _client


def _extract_text_blocks(content: list[Any]) -> str:
    parts = []
    for b in content:
        if getattr(b, "type", None) == "text":
            parts.append(getattr(b, "text", ""))
    return "\n".join(parts).strip()


def _gather_search_context(query: str, cfg: ModelConfig,
                           client: Anthropic) -> tuple[list[dict], list[Any]]:
    """Step 1: auto-tool loop. Lets the model run web_search 0..max_uses times.
    Returns (transcript, last_assistant_content) so step 2 can append it."""
    tools: list[dict[str, Any]] = [cfg.web_search_tool]

    messages: list[dict] = [
        {"role": "user", "content":
         CONSISTENCY_USER_TEMPLATE.format(query=query)}
    ]
    last_content: list[Any] = []
    for _ in range(MAX_AGENT_ITERATIONS):
        resp = client.messages.create(
            model=cfg.model,
            max_tokens=MAX_TOKENS,
            system=BASE_AGENT_SYSTEM,
            tools=tools,
            tool_choice={"type": "auto"},
            messages=messages,
        )
        last_content = list(resp.content)
        if resp.stop_reason in ("end_turn", "stop_sequence"):
            break
        if resp.stop_reason == "pause_turn":
            messages.append({"role": "assistant", "content": last_content})
            continue
        # Server-side tools (web_search) auto-execute and stay in the same response;
        # if the loop reports tool_use without an unresolved client tool, we're done.
        break
    return messages, last_content


def call_agent(query: str, model: str = DEFAULT_MODEL,
               max_uses: int = 3, client: Anthropic | None = None,
               temperature: float | None = None) -> dict[str, Any]:
    """Tier 4 base. Returns the parsed return_hex_list payload as a dict."""
    cli = client or get_client()
    cfg = model_config(model, max_uses=max_uses)

    messages, search_content = _gather_search_context(query, cfg, cli)

    # Step 2: append step-1 result, force the structured return.
    if search_content:
        messages.append({"role": "assistant", "content": search_content})
        messages.append({"role": "user", "content":
                          "Now return your final ranked hex list via "
                          "return_hex_list."})

    extra: dict[str, Any] = {}
    if temperature is not None:
        extra["temperature"] = temperature

    final = cli.messages.create(
        model=cfg.model,
        max_tokens=MAX_TOKENS,
        system=BASE_AGENT_SYSTEM,
        tools=[RETURN_HEX_LIST_TOOL],
        tool_choice={"type": "tool", "name": "return_hex_list"},
        messages=messages,
        **extra,
    )

    for b in final.content:
        if getattr(b, "type", None) == "tool_use" and b.name == "return_hex_list":
            return dict(b.input)

    raise RuntimeError("Tier 4 agent did not call return_hex_list")


def to_candidates(payload: dict[str, Any], k: int = 5) -> list[Candidate]:
    """Convert return_hex_list payload to ranked Candidates."""
    cands_in = payload.get("candidates", [])
    out: list[Candidate] = []
    for c in cands_in[:max(k, 5)]:
        score = CONFIDENCE_TO_SCORE.get(c.get("confidence", "medium"), 0.7)
        out.append(Candidate(
            hex=c["hex"].upper(),
            name=c.get("name", ""),
            score=score,
            source=f"llm_{payload.get('source', 'knowledge')}",
        ))
    return out
