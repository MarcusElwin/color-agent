"""Tiered router. The whole point: cheap deterministic lookups handle the
common case; the LLM agent only runs when the dictionaries can't help.

Routing inside Tier 4 preserves the original LLM-only spec:
  high overall confidence -> base agent only
  medium                  -> base + reflect
  low or brand-y query    -> consistency"""

from __future__ import annotations

import re
import time
from typing import Callable, Literal

from color_agent.agent import call_agent, to_candidates
from color_agent.consistency import consistent
from color_agent.normalize import normalize, parse_hex
from color_agent.reflect import reflect
from color_agent.tier1 import tier1
from color_agent.tier23 import hex_neighbors, tier2_or_3
from color_agent.tier_local import hex_neighbors_local, tier_local
from color_agent.types import Candidate, Result

ForceLayer = Literal["tier1", "tier2_3", "tier4_base", "tier4_reflect", "tier4_consistent"]
ProgressFn = Callable[[str], None]


def _noop(_msg: str) -> None:
    pass

# Hand-tuned brand/obscure hints — escalate Tier 4 to consistency on first sight.
_OBSCURE = re.compile(
    r"\b(pantone|pms|brand|logo|specific|exact|trademark|hex|rgb)\b", re.I
)


def _tier4(query: str, model: str = "claude-sonnet-4-6", k: int = 5,
           on_progress: ProgressFn = _noop,
           ) -> tuple[list[Candidate], str, bool, float | None]:
    """Run Tier 4 with sub-routing. Returns (candidates, tier_label, confident, spread)."""
    if _OBSCURE.search(query):
        on_progress(f"Tier 4 consistency • brand/obscure query • {model}")
        cands, spread = consistent(query, model=model, k=k)
        return cands, "4-consistent", cands[0].score >= 0.9, spread

    on_progress(f"Tier 4 base • web_search step • {model}")
    initial = call_agent(query, model=model)
    overall = initial.get("overall_confidence", "medium")

    if overall == "high":
        cands = to_candidates(initial, k=k)
        return cands, "4-base", True, None

    if overall == "medium":
        on_progress(f"Tier 4 reflection • model said medium • {model}")
        reviewed = reflect(query, initial)
        cands = to_candidates(reviewed, k=k)
        return cands, "4-reflect", True, None

    on_progress(f"Tier 4 consistency • model said low • N=5 samples")
    cands, spread = consistent(query, model=model, k=k)
    return cands, "4-consistent", cands[0].score >= 0.9, spread


def to_hex(query: str, k: int = 5,
           force: ForceLayer | None = None,
           model: str = "claude-sonnet-4-6",
           on_progress: ProgressFn | None = None) -> Result:
    progress = on_progress or _noop
    started = time.time()
    progress("Normalizing query")
    normalized = normalize(query)

    # Bare-hex input short-circuits to nearest-named neighbors. Try the
    # local 32k dataset first (no network); fall back to color.pizza only
    # if local somehow returns nothing; final fallback is the input hex
    # itself so the user always gets an answer.
    parsed_hex = parse_hex(query)
    if parsed_hex is not None:
        progress(f"Hex passthrough • local nearest-named for {parsed_hex}")
        cands = hex_neighbors_local(parsed_hex, k=k)
        if not cands:
            try:
                cands = hex_neighbors(parsed_hex, k=k)
            except Exception:
                pass
        if not cands:
            cands = [Candidate(hex=parsed_hex, name="(hex)", score=1.0,
                                source="parsed_hex")]
        return Result(
            query=query, normalized=normalized, candidates=cands,
            confident=True, tier="hex", spread=None,
            latency_ms=int((time.time() - started) * 1000),
        )

    if force == "tier1":
        progress("Tier 1 (forced) • CSS named lookup")
        cands = tier1(normalized, k=k) or []
        return Result(query, normalized, cands, bool(cands), "1",
                       latency_ms=int((time.time() - started) * 1000))

    if force == "tier2_3":
        progress("Tier 2/3 (forced) • color.pizza name search")
        result = tier2_or_3(normalized, k=k)
        if result:
            cands, tier, confident = result
            return Result(query, normalized, cands, confident, tier,
                           latency_ms=int((time.time() - started) * 1000))
        return Result(query, normalized, [], False, "miss",
                       latency_ms=int((time.time() - started) * 1000))

    if force == "tier4_base":
        progress(f"Tier 4 base (forced) • {model}")
        cands = to_candidates(call_agent(query, model=model), k=k)
        return Result(query, normalized, cands, True, "4-base",
                       latency_ms=int((time.time() - started) * 1000))

    if force == "tier4_reflect":
        progress(f"Tier 4 base + reflect (forced) • {model}")
        initial = call_agent(query, model=model)
        progress(f"Tier 4 reflection • {model}")
        cands = to_candidates(reflect(query, initial), k=k)
        return Result(query, normalized, cands, True, "4-reflect",
                       latency_ms=int((time.time() - started) * 1000))

    if force == "tier4_consistent":
        progress(f"Tier 4 consistency (forced) • N=5 samples • {model}")
        cands, spread = consistent(query, model=model, k=k)
        return Result(query, normalized, cands, cands[0].score >= 0.9,
                       "4-consistent", spread=spread,
                       latency_ms=int((time.time() - started) * 1000))

    # Auto-routing: cheap to expensive.
    progress("Tier 1 • CSS named lookup")
    t1 = tier1(normalized, k=k)
    if t1:
        return Result(query, normalized, t1, True, "1",
                       latency_ms=int((time.time() - started) * 1000))

    # Tier 2.5: in-process 32k dataset. If confident, return immediately —
    # color.pizza adds nothing and just costs a network round-trip (often a
    # 1.5s 403 retry on commercial-feel networks).
    progress("Tier 2.5 • local 32k name dictionary")
    tlocal = tier_local(normalized, k=k)
    if tlocal:
        cands, tier, confident = tlocal
        if confident:
            return Result(query, normalized, cands, True, tier,
                           latency_ms=int((time.time() - started) * 1000))
        local_fallback = (cands, tier, confident)
    else:
        local_fallback = None

    progress("Tier 2/3 • color.pizza name search")
    try:
        t23 = tier2_or_3(normalized, k=k)
    except Exception:
        t23 = None

    if t23:
        cands, tier, confident = t23
        if not confident:
            try:
                cands4, t4, conf4, spread = _tier4(query, model=model, k=k,
                                                    on_progress=progress)
                return Result(query, normalized, cands4, conf4, t4,
                               spread=spread,
                               latency_ms=int((time.time() - started) * 1000))
            except Exception:
                pass
        return Result(query, normalized, cands, confident, tier,
                       latency_ms=int((time.time() - started) * 1000))

    # color.pizza had nothing. Prefer the (non-confident) local fallback
    # over the LLM — much faster, and the dataset coverage is decent.
    if local_fallback is not None:
        cands, tier, confident = local_fallback
        return Result(query, normalized, cands, confident, tier,
                       latency_ms=int((time.time() - started) * 1000))

    cands4, t4, conf4, spread = _tier4(query, model=model, k=k,
                                        on_progress=progress)
    return Result(query, normalized, cands4, conf4, t4, spread=spread,
                   latency_ms=int((time.time() - started) * 1000))
