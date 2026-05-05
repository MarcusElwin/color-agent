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
from color_agent.color_pizza import (
    ColorPizzaPermanentError, ColorPizzaTransientError,
)
from color_agent.consistency import consistent
from color_agent.normalize import normalize, parse_hex
from color_agent.reflect import reflect
from color_agent.tier1 import tier1
from color_agent.tier23 import hex_neighbors, tier2_or_3
from color_agent.tier_local import tier_local
from color_agent.types import Candidate, Result, TraceStep

ForceLayer = Literal["tier1", "tier2_3", "tier4_base", "tier4_reflect", "tier4_consistent"]
ProgressFn = Callable[[str], None]
EventFn = Callable[[dict], None]


def _noop(_msg: str) -> None:
    pass


def _noop_event(_evt: dict) -> None:
    pass


class _Trace:
    """Tiny helper that emits structured trace events AND collects them for the
    final Result. CLI renders them live; result panel renders them post-hoc;
    --json output ships them so callers can audit the routing path."""

    def __init__(self, on_event: EventFn, on_progress: ProgressFn):
        self.on_event = on_event
        self.on_progress = on_progress
        self._t0_step: float | None = None
        self._current: str | None = None
        self._current_tier: str | None = None
        self.steps: list[TraceStep] = []

    def step(self, name: str, **kw) -> None:
        if self._current is not None:
            self.end(outcome="(skipped)")
        self._current = name
        self._current_tier = kw.get("tier")
        self._t0_step = time.time()
        self.on_progress(name)
        self.on_event({"type": "step_start", "name": name, **kw})

    def end(self, outcome: str, **kw) -> None:
        if self._current is None:
            return
        dt_ms = int((time.time() - (self._t0_step or time.time())) * 1000)
        self.steps.append(TraceStep(
            tier=self._current_tier or "?",
            name=self._current,
            outcome=outcome,
            duration_ms=dt_ms,
            confident=kw.get("confident"),
            candidates=kw.get("candidates"),
            top_hex=kw.get("top_hex"),
        ))
        self.on_event({
            "type": "step_end", "name": self._current,
            "outcome": outcome, "duration_ms": dt_ms, **kw,
        })
        self._current = None
        self._current_tier = None
        self._t0_step = None

# Hand-tuned brand/obscure hints — escalate Tier 4 to consistency on first sight.
_OBSCURE = re.compile(
    r"\b(pantone|pms|brand|logo|specific|exact|trademark|hex|rgb)\b", re.I
)


def _tier4(query: str, model: str = "claude-sonnet-4-6", k: int = 5,
           on_progress: ProgressFn = _noop, trace: _Trace | None = None,
           ) -> tuple[list[Candidate], str, bool, float | None]:
    """Run Tier 4 with sub-routing. Returns (candidates, tier_label, confident, spread)."""
    t = trace or _Trace(_noop_event, on_progress)

    if _OBSCURE.search(query):
        t.step(f"Tier 4 consistency • brand/obscure query • {model}",
                tier="4-consistent", model=model)
        cands, spread = consistent(query, model=model, k=k)
        confident = cands[0].score >= 0.9
        t.end(outcome="reasoned across N=5 samples",
               candidates=len(cands), confident=confident, spread=spread,
               top_hex=cands[0].hex)
        return cands, "4-consistent", confident, spread

    t.step(f"Tier 4 base • web_search step • {model}",
            tier="4-base", model=model)
    initial = call_agent(query, model=model)
    overall = initial.get("overall_confidence", "medium")
    cands_initial = to_candidates(initial, k=k)
    t.end(outcome=f"model self-reported {overall}",
           candidates=len(cands_initial), confident=(overall == "high"),
           top_hex=cands_initial[0].hex if cands_initial else None,
           model_confidence=overall)

    if overall == "high":
        return cands_initial, "4-base", True, None

    if overall == "medium":
        t.step(f"Tier 4 reflection • model said medium • {model}",
                tier="4-reflect", model=model)
        reviewed = reflect(query, initial)
        cands = to_candidates(reviewed, k=k)
        t.end(outcome="reflected & re-ranked",
               candidates=len(cands), confident=True,
               top_hex=cands[0].hex if cands else None)
        return cands, "4-reflect", True, None

    t.step(f"Tier 4 consistency • model said low • N=5 samples",
            tier="4-consistent", model=model)
    cands, spread = consistent(query, model=model, k=k)
    confident = cands[0].score >= 0.9
    t.end(outcome="reasoned across N=5 samples",
           candidates=len(cands), confident=confident, spread=spread,
           top_hex=cands[0].hex)
    return cands, "4-consistent", confident, spread


def to_hex(query: str, k: int = 5,
           force: ForceLayer | None = None,
           model: str = "claude-sonnet-4-6",
           on_progress: ProgressFn | None = None,
           on_event: EventFn | None = None) -> Result:
    progress = on_progress or _noop
    event = on_event or _noop_event
    trace = _Trace(event, progress)
    result = _to_hex_inner(query, k, force, model, trace, progress)
    result.trace = list(trace.steps)
    return result


def _to_hex_inner(query: str, k: int, force: ForceLayer | None,
                   model: str, trace: "_Trace",
                   progress: ProgressFn) -> Result:
    started = time.time()
    progress("Normalizing query")
    normalized = normalize(query)

    # Bare-hex input short-circuits to nearest-named neighbors.
    parsed_hex = parse_hex(query)
    if parsed_hex is not None:
        trace.step(f"Hex passthrough • looking up nearest names for {parsed_hex}",
                    tier="hex")
        try:
            cands = hex_neighbors(parsed_hex, k=k)
            confident = bool(cands)
        except Exception:
            cands = [Candidate(hex=parsed_hex, name="(hex)", score=1.0,
                                source="parsed_hex")]
            confident = True
        trace.end(outcome=f"resolved {parsed_hex}",
                   candidates=len(cands), confident=confident,
                   top_hex=cands[0].hex if cands else None)
        return Result(
            query=query, normalized=normalized, candidates=cands,
            confident=confident, tier="hex", spread=None,
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
    trace.step("Tier 1 • CSS named lookup", tier="1")
    t1 = tier1(normalized, k=k)
    if t1:
        trace.end(outcome="hit",
                   candidates=len(t1), confident=True, top_hex=t1[0].hex)
        return Result(query, normalized, t1, True, "1",
                       latency_ms=int((time.time() - started) * 1000))
    trace.end(outcome="miss")

    trace.step("Tier 2.5 • local 32k name dictionary", tier="local")
    tlocal = tier_local(normalized, k=k)
    if tlocal:
        cands, tier, confident = tlocal
        trace.end(outcome=f"matched as {tier}",
                   candidates=len(cands), confident=confident,
                   top_hex=cands[0].hex)
        if confident:
            return Result(query, normalized, cands, True, tier,
                           latency_ms=int((time.time() - started) * 1000))
        # Hold the local fuzzy match as a fallback; still try color.pizza
        # in case its `default`-list curation produces a better top-1.
        local_fallback = (cands, tier, confident)
    else:
        trace.end(outcome="miss")
        local_fallback = None

    trace.step("Tier 2/3 • color.pizza name search", tier="2_3")
    t23: tuple[list[Candidate], str, bool] | None = None
    color_pizza_failed = False
    try:
        t23 = tier2_or_3(normalized, k=k)
    except ColorPizzaTransientError as e:
        color_pizza_failed = True
        trace.end(outcome=f"transient error ({e}) — escalating to LLM",
                   confident=False)
    except ColorPizzaPermanentError as e:
        color_pizza_failed = True
        trace.end(outcome=f"permanent error ({e}) — escalating to LLM",
                   confident=False)
    else:
        if t23:
            cands, tier, confident = t23
            trace.end(outcome=f"matched as {tier}",
                       candidates=len(cands), confident=confident,
                       top_hex=cands[0].hex)
        else:
            trace.end(outcome="no plausible match")

    if t23:
        cands, tier, confident = t23
        if not confident:
            try:
                cands4, t4, conf4, spread = _tier4(query, model=model, k=k,
                                                    on_progress=progress,
                                                    trace=trace)
                return Result(query, normalized, cands4, conf4, t4,
                               spread=spread,
                               latency_ms=int((time.time() - started) * 1000))
            except Exception:
                pass
        return Result(query, normalized, cands, confident, tier,
                       latency_ms=int((time.time() - started) * 1000))

    if local_fallback is not None:
        cands, tier, confident = local_fallback
        return Result(query, normalized, cands, confident, tier,
                       latency_ms=int((time.time() - started) * 1000))

    cands4, t4, conf4, spread = _tier4(query, model=model, k=k,
                                        on_progress=progress, trace=trace)
    if color_pizza_failed:
        conf4 = False
    return Result(query, normalized, cands4, conf4, t4, spread=spread,
                   latency_ms=int((time.time() - started) * 1000))
