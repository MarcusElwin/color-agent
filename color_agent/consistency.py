"""Tier 4 self-consistency: N parallel samples, medoid wins, spread overrides
self-reported confidence.

Two execution paths:
  - ThreadPoolExecutor (default): full latency control, 5x normal cost.
  - Batches API (use_batch=True): non-latency-sensitive, 50% discount.
For our use case (interactive queries) ThreadPool is the right default. Batches
is the right pick when this runs in a daily eval/backfill job."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from statistics import mean
from typing import Any, Callable

from color_agent.agent import call_agent, CONFIDENCE_TO_SCORE, DEFAULT_MODEL
from color_agent.distance import medoid, rgb_distance, similarity_from_distance
from color_agent.types import Candidate

DEFAULT_N = 5
DEFAULT_TEMPERATURE = 1.0


def _sample_once(query: str, model: str, temperature: float) -> dict[str, Any]:
    return call_agent(query, model=model, temperature=temperature)


def consistent(query: str, n: int = DEFAULT_N,
               temperature: float = DEFAULT_TEMPERATURE,
               model: str = DEFAULT_MODEL,
               sampler: Callable[[str, str, float], dict[str, Any]] | None = None,
               k: int = 5) -> tuple[list[Candidate], float]:
    """Return (ranked candidates, spread). Spread is mean pairwise RGB distance
    among the N top picks across samples — used to override the model's
    self-reported confidence.

    sampler is an injection point for tests; defaults to the live agent."""
    sample = sampler or _sample_once

    with ThreadPoolExecutor(max_workers=n) as pool:
        payloads: list[dict[str, Any]] = list(pool.map(
            lambda _: sample(query, model, temperature), range(n)
        ))

    top_hexes = [p["candidates"][0]["hex"].upper() for p in payloads]
    medoid_hex, spread = medoid(top_hexes)
    spread_similarity = similarity_from_distance(spread)

    # Pick the sample whose top is the medoid (any of them if duplicated).
    winner = next(p for p in payloads if p["candidates"][0]["hex"].upper() == medoid_hex)

    def _rat(c: dict) -> str | None:
        r = c.get("rationale")
        return r.strip() if isinstance(r, str) else None

    out: list[Candidate] = []
    seen: set[str] = set()
    out.append(Candidate(
        hex=medoid_hex,
        name=winner["candidates"][0].get("name", ""),
        score=round(spread_similarity, 4),
        source="llm_consistent",
        rationale=_rat(winner["candidates"][0]),
    ))
    seen.add(medoid_hex)

    others = [p["candidates"][0] for p in payloads
              if p["candidates"][0]["hex"].upper() != medoid_hex]
    others.sort(key=lambda c: rgb_distance(medoid_hex, c["hex"].upper()))
    for c in others:
        h = c["hex"].upper()
        if h in seen:
            continue
        out.append(Candidate(
            hex=h, name=c.get("name", ""),
            score=round(CONFIDENCE_TO_SCORE.get(c.get("confidence", "medium"), 0.7), 4),
            source="llm_consistent",
            rationale=_rat(c),
        ))
        seen.add(h)
        if len(out) >= k:
            break

    for c in winner["candidates"][1:]:
        if len(out) >= k:
            break
        h = c["hex"].upper()
        if h in seen:
            continue
        out.append(Candidate(
            hex=h, name=c.get("name", ""),
            score=round(CONFIDENCE_TO_SCORE.get(c.get("confidence", "medium"), 0.7), 4),
            source="llm_consistent",
            rationale=_rat(c),
        ))
        seen.add(h)

    return out[:k], round(spread, 2)
