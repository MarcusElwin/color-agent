"""Tier 2.5: in-process color-name dictionary (~32k entries from
meodai/color-name-list, MIT-licensed). Sits between Tier 1 (CSS) and Tier 2/3
(color.pizza network) so the common case never hits the network.

color.pizza serves HTTP 403 to many networks ("commercial-feel traffic"
detection) — without this tier, "cobalt blue" would escape to the LLM and
cost ~30s per query. With it, ~290 ms.

Two entry points, mirroring tier23.py's contract:
  tier_local(name)         → name → hex (Levenshtein-flavored fuzzy match)
  hex_neighbors_local(hex) → hex → nearest names (RGB KNN)
"""

from __future__ import annotations

import csv
import difflib
from functools import lru_cache
from importlib import resources

from color_agent.distance import knn, rgb_distance, similarity_from_distance
from color_agent.types import Candidate

EXACT_THRESHOLD = 1.0
FUZZY_MIN = 0.65
HIGH_CONFIDENCE_FUZZY = 0.85
TIGHT_CLUSTER_RGB_DISTANCE = 25
DEFAULT_K = 5
MAX_FUZZY_CANDIDATES = 2000


@lru_cache(maxsize=1)
def _load() -> tuple[dict[str, str], dict[str, list[tuple[str, str]]]]:
    """Returns (name_to_hex_lower, first_char_index)."""
    name_to_hex: dict[str, str] = {}
    first_char: dict[str, list[tuple[str, str]]] = {}
    pkg = resources.files("color_agent")
    with pkg.joinpath("_colornames.csv").open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # "name,hex" header
        for row in reader:
            if len(row) < 2:
                continue
            name, hex_ = row[0], row[1]
            key = name.strip().lower()
            up = hex_.strip().upper()
            if not key or not up.startswith("#") or len(up) != 7:
                continue
            name_to_hex[key] = up
            first_char.setdefault(key[0], []).append((key, up))
    return name_to_hex, first_char


def _similarity(a: str, b: str) -> float:
    """Levenshtein-flavored ratio + token-overlap bonus. Without the bonus,
    `cobalt blue` over-matches `baltic blue` (high char overlap, no shared
    tokens). 70% char + 30% token: an exact-token match (`cobalt blue` →
    `cyan cobalt blue`) wins over a same-shape mismatch."""
    base = difflib.SequenceMatcher(None, a, b).ratio()
    a_toks = set(a.split())
    b_toks = set(b.split())
    if not a_toks:
        return base
    overlap = len(a_toks & b_toks) / len(a_toks)
    return 0.7 * base + 0.3 * overlap


def _cluster_is_tight(hexes: list[str], n: int = 3) -> bool:
    """Top-n results converge in RGB space → confident even if no individual
    similarity is high. Same logic as tier23._cluster_is_tight."""
    if len(hexes) < 2:
        return False
    sub = hexes[:n]
    return max(
        rgb_distance(a, b) for i, a in enumerate(sub) for b in sub[i + 1:]
    ) <= TIGHT_CLUSTER_RGB_DISTANCE


def _to_candidates(matches: list[tuple[str, str, float]],
                    source: str, k: int) -> list[Candidate]:
    return [
        Candidate(hex=h, name=n, score=round(s, 4), source=source)
        for n, h, s in matches[:k]
    ]


def tier_local(normalized: str, k: int = DEFAULT_K
               ) -> tuple[list[Candidate], str, bool] | None:
    """Local fuzzy lookup. None means genuinely no plausible match. Same
    return shape as tier23.tier2_or_3."""
    name_to_hex, first_char = _load()
    if not normalized:
        return None

    exact = name_to_hex.get(normalized)
    if exact is not None:
        all_pairs = [(n, h, rgb_distance(exact, h))
                      for n, h in name_to_hex.items() if n != normalized]
        all_pairs.sort(key=lambda t: t[2])
        cands = [Candidate(hex=exact, name=normalized,
                            score=1.0, source="local-exact")]
        for n, h, d in all_pairs[:k - 1]:
            cands.append(Candidate(
                hex=h, name=n,
                score=round(similarity_from_distance(d), 4),
                source="local-exact",
            ))
        return cands, "local-exact", True

    pool = first_char.get(normalized[0], [])
    if not pool or len(pool) > MAX_FUZZY_CANDIDATES:
        pool = list(name_to_hex.items())

    scored = [(n, h, _similarity(normalized, n)) for n, h in pool]
    scored.sort(key=lambda t: t[2], reverse=True)

    if not scored or scored[0][2] < FUZZY_MIN:
        return None

    cands = _to_candidates(scored, source="local-fuzzy", k=k)
    top_sim = scored[0][2]
    confident = (
        top_sim >= HIGH_CONFIDENCE_FUZZY
        or _cluster_is_tight([c.hex for c in cands])
    )
    return cands, "local-fuzzy", confident


def hex_neighbors_local(hex_: str, k: int = DEFAULT_K) -> list[Candidate]:
    """Hex → nearest-named neighbors via RGB KNN over the local 32k dataset.
    Drop-in replacement for tier23.hex_neighbors when color.pizza is unreachable
    (which, on networks where color.pizza 403s, is "always")."""
    name_to_hex, _ = _load()
    if not name_to_hex:
        return []
    target = hex_.upper()
    nearest = knn(target, name_to_hex, k=k)
    return [
        Candidate(
            hex=h, name=n,
            score=round(similarity_from_distance(d), 4),
            source="local-hex",
        )
        for n, h, d in nearest
    ]
