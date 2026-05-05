"""Tier 2.5: in-process color-name dictionary (~32k entries from
meodai/color-name-list, MIT-licensed). Sits between Tier 1 (CSS) and Tier 2/3
(color.pizza network) so the common case never hits the network.

Why this tier exists: color.pizza serves 403 from many IPs / commercial-feel
networks, which used to make every "cobalt blue" query escape to the LLM
(~30s wait). The local dataset has the same coverage and resolves in <10ms.

Contract mirrors tier23.tier2_or_3:
  return (candidates, tier_label, confident) | None
where tier_label is "local-exact" or "local-fuzzy" so the eval can tell
them apart from network tiers.
"""

from __future__ import annotations

import csv
import difflib
from functools import lru_cache
from importlib import resources

from color_agent.distance import rgb_distance
from color_agent.types import Candidate

EXACT_THRESHOLD = 1.0
FUZZY_MIN = 0.65
HIGH_CONFIDENCE_FUZZY = 0.85
TIGHT_CLUSTER_RGB_DISTANCE = 25
DEFAULT_K = 5

# Cap the upper end of fuzzy candidate evaluation; full O(N) Levenshtein over
# 32k entries would still be fast (~50ms) but we can do better. Quick prefilter:
# any entry sharing the first character with the query is a candidate; everything
# else is statistically unlikely to win. Reduces 32k→~1.5k for typical queries.
MAX_FUZZY_CANDIDATES = 2000


@lru_cache(maxsize=1)
def _load() -> tuple[dict[str, str], dict[str, list[tuple[str, str]]]]:
    """Returns (name_to_hex_lower, first_char_index).
    name_to_hex_lower: lowercased-name -> '#RRGGBB' (uppercase)
    first_char_index: first letter -> [(lowercased-name, hex), ...]"""
    name_to_hex: dict[str, str] = {}
    first_char: dict[str, list[tuple[str, str]]] = {}
    pkg = resources.files("color_agent")
    with pkg.joinpath("_colornames.csv").open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip "name,hex" header
        for row in reader:
            if len(row) < 2:
                continue
            name, hex_ = row[0], row[1]
            key = name.strip().lower()
            up = hex_.strip().upper()
            if not key or not up.startswith("#") or len(up) != 7:
                continue
            name_to_hex[key] = up
            if key:
                first_char.setdefault(key[0], []).append((key, up))
    return name_to_hex, first_char


def _similarity(a: str, b: str) -> float:
    """Levenshtein-flavored ratio with a token-overlap bonus. SequenceMatcher
    alone over-rewards character overlap (`cobalt blue` matches `baltic blue`
    too well); the bonus boosts results where every query token appears in
    the candidate name. Output stays in 0..1 so thresholds carry over."""
    base = difflib.SequenceMatcher(None, a, b).ratio()
    a_toks = set(a.split())
    b_toks = set(b.split())
    if not a_toks:
        return base
    overlap = len(a_toks & b_toks) / len(a_toks)
    # 70% character similarity + 30% token overlap. Tuned so an exact-token
    # match (`cobalt blue` -> `cyan cobalt blue`) wins over a same-shape
    # mismatch (`cobalt blue` -> `baltic blue`).
    return 0.7 * base + 0.3 * overlap


def _cluster_is_tight(hexes: list[str], n: int = 3) -> bool:
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
    """Local fuzzy lookup. None means genuinely no plausible match — caller
    should escalate. Returns same shape as tier23.tier2_or_3."""
    name_to_hex, first_char = _load()
    if not normalized:
        return None

    # Fast path: exact match on the lowercased name.
    exact = name_to_hex.get(normalized)
    if exact is not None:
        # Pad with K-1 closest-RGB neighbors from the local dataset.
        all_pairs = [(n, h, rgb_distance(exact, h))
                      for n, h in name_to_hex.items() if n != normalized]
        all_pairs.sort(key=lambda t: t[2])
        cands = [Candidate(hex=exact, name=normalized,
                            score=1.0, source="local-exact")]
        for n, h, d in all_pairs[:k - 1]:
            cands.append(Candidate(
                hex=h, name=n,
                score=round(max(0.0, 1.0 - d / 441.67), 4),
                source="local-exact",
            ))
        return cands, "local-exact", True

    # Fuzzy path with first-char prefilter.
    pool = first_char.get(normalized[0], [])
    if not pool or len(pool) > MAX_FUZZY_CANDIDATES:
        # Either no candidates with that first letter, or too many — fall back
        # to scanning the full dataset. Still fast (~50ms over 32k).
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
