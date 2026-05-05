"""Tiers 2 and 3: color.pizza name search.
  Tier 2 = top similarity == 1.0 (exact name match)
  Tier 3 = top similarity in [FUZZY_MIN, 1.0) (fuzzy match)
Both return the top-K from the same response — already ranked by the API.

Threshold tuning: FUZZY_MIN was originally 0.85 but the eval showed common
standard-named queries (`salmon pink`, `mustard yellow`, `burnt sienna`)
score 0.65–0.80 even though they're real entries. Lowering the floor to 0.65
catches them; the cluster-tightness check below is what gates `confident`."""

from color_agent.color_pizza import ColorPizzaClient, get_client
from color_agent.distance import rgb_distance
from color_agent.types import Candidate

EXACT_THRESHOLD = 1.0
FUZZY_MIN = 0.65               # was 0.85; see docstring
HIGH_CONFIDENCE_FUZZY = 0.85   # was 0.92; aligns with where Levenshtein gets reliable
TIGHT_CLUSTER_RGB_DISTANCE = 25  # max pairwise distance among top-3 to count as "cluster"


def _to_candidates(colors: list[dict], source: str, k: int) -> list[Candidate]:
    out: list[Candidate] = []
    for c in colors[:k]:
        out.append(Candidate(
            hex=c["hex"].upper(),
            name=c["name"],
            score=round(float(c.get("similarity", 0.0)), 4),
            source=source,
        ))
    return out


def _cluster_is_tight(colors: list[dict], n: int = 3) -> bool:
    """If the top-N results are all close in RGB space, the fuzzy match is
    semantically convergent even if any single similarity is mediocre — that's
    a real signal we shouldn't waste an LLM call on."""
    if len(colors) < 2:
        return False
    hexes = [c["hex"].upper() for c in colors[:n]]
    return max(
        rgb_distance(a, b)
        for i, a in enumerate(hexes) for b in hexes[i + 1:]
    ) <= TIGHT_CLUSTER_RGB_DISTANCE


def tier2_or_3(normalized: str, k: int = 5,
               client: ColorPizzaClient | None = None,
               list_: str = "default") -> tuple[list[Candidate], str, bool] | None:
    """Return (candidates, tier_label, confident) or None on no usable match.

    tier_label is '2' for exact, '3' for fuzzy. confident is True for tier 2,
    and for tier 3 when EITHER (a) top similarity >= HIGH_CONFIDENCE_FUZZY
    OR (b) the top-3 results form a tight RGB cluster (semantic convergence
    on a single hex even if no individual name passes the strict threshold)."""
    cli = client or get_client()
    data = cli.name_search(normalized, list_=list_, max_results=max(k * 2, 10))
    colors = data.get("colors", [])
    if not colors:
        return None

    top_sim = float(colors[0].get("similarity", 0.0))
    if top_sim < FUZZY_MIN:
        return None

    if top_sim >= EXACT_THRESHOLD:
        cands = _to_candidates(colors, source="color_pizza_exact", k=k)
        return cands, "2", True

    cands = _to_candidates(colors, source="color_pizza_fuzzy", k=k)
    confident = top_sim >= HIGH_CONFIDENCE_FUZZY or _cluster_is_tight(colors)
    return cands, "3", confident


def hex_neighbors(hex_: str, k: int = 5,
                  client: ColorPizzaClient | None = None,
                  list_: str = "default") -> list[Candidate]:
    """For bare-hex queries: ask color.pizza for the named neighbor of the hex.
    Result has CIEDE2000 distance in `distance` field; convert to a 0..1 score."""
    cli = client or get_client()
    data = cli.hex_lookup(hex_, list_=list_)
    colors = data.get("colors", [])
    out: list[Candidate] = []
    for c in colors[:k]:
        # CIEDE2000: ~0 = identical, ~30+ = very different. Normalize crudely.
        d = float(c.get("distance", 0.0))
        score = max(0.0, 1.0 - d / 50.0)
        out.append(Candidate(
            hex=c["hex"].upper(),
            name=c["name"],
            score=round(score, 4),
            source="color_pizza_hex",
        ))
    return out
