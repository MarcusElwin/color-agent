"""Tiers 2 and 3: color.pizza name search.
  Tier 2 = top similarity == 1.0 (exact name match)
  Tier 3 = top similarity in [FUZZY_MIN, 1.0) (fuzzy match)
Both return the top-K from the same response — already ranked by the API."""

from color_agent.color_pizza import ColorPizzaClient, get_client
from color_agent.types import Candidate

EXACT_THRESHOLD = 1.0
FUZZY_MIN = 0.85
HIGH_CONFIDENCE_FUZZY = 0.92


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


def tier2_or_3(normalized: str, k: int = 5,
               client: ColorPizzaClient | None = None,
               list_: str = "default") -> tuple[list[Candidate], str, bool] | None:
    """Return (candidates, tier_label, confident) or None on no usable match.

    tier_label is '2' for exact, '3' for fuzzy. confident is True for tier 2
    and for tier 3 when top similarity >= HIGH_CONFIDENCE_FUZZY."""
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
    confident = top_sim >= HIGH_CONFIDENCE_FUZZY
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
