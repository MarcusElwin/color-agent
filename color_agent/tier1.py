"""Tier 1: in-process CSS-named-colors lookup. On exact hit, pad to K candidates
using nearest-RGB neighbors from the same dict so the caller always gets a
ranked list."""

from color_agent.css_colors import CSS_NAMED_COLORS, lookup as css_lookup
from color_agent.distance import knn, similarity_from_distance
from color_agent.types import Candidate


def tier1(normalized: str, k: int = 5) -> list[Candidate] | None:
    """Return >= k candidates if the normalized query matches a CSS name.
    First candidate is the exact match (score 1.0); the rest are RGB-nearest."""
    hex_ = css_lookup(normalized)
    if hex_ is None:
        return None

    name = normalized.replace(" ", "")
    out: list[Candidate] = [Candidate(hex=hex_, name=name, score=1.0, source="css")]

    neighbors = knn(hex_, CSS_NAMED_COLORS, k=k - 1, exclude={name})
    for nb_name, nb_hex, dist in neighbors:
        out.append(Candidate(
            hex=nb_hex,
            name=nb_name,
            score=round(similarity_from_distance(dist), 4),
            source="css",
        ))
    return out
