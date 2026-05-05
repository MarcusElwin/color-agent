"""RGB distance, hex parsing, medoid, K-nearest-neighbors.

Euclidean RGB is intentionally cheap — for "is this answer in the right hue
family" granularity it's enough. color.pizza already exposes CIEDE2000 for
the cases where we need perceptual distance."""

from statistics import mean


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02X}{g:02X}{b:02X}"


def rgb_distance(a: str, b: str) -> float:
    ar, ag, ab = hex_to_rgb(a)
    br, bg, bb = hex_to_rgb(b)
    return ((ar - br) ** 2 + (ag - bg) ** 2 + (ab - bb) ** 2) ** 0.5


MAX_RGB_DISTANCE = 441.6729559300637  # sqrt(3) * 255


def similarity_from_distance(d: float) -> float:
    return max(0.0, 1.0 - d / MAX_RGB_DISTANCE)


def medoid(hexes: list[str]) -> tuple[str, float]:
    """Return (medoid_hex, mean_pairwise_spread). Medoid is robust to outliers
    and always an actual sample (vs centroid which can be a never-output color)."""
    if not hexes:
        raise ValueError("medoid requires at least one hex")
    if len(hexes) == 1:
        return hexes[0], 0.0
    best_hex, best_score = hexes[0], float("inf")
    for h in hexes:
        score = sum(rgb_distance(h, other) for other in hexes if other != h)
        if score < best_score:
            best_hex, best_score = h, score
    spread = mean(
        rgb_distance(a, b) for i, a in enumerate(hexes) for b in hexes[i + 1:]
    )
    return best_hex, spread


def knn(target_hex: str, candidates: dict[str, str], k: int,
        exclude: set[str] | None = None) -> list[tuple[str, str, float]]:
    """Return [(name, hex, distance), ...] sorted by distance ascending.
    candidates: name -> hex dict. exclude: names to skip."""
    exclude = exclude or set()
    scored = [
        (name, hex_, rgb_distance(target_hex, hex_))
        for name, hex_ in candidates.items()
        if name not in exclude
    ]
    scored.sort(key=lambda t: t[2])
    return scored[:k]
