import pytest

from color_agent.distance import (
    MAX_RGB_DISTANCE, hex_to_rgb, knn, medoid, rgb_distance,
    rgb_to_hex, similarity_from_distance,
)


def test_identical_distance_zero():
    assert rgb_distance("#0047AB", "#0047AB") == 0


def test_max_distance_black_to_white():
    assert rgb_distance("#000000", "#FFFFFF") == pytest.approx(MAX_RGB_DISTANCE, rel=1e-6)


def test_roundtrip_hex_rgb():
    assert rgb_to_hex(*hex_to_rgb("#0047AB")) == "#0047AB"


def test_medoid_picks_cluster_center():
    hexes = ["#0047AB", "#0050B0", "#0045A8", "#FF0000"]
    chosen, spread = medoid(hexes)
    assert chosen in {"#0047AB", "#0050B0", "#0045A8"}
    assert spread > 100


def test_medoid_tight_cluster_low_spread():
    chosen, spread = medoid(["#0047AB", "#0048AC", "#0046AA"])
    assert spread < 5


def test_medoid_single_input():
    assert medoid(["#0047AB"]) == ("#0047AB", 0.0)


def test_medoid_empty_raises():
    with pytest.raises(ValueError):
        medoid([])


def test_knn_returns_sorted_ascending():
    candidates = {"red": "#FF0000", "darkred": "#8B0000", "blue": "#0000FF"}
    out = knn("#FF0000", candidates, k=2)
    assert out[0][0] == "red"  # exact match closest
    assert out[1][0] == "darkred"  # then darkred (similar hue)
    assert out[0][2] <= out[1][2]


def test_knn_excludes():
    candidates = {"red": "#FF0000", "darkred": "#8B0000", "blue": "#0000FF"}
    out = knn("#FF0000", candidates, k=2, exclude={"red"})
    assert out[0][0] == "darkred"
    assert all(name != "red" for name, _, _ in out)


def test_similarity_from_distance():
    assert similarity_from_distance(0) == 1.0
    assert similarity_from_distance(MAX_RGB_DISTANCE) == 0.0
    assert 0 < similarity_from_distance(100) < 1
