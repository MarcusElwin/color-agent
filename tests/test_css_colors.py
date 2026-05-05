import re

from color_agent.css_colors import CSS_NAMED_COLORS, lookup
from color_agent.tier1 import tier1


def test_count_is_141_american_spellings():
    assert len(CSS_NAMED_COLORS) == 141


def test_all_values_are_uppercase_hex():
    pattern = re.compile(r"^#[0-9A-F]{6}$")
    for name, hex_ in CSS_NAMED_COLORS.items():
        assert pattern.match(hex_), f"{name} -> {hex_}"


def test_well_known_entries():
    assert CSS_NAMED_COLORS["crimson"] == "#DC143C"
    assert CSS_NAMED_COLORS["rebeccapurple"] == "#663399"
    assert CSS_NAMED_COLORS["cornflowerblue"] == "#6495ED"


def test_aqua_cyan_share_hex():
    assert CSS_NAMED_COLORS["aqua"] == CSS_NAMED_COLORS["cyan"]


def test_lookup_handles_space_collapsed():
    # Tier 1 should match both 'cornflowerblue' and 'cornflower blue' as user input
    assert lookup("cornflowerblue") == "#6495ED"
    assert lookup("cornflower blue") == "#6495ED"


def test_lookup_miss():
    assert lookup("not-a-color") is None


def test_tier1_returns_five_candidates():
    out = tier1("crimson", k=5)
    assert out is not None
    assert len(out) == 5
    assert out[0].name == "crimson"
    assert out[0].hex == "#DC143C"
    assert out[0].score == 1.0
    assert out[0].source == "css"


def test_tier1_neighbors_ranked_by_similarity():
    out = tier1("red", k=5)
    assert out is not None
    scores = [c.score for c in out]
    assert scores == sorted(scores, reverse=True)
    assert all(c.source == "css" for c in out)


def test_tier1_miss_returns_none():
    assert tier1("not-a-color", k=5) is None
