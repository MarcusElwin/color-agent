"""Live tests — gated by `live` marker AND ANTHROPIC_API_KEY env var.
Run with:  pytest -m live"""

import os

import pytest

from color_agent.distance import rgb_distance
from color_agent.router import to_hex

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="no API key"),
]


# Tier 1 should resolve all of these without any network/LLM call.
@pytest.mark.parametrize("query,expected", [
    ("crimson", "#DC143C"),
    ("rebeccapurple", "#663399"),
    ("cornflowerblue", "#6495ED"),
    ("grey", "#808080"),
])
def test_tier1_lookups(query, expected):
    r = to_hex(query)
    assert r.tier == "1"
    assert r.candidates[0].hex == expected
    assert len(r.candidates) >= 5


# Tier 2/3 hits color.pizza but no LLM.
@pytest.mark.parametrize("query,target,tol", [
    ("cobalt blue", "#0047AB", 40),
    ("burnt sienna", "#E97451", 60),
])
def test_tier_2_3_lookups(query, target, tol):
    r = to_hex(query)
    assert r.tier in {"2", "3", "4-base", "4-reflect", "4-consistent"}
    assert rgb_distance(r.candidates[0].hex, target) <= tol


# Brand / descriptive should reach Tier 4.
def test_brand_query_reaches_tier4():
    r = to_hex("Pantone 1837 Tiffany blue")
    assert r.tier.startswith("4-")
    # Tiffany blue is roughly #0ABAB5 — give a wide tolerance for brand fidelity.
    assert rgb_distance(r.candidates[0].hex, "#0ABAB5") <= 60


def test_descriptive_query_returns_five_candidates():
    r = to_hex("the color of a flamingo at sunset")
    assert len(r.candidates) >= 5
    assert all(c.score is not None for c in r.candidates)
