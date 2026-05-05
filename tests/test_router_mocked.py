from unittest.mock import patch

from color_agent.types import Candidate
from color_agent.router import to_hex


def _stub_candidates(hex_, source, n=5):
    return [Candidate(hex=hex_, name="x", score=1.0 - i * 0.05, source=source)
            for i in range(n)]


def test_css_query_uses_tier1_only():
    """A plain CSS-named color must NOT touch color.pizza or the LLM."""
    with patch("color_agent.router.tier2_or_3") as t23, \
         patch("color_agent.router.call_agent") as agent, \
         patch("color_agent.router.consistent") as cons:
        result = to_hex("crimson")
        assert result.tier == "1"
        assert result.confident is True
        assert result.candidates[0].hex == "#DC143C"
        assert not t23.called
        assert not agent.called
        assert not cons.called


def test_grey_aliases_to_gray_in_tier1():
    result = to_hex("grey")
    assert result.tier == "1"
    assert result.candidates[0].hex == "#808080"
    assert result.normalized == "gray"


def test_punctuation_normalized_before_tier1():
    # Not in CSS dict; force tier1 so we don't hit the network when verifying
    # that normalize ran first.
    result = to_hex("Cobalt-Blue!", force="tier1")
    assert result.normalized == "cobalt blue"
    assert result.candidates == []


def test_tier1_miss_falls_through_to_color_pizza():
    with patch("color_agent.router.tier2_or_3") as t23, \
         patch("color_agent.router.call_agent") as agent:
        t23.return_value = (_stub_candidates("#0047AB", "color_pizza_exact"),
                             "2", True)
        result = to_hex("cobalt blue")
        assert result.tier == "2"
        assert t23.called
        assert not agent.called


def test_color_pizza_exact_returns_confident():
    with patch("color_agent.router.tier2_or_3") as t23:
        t23.return_value = (_stub_candidates("#0047AB", "color_pizza_exact"),
                             "2", True)
        result = to_hex("cobalt blue")
        assert result.confident is True
        assert result.tier == "2"


def test_color_pizza_low_confidence_fuzzy_escalates_to_llm():
    """Tier 3 with confident=False should kick to Tier 4."""
    with patch("color_agent.router.tier2_or_3") as t23, \
         patch("color_agent.router._tier4") as tier4:
        t23.return_value = (_stub_candidates("#888888", "color_pizza_fuzzy"),
                             "3", False)
        tier4.return_value = (
            _stub_candidates("#777777", "llm_knowledge"), "4-base", True, None,
        )
        result = to_hex("kind of greyish maybe")
        assert tier4.called
        assert result.tier == "4-base"


def test_complete_miss_routes_to_llm():
    with patch("color_agent.router.tier2_or_3") as t23, \
         patch("color_agent.router._tier4") as tier4:
        t23.return_value = None
        tier4.return_value = (
            _stub_candidates("#FC8EAC", "llm_knowledge"), "4-base", True, None,
        )
        result = to_hex("the color of a flamingo at sunset")
        assert tier4.called
        assert result.tier == "4-base"


def test_brand_query_skips_to_consistency_inside_tier4():
    with patch("color_agent.router.tier2_or_3") as t23, \
         patch("color_agent.router.consistent") as cons:
        t23.return_value = None
        cons.return_value = (_stub_candidates("#0ABAB5", "llm_consistent"), 8.0)
        result = to_hex("Pantone 1837 Tiffany blue")
        assert cons.called
        assert result.tier == "4-consistent"
        assert result.spread == 8.0


def test_tier4_high_confidence_returns_base_only():
    with patch("color_agent.router.tier2_or_3") as t23, \
         patch("color_agent.router.call_agent") as agent, \
         patch("color_agent.router.reflect") as refl, \
         patch("color_agent.router.consistent") as cons:
        t23.return_value = None
        agent.return_value = {
            "candidates": [{"hex": "#0047AB", "name": "x",
                              "confidence": "high", "rationale": "p"}] * 5,
            "overall_confidence": "high", "source": "knowledge",
        }
        result = to_hex("some descriptive shade")
        assert result.tier == "4-base"
        assert not refl.called
        assert not cons.called


def test_tier4_medium_routes_to_reflect():
    with patch("color_agent.router.tier2_or_3") as t23, \
         patch("color_agent.router.call_agent") as agent, \
         patch("color_agent.router.reflect") as refl:
        t23.return_value = None
        agent.return_value = {
            "candidates": [{"hex": "#888888", "name": "x",
                              "confidence": "medium", "rationale": "p"}] * 5,
            "overall_confidence": "medium", "source": "knowledge",
        }
        refl.return_value = {
            "candidates": [{"hex": "#777777", "name": "x",
                              "confidence": "high", "rationale": "p"}] * 5,
            "overall_confidence": "high", "source": "knowledge",
        }
        result = to_hex("some descriptive shade")
        assert refl.called
        assert result.tier == "4-reflect"


def test_tier4_low_routes_to_consistency():
    with patch("color_agent.router.tier2_or_3") as t23, \
         patch("color_agent.router.call_agent") as agent, \
         patch("color_agent.router.consistent") as cons:
        t23.return_value = None
        agent.return_value = {
            "candidates": [{"hex": "#888888", "name": "x",
                              "confidence": "low", "rationale": "p"}] * 5,
            "overall_confidence": "low", "source": "knowledge",
        }
        cons.return_value = (_stub_candidates("#777777", "llm_consistent"), 12.0)
        result = to_hex("some descriptive shade")
        assert cons.called
        assert result.tier == "4-consistent"


def test_force_tier1():
    result = to_hex("crimson", force="tier1")
    assert result.tier == "1"


def test_force_tier1_miss_returns_empty():
    result = to_hex("not-a-color", force="tier1")
    assert result.tier == "1"
    assert result.candidates == []
    assert result.confident is False


def test_bare_hex_input_uses_hex_neighbors():
    with patch("color_agent.router.hex_neighbors") as hn:
        hn.return_value = _stub_candidates("#0047AB", "color_pizza_hex")
        result = to_hex("#0047AB")
        assert result.tier == "hex"
        assert hn.called


def test_progress_callback_emits_phases_for_tier1():
    seen: list[str] = []
    to_hex("crimson", on_progress=seen.append)
    # Should at least announce normalize + Tier 1
    joined = " | ".join(seen).lower()
    assert "normaliz" in joined
    assert "tier 1" in joined


def test_progress_callback_emits_tier4_phase_on_brand():
    from unittest.mock import patch
    seen: list[str] = []
    with patch("color_agent.router.tier2_or_3", return_value=None), \
         patch("color_agent.router.consistent") as cons:
        cons.return_value = (_stub_candidates("#0ABAB5", "llm_consistent"), 8.0)
        to_hex("Pantone 1837", on_progress=seen.append)
    joined = " | ".join(seen).lower()
    assert "tier 4 consistency" in joined


def test_color_pizza_failure_falls_through_to_llm():
    with patch("color_agent.router.tier2_or_3", side_effect=Exception("boom")), \
         patch("color_agent.router._tier4") as tier4:
        tier4.return_value = (
            _stub_candidates("#0047AB", "llm_knowledge"), "4-base", True, None,
        )
        result = to_hex("some unknown shade")
        assert tier4.called
        assert result.tier == "4-base"
