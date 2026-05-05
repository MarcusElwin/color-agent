"""Tier 2.5 — local 32k color-name dictionary."""

from color_agent.tier_local import tier_local


def test_exact_match_returns_confident():
    out = tier_local("cobalt")
    assert out is not None
    cands, tier, confident = out
    assert tier == "local-exact"
    assert confident is True
    assert cands[0].name == "cobalt"
    assert cands[0].hex == "#030AA7"
    assert len(cands) == 5


def test_fuzzy_cobalt_blue_finds_cobalt():
    """The exact name 'Cobalt Blue' isn't in the dataset (curated 'default'
    excludes it), but fuzzy matching should still land on a near hex."""
    out = tier_local("cobalt blue")
    assert out is not None
    cands, tier, _ = out
    assert tier == "local-fuzzy"
    # Top candidate should be cobalt-family blue, not red/green
    r, g, b = (int(cands[0].hex[i:i+2], 16) for i in (1, 3, 5))
    assert b > r and b > g  # blue dominant


def test_unknown_query_returns_none():
    """Genuinely-not-a-color string should return None, not garbage."""
    assert tier_local("xyzzy") is None or _is_low_confidence(
        tier_local("xyzzy")
    )


def _is_low_confidence(out) -> bool:
    if out is None:
        return True
    return out[0][0].score < 0.7


def test_tier_local_handles_empty_query():
    assert tier_local("") is None


def test_tier_local_returns_k_candidates():
    out = tier_local("cobalt", k=8)
    assert out is not None
    cands, _, _ = out
    assert len(cands) == 8


def test_token_overlap_bonus_beats_pure_char_similarity():
    """Regression: 'cobalt blue' was top-matching 'baltic blue' (high char
    similarity, no token overlap) instead of cobalt-family blues. Token bonus
    should put cobalt-containing names on top."""
    out = tier_local("cobalt blue")
    assert out is not None
    cands, _, _ = out
    # Top result must contain "cobalt" as a token
    assert "cobalt" in cands[0].name.split()


def test_tight_cluster_marks_local_fuzzy_confident():
    """When top-3 fuzzy results are RGB-close, mark confident even if no single
    similarity is high — same logic as tier23 but for the local dataset."""
    out = tier_local("cobalt blue")
    assert out is not None
    _, _, confident = out
    # Three cobalt-family blues should cluster tightly enough to be confident
    assert confident is True
