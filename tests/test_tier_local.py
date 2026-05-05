"""Tier 2.5 — local 32k color-name dictionary."""

from color_agent.tier_local import hex_neighbors_local, tier_local


def test_exact_match_returns_confident():
    out = tier_local("cobalt")
    assert out is not None
    cands, tier, confident = out
    assert tier == "local-exact"
    assert confident is True
    assert cands[0].name == "cobalt"
    assert cands[0].hex == "#030AA7"
    assert len(cands) == 5


def test_fuzzy_cobalt_blue_finds_cobalt_family():
    """Exact 'Cobalt Blue' isn't in the curated dataset but fuzzy should
    still land on a cobalt-family hex (the regression that motivated this PR)."""
    out = tier_local("cobalt blue")
    assert out is not None
    cands, tier, confident = out
    assert tier == "local-fuzzy"
    assert "cobalt" in cands[0].name.split()
    r, g, b = (int(cands[0].hex[i:i+2], 16) for i in (1, 3, 5))
    assert b > r and b > g  # blue dominant


def test_token_overlap_bonus_beats_pure_char_similarity():
    """`cobalt blue` was top-matching `baltic blue` before the token bonus.
    Now token-containing names win."""
    out = tier_local("cobalt blue")
    assert out is not None
    cands, _, _ = out
    assert "cobalt" in cands[0].name.split()


def test_unknown_query_returns_none():
    """Genuinely-not-a-color string returns None instead of garbage."""
    assert tier_local("xyzzy") is None


def test_empty_query():
    assert tier_local("") is None


def test_returns_k_candidates():
    out = tier_local("cobalt", k=8)
    assert out is not None
    cands, _, _ = out
    assert len(cands) == 8


def test_tight_cluster_is_confident():
    """When top-3 fuzzy hits cluster in RGB space, mark confident even
    though no individual similarity may pass the strict threshold."""
    out = tier_local("cobalt blue")
    assert out is not None
    _, _, confident = out
    assert confident is True


# --- hex_neighbors_local ----------------------------------------------------


def test_hex_neighbors_local_returns_k():
    out = hex_neighbors_local("#0047AB", k=5)
    assert len(out) == 5
    assert all(c.source == "local-hex" for c in out)
    assert all(c.hex.startswith("#") and len(c.hex) == 7 for c in out)


def test_hex_neighbors_local_ranks_by_proximity():
    """Closest hex first; descending RGB distance."""
    out = hex_neighbors_local("#0047AB", k=5)
    scores = [c.score for c in out]
    assert scores == sorted(scores, reverse=True)
    # Top match should be in the cobalt-blue family (blue dominant)
    r, g, b = (int(out[0].hex[i:i+2], 16) for i in (1, 3, 5))
    assert b > r and b > g


def test_hex_neighbors_local_handles_lowercase():
    """Input in any case should resolve."""
    out = hex_neighbors_local("0047ab", k=3)
    assert len(out) == 3
