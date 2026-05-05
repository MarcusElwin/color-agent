from color_agent.consistency import consistent


def _payload(top_hex, name="x"):
    return {
        "candidates": [
            {"hex": top_hex, "name": name, "confidence": "high",
             "rationale": "p"},
            {"hex": "#222222", "name": "alt1", "confidence": "medium",
             "rationale": "p"},
            {"hex": "#333333", "name": "alt2", "confidence": "medium",
             "rationale": "p"},
            {"hex": "#444444", "name": "alt3", "confidence": "low",
             "rationale": "p"},
            {"hex": "#555555", "name": "alt4", "confidence": "low",
             "rationale": "p"},
        ],
        "overall_confidence": "high",
        "source": "knowledge",
    }


def make_sampler(top_hexes):
    samples = iter(top_hexes)

    def sampler(query, model, temperature):
        return _payload(next(samples))
    return sampler


def test_tight_cluster_high_score():
    sampler = make_sampler(["#0047AB", "#0048AC", "#0046AA", "#0047AB", "#0049AD"])
    cands, spread = consistent("cobalt blue", n=5, sampler=sampler, k=5)
    assert spread < 10
    assert cands[0].source == "llm_consistent"
    assert cands[0].score > 0.95
    # Medoid should be one of the cluster
    assert cands[0].hex in {"#0047AB", "#0048AC", "#0046AA", "#0049AD"}


def test_scattered_low_score():
    sampler = make_sampler(["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF"])
    cands, spread = consistent("ambiguous", n=5, sampler=sampler, k=5)
    assert spread > 100
    assert cands[0].score < 0.8


def test_returns_k_candidates():
    sampler = make_sampler(["#0047AB", "#0048AC", "#0046AA", "#0047AB", "#0049AD"])
    cands, _ = consistent("cobalt blue", n=5, sampler=sampler, k=5)
    assert len(cands) == 5
    # No duplicate hexes
    assert len({c.hex for c in cands}) == 5


def test_medoid_wins_against_outlier():
    # 4 close + 1 far. Medoid should be from the cluster, not the outlier.
    sampler = make_sampler(["#0047AB", "#0048AC", "#0046AA", "#0047AB", "#FF0000"])
    cands, _ = consistent("cobalt blue", n=5, sampler=sampler, k=5)
    assert cands[0].hex != "#FF0000"


def test_consistent_preserves_rationale():
    """Rationale from each sample's top-pick should survive into the merged
    candidate list."""
    sampler = make_sampler(["#0047AB", "#0050B0", "#0046AA", "#0048AC", "#0049AD"])
    cands, _ = consistent("cobalt blue", n=5, sampler=sampler, k=5)
    # The fixture sets rationale="p" on every candidate
    assert all(c.rationale == "p" for c in cands)
