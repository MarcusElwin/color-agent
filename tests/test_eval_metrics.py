"""compute_metrics is a pure function — easy to drive with synthetic results."""

from color_agent.eval import compute_metrics


def _case(q: str, expected: str, tol: int, tier: str, dist: float,
          all_dists: list[float] | None = None,
          confident: bool = True, split: str = "lookup_resolvable",
          category: str = "css_named", error: bool = False,
          got: str | None = "#000000") -> dict:
    return {
        "q": q, "expected": expected, "tol": tol, "category": category,
        "split": split,
        "got": got, "tier": tier, "confident": confident,
        "dist": dist, "passed": (dist <= tol) and not error,
        "latency_ms": 100,
        "all_dists": all_dists if all_dists is not None else [dist] * 5,
        "all_hexes": ["#000000"] * 5,
        **({"error": "boom"} if error else {}),
    }


def test_top_k_hit_rates():
    """Top-1 misses but top-3 hits if a within-tol candidate is at index 1 or 2."""
    results = [
        _case("a", "#FF0000", tol=10, tier="1", dist=50,
               all_dists=[50, 5, 200, 200, 200]),
        _case("b", "#FF0000", tol=10, tier="1", dist=0),
    ]
    m = compute_metrics(results)
    assert m["top1"] == 1   # case b
    assert m["top3"] == 2   # both, b at index 0 and a at index 1
    assert m["top5"] == 2


def test_routing_accuracy():
    """3 lookup_resolvable: 2 stayed in tier 1/2, 1 escaped to tier 4."""
    results = [
        _case("a", "#000", 10, tier="1", dist=0, split="lookup_resolvable"),
        _case("b", "#000", 10, tier="2", dist=0, split="lookup_resolvable"),
        _case("c", "#000", 10, tier="4-base", dist=0, split="lookup_resolvable"),
        _case("d", "#000", 10, tier="4-base", dist=0, split="llm_required",
               category="brand"),
    ]
    m = compute_metrics(results)
    assert m["routing_accuracy_pct"] == 66.7
    assert m["routing_unnecessary_t4"] == 1
    assert m["lookup_resolvable_total"] == 3


def test_tier4_efficiency():
    """Of 3 tier-4 calls, only 1 was actually llm_required."""
    results = [
        _case("a", "#000", 10, tier="4-base", dist=0, split="lookup_resolvable"),
        _case("b", "#000", 10, tier="4-base", dist=0, split="lookup_resolvable"),
        _case("c", "#000", 10, tier="4-base", dist=0, split="llm_required",
               category="brand"),
    ]
    m = compute_metrics(results)
    assert m["tier4_calls"] == 3
    assert m["tier4_necessary"] == 1
    assert m["tier4_efficiency_pct"] == 33.3


def test_per_tier_accuracy():
    """Tier 1 100% / Tier 4 50%."""
    results = [
        _case("a", "#000", 10, tier="1", dist=0),
        _case("b", "#000", 10, tier="1", dist=5),
        _case("c", "#000", 10, tier="4-base", dist=0, split="llm_required"),
        _case("d", "#000", 10, tier="4-base", dist=50, split="llm_required"),
    ]
    m = compute_metrics(results)
    assert m["per_tier_accuracy"]["1"] == {"passed": 2, "total": 2}
    assert m["per_tier_accuracy"]["4-base"] == {"passed": 1, "total": 2}


def test_confident_accuracy():
    """Confident accuracy is computed only on confident=True cases."""
    results = [
        _case("a", "#000", 10, tier="1", dist=0, confident=True),
        _case("b", "#000", 10, tier="1", dist=50, confident=True),  # wrong
        _case("c", "#000", 10, tier="3", dist=0, confident=False),
    ]
    m = compute_metrics(results)
    assert m["confident_total"] == 2
    assert m["confident_accuracy_pct"] == 50.0


def test_failure_breakdown():
    results = [
        _case("a", "#000", 10, tier="1", dist=0),                    # passed
        _case("b", "#000", 10, tier="1", dist=99, got="#FF0000"),    # wrong hex
        _case("c", "#000", 10, tier="miss", dist=99, got=None),      # no result
        _case("d", "#000", 10, tier="1", dist=0, error=True),        # errored
    ]
    m = compute_metrics(results)
    assert m["failures_wrong_hex"] == 1
    assert m["failures_no_result"] == 1
    assert m["errored"] == 1


def test_cost_estimate_zero_when_no_tier4():
    results = [_case("a", "#000", 10, tier="1", dist=0) for _ in range(5)]
    m = compute_metrics(results)
    assert m["estimated_cost_usd"] == 0.0


def test_cost_estimate_grows_with_tier4_calls():
    base = [_case(f"q{i}", "#000", 10, tier="4-base", dist=0,
                   split="llm_required", category="brand")
             for i in range(3)]
    consistent = [_case(f"q{i}", "#000", 10, tier="4-consistent", dist=0,
                         split="llm_required", category="brand")
                   for i in range(3)]
    base_cost = compute_metrics(base)["estimated_cost_usd"]
    cons_cost = compute_metrics(consistent)["estimated_cost_usd"]
    # consistent (5 samples) should be materially more expensive than 4-base
    assert cons_cost > base_cost * 2


def test_distance_summary():
    results = [
        _case("a", "#000", 100, tier="1", dist=0),
        _case("b", "#000", 100, tier="1", dist=10),
        _case("c", "#000", 100, tier="1", dist=50),
    ]
    m = compute_metrics(results)
    assert m["mean_dist"] == 20.0
    assert m["median_dist"] == 10.0
    assert m["max_dist"] == 50.0


def test_empty_results():
    assert compute_metrics([]) == {"total": 0}
