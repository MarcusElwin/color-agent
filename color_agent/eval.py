"""Eval harness packaged as a module so the installed CLI can import it.

Reports:
  - Accuracy by category (% within tolerance)
  - Routing accuracy (% of `lookup_resolvable` that did NOT hit Tier 4)
  - Latency p50 / p95
"""

from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

from rich.box import ROUNDED
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from color_agent.agent import DEFAULT_MODEL
from color_agent.distance import rgb_distance
from color_agent.router import to_hex

DATASET_PATH = Path(__file__).resolve().parent.parent / "evals" / "dataset.json"


# Tier color mapping reused from cli.py palette so the eval rendering matches
# the per-query CLI output users already know.
TIER_COLOR = {
    "1":            "green",
    "2":            "cyan",
    "3":            "yellow",
    "4-base":       "magenta",
    "4-reflect":    "magenta",
    "4-consistent": "magenta",
    "hex":          "blue",
    "miss":         "red",
}


def _evaluate_case(case: dict, model: str = DEFAULT_MODEL) -> dict:
    """Run one case end-to-end. Pure function — no I/O beyond to_hex itself.
    Captures the full candidate list so we can compute top-K hit rates."""
    t0 = time.time()
    try:
        r = to_hex(case["q"], model=model)
        top = r.candidates[0] if r.candidates else None
        d = rgb_distance(top.hex, case["expected"]) if top else float("inf")
        all_dists = [rgb_distance(c.hex, case["expected"]) for c in r.candidates]
        return {
            **case,
            "got": top.hex if top else None,
            "tier": r.tier,
            "confident": r.confident,
            "dist": round(d, 1),
            "passed": d <= case["tol"],
            "latency_ms": int((time.time() - t0) * 1000),
            "spread": r.spread,
            "k": len(r.candidates),
            "all_dists": [round(x, 1) for x in all_dists],
            "all_hexes": [c.hex for c in r.candidates],
            "model": model,
        }
    except Exception as e:
        return {
            **case, "error": repr(e), "passed": False,
            "latency_ms": int((time.time() - t0) * 1000),
            "model": model,
        }


# Rough $/1M-token costs — accurate enough for relative comparison across runs;
# tweak when Anthropic adjusts pricing.
MODEL_COST_PER_M_INPUT = {
    "claude-sonnet-4-6": 3.0, "claude-sonnet-4-5": 3.0,
    "claude-opus-4-7":  15.0,
    "claude-haiku-4-5":  1.0,
}
MODEL_COST_PER_M_OUTPUT = {
    "claude-sonnet-4-6": 15.0, "claude-sonnet-4-5": 15.0,
    "claude-opus-4-7":   75.0,
    "claude-haiku-4-5":   5.0,
}
# Tier 4 call shapes: rough mean tokens-in/out per LLM call. Used for cost estimation
# only — overestimates are fine since this is a "did routing save money?" gauge.
TIER4_TOKENS = {
    "4-base":       (1500, 800),    # 2 calls (web_search + force) ~ avg
    "4-reflect":    (3500, 1500),   # base + reflect
    "4-consistent": (8000, 4000),   # 5 parallel base samples
}


def _estimate_cost_usd(results: list[dict],
                       model: str | None = None) -> float:
    """Approximate $ spent on Tier 4 LLM calls. Per-row `model` field wins;
    falls back to the explicit `model` arg, then to Sonnet 4.6 pricing."""
    total = 0.0
    for r in results:
        tier = r.get("tier", "")
        toks = TIER4_TOKENS.get(tier)
        if not toks:
            continue
        m = r.get("model") or model or "claude-sonnet-4-6"
        in_rate = MODEL_COST_PER_M_INPUT.get(m, 3.0) / 1_000_000
        out_rate = MODEL_COST_PER_M_OUTPUT.get(m, 15.0) / 1_000_000
        total += toks[0] * in_rate + toks[1] * out_rate
    return round(total, 4)


def compute_metrics(results: list[dict]) -> dict:
    """Pure function. Returns a dict of computed metrics; testable in isolation
    without any I/O or rendering. Keys are designed to be stable for the JSON
    output, dashboard wiring, etc."""
    n = len(results)
    if n == 0:
        return {"total": 0}

    passed = [r for r in results if r.get("passed")]
    errored = [r for r in results if r.get("error")]

    # --- Top-K hit rates: did the expected hex appear within the first K? ----
    def hit_at(k: int) -> int:
        hits = 0
        for r in results:
            tol = r.get("tol", float("inf"))
            dists = r.get("all_dists", [r.get("dist", float("inf"))])
            if any(d <= tol for d in dists[:k]):
                hits += 1
        return hits

    top1 = hit_at(1)
    top3 = hit_at(3)
    top5 = hit_at(5)

    # --- Distance distribution -----------------------------------------------
    dists = [r["dist"] for r in results
             if isinstance(r.get("dist"), (int, float))]
    mean_dist = round(statistics.mean(dists), 1) if dists else None
    median_dist = round(statistics.median(dists), 1) if dists else None
    max_dist = round(max(dists), 1) if dists else None

    # --- Latency -------------------------------------------------------------
    latencies = sorted(r.get("latency_ms", 0) for r in results)
    p50 = statistics.median(latencies) if latencies else 0
    p95 = (latencies[int(len(latencies) * 0.95)]
           if len(latencies) > 1 else (latencies[0] if latencies else 0))
    mean_lat = round(statistics.mean(latencies), 1) if latencies else 0

    # --- Tier mix ------------------------------------------------------------
    tier_counts: dict[str, int] = defaultdict(int)
    for r in results:
        tier_counts[str(r.get("tier", "miss"))] += 1
    tier_mix = {t: tier_counts[t] for t in sorted(tier_counts)}

    # --- Per-tier accuracy ---------------------------------------------------
    per_tier_acc: dict[str, dict[str, int]] = {}
    for r in results:
        t = str(r.get("tier", "miss"))
        slot = per_tier_acc.setdefault(t, {"passed": 0, "total": 0})
        slot["total"] += 1
        if r.get("passed"):
            slot["passed"] += 1

    # --- Routing: lookup_resolvable that escaped to Tier 4 -------------------
    lookup = [r for r in results if r.get("split") == "lookup_resolvable"]
    routing_kept_local = sum(
        1 for r in lookup if not str(r.get("tier", "")).startswith("4")
    )
    routing_pct = (routing_kept_local / len(lookup) * 100) if lookup else None
    unnecessary_t4 = len(lookup) - routing_kept_local

    # --- Tier 4 utilization (inverse view of routing) ------------------------
    tier4 = [r for r in results if str(r.get("tier", "")).startswith("4")]
    necessary_t4 = sum(1 for r in tier4 if r.get("split") == "llm_required")
    tier4_efficiency = (necessary_t4 / len(tier4) * 100) if tier4 else None

    # --- LLM-required correctly reaching Tier 4 ------------------------------
    llm_req = [r for r in results if r.get("split") == "llm_required"]
    llm_used = sum(1 for r in llm_req if str(r.get("tier", "")).startswith("4"))

    # --- Confidence calibration ----------------------------------------------
    conf_cases = [r for r in results if r.get("confident")]
    conf_passed = sum(1 for r in conf_cases if r.get("passed"))
    conf_accuracy = (conf_passed / len(conf_cases) * 100) if conf_cases else None

    # --- Failure breakdown ---------------------------------------------------
    wrong_hex = [r for r in results
                  if not r.get("passed") and not r.get("error")
                  and r.get("got") is not None]
    no_result = [r for r in results
                  if not r.get("passed") and not r.get("error")
                  and r.get("got") is None]

    return {
        "total": n,
        "passed": len(passed),
        "accuracy_pct": round(len(passed) / n * 100, 1),
        "errored": len(errored),

        "top1": top1, "top3": top3, "top5": top5,
        "top1_pct": round(top1 / n * 100, 1),
        "top3_pct": round(top3 / n * 100, 1),
        "top5_pct": round(top5 / n * 100, 1),

        "mean_dist": mean_dist,
        "median_dist": median_dist,
        "max_dist": max_dist,

        "latency_p50_ms": int(p50),
        "latency_p95_ms": int(p95),
        "latency_mean_ms": mean_lat,

        "tier_mix": tier_mix,
        "per_tier_accuracy": per_tier_acc,

        "routing_accuracy_pct": round(routing_pct, 1) if routing_pct is not None else None,
        "routing_unnecessary_t4": unnecessary_t4,
        "lookup_resolvable_total": len(lookup),

        "tier4_efficiency_pct": (round(tier4_efficiency, 1)
                                  if tier4_efficiency is not None else None),
        "tier4_calls": len(tier4),
        "tier4_necessary": necessary_t4,

        "llm_required_routed": llm_used,
        "llm_required_total": len(llm_req),

        "confident_accuracy_pct": (round(conf_accuracy, 1)
                                    if conf_accuracy is not None else None),
        "confident_total": len(conf_cases),

        "failures_wrong_hex": len(wrong_hex),
        "failures_no_result": len(no_result),

        "estimated_cost_usd": _estimate_cost_usd(results),
        "models_used": sorted({r.get("model") for r in results if r.get("model")}),
    }


def run(path: Path = DATASET_PATH,
        on_case_done: Callable[[dict], None] | None = None,
        model: str = DEFAULT_MODEL) -> list[dict]:
    """Run every case in the dataset, optionally invoking on_case_done after
    each so callers can render progress."""
    cases = json.loads(Path(path).read_text())
    results: list[dict] = []
    for c in cases:
        result = _evaluate_case(c, model=model)
        results.append(result)
        if on_case_done is not None:
            on_case_done(result)
    return results


def run_with_progress(path: Path = DATASET_PATH,
                      console: Console | None = None,
                      model: str = DEFAULT_MODEL) -> list[dict]:
    """Run() with a live Rich progress bar showing per-case status."""
    console = console or Console()
    cases = json.loads(Path(path).read_text())

    results: list[dict] = []
    columns = [
        SpinnerColumn(style="cyan"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=30, complete_style="green", finished_style="bold green"),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    ]

    pass_count = 0
    fail_count = 0
    with Progress(*columns, console=console, transient=False) as progress:
        task = progress.add_task("[bold]Eval[/]", total=len(cases))
        for c in cases:
            progress.update(task, description=f"[bold]Eval[/] • {c['q'][:40]}")
            result = _evaluate_case(c)
            results.append(result)
            if result.get("passed"):
                pass_count += 1
            else:
                fail_count += 1
            progress.update(
                task,
                advance=1,
                description=(
                    f"[bold]Eval[/] • [green]{pass_count} pass[/] / "
                    f"[red]{fail_count} fail[/]"
                ),
            )
    return results


def _format_tier(tier: str | None) -> Text:
    if tier is None:
        return Text("-", style="dim")
    color = TIER_COLOR.get(tier, "white")
    return Text(tier, style=f"bold {color}")


def _format_pass(passed: bool) -> Text:
    return (Text(" PASS ", style="bold white on green") if passed
            else Text(" FAIL ", style="bold white on red"))


def _swatch(hex_: str | None) -> Text:
    if not hex_:
        return Text("-", style="dim")
    h = hex_.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    fg = "black" if luminance > 140 else "white"
    return Text(f" {hex_} ", style=f"{fg} on rgb({r},{g},{b})")


def report(results: list[dict], console: Console | None = None) -> None:
    """Render the report as Rich panels + tables."""
    console = console or Console()
    metrics = compute_metrics(results)

    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)

    # --- Top headline panel ----------------------------------------------------
    def _color_pct(pct: float | None, good: float = 95, ok: float = 70) -> str:
        if pct is None:
            return "white"
        return "green" if pct >= good else "yellow" if pct >= ok else "red"

    headline = Text.assemble(
        ("Accuracy: ", "white"),
        (f"{metrics['passed']}/{metrics['total']} ", "bold cyan"),
        (f"({metrics['accuracy_pct']}%)", "bold cyan"),
        "  ", ("•", "dim"), "  ",
        ("Top-1/3/5: ", "white"),
        (f"{metrics['top1_pct']}%", _color_pct(metrics['top1_pct'])),
        ("/", "dim"),
        (f"{metrics['top3_pct']}%", _color_pct(metrics['top3_pct'])),
        ("/", "dim"),
        (f"{metrics['top5_pct']}%", _color_pct(metrics['top5_pct'])),
        "\n",
        ("Distance: ", "white"),
        (f"mean {metrics['mean_dist']} ", "magenta"),
        ("• ", "dim"),
        (f"median {metrics['median_dist']} ", "magenta"),
        ("• ", "dim"),
        (f"max {metrics['max_dist']}", "magenta"),
        "\n",
        ("Latency: ", "white"),
        (f"p50 {metrics['latency_p50_ms']}ms ", "cyan"),
        ("• ", "dim"),
        (f"p95 {metrics['latency_p95_ms']}ms ", "cyan"),
        ("• ", "dim"),
        (f"mean {metrics['latency_mean_ms']:.0f}ms", "cyan"),
    )
    if metrics["routing_accuracy_pct"] is not None:
        headline.append("\n")
        headline.append("Routing accuracy: ", style="white")
        headline.append(
            f"{metrics['lookup_resolvable_total'] - metrics['routing_unnecessary_t4']}"
            f"/{metrics['lookup_resolvable_total']} "
            f"({metrics['routing_accuracy_pct']}%)",
            style=f"bold {_color_pct(metrics['routing_accuracy_pct'])}",
        )
        headline.append("  (target ≥95%)", style="dim")
        if metrics["tier4_efficiency_pct"] is not None:
            headline.append("  •  Tier-4 necessity: ", style="white")
            headline.append(
                f"{metrics['tier4_necessary']}/{metrics['tier4_calls']} "
                f"({metrics['tier4_efficiency_pct']}%)",
                style=f"bold {_color_pct(metrics['tier4_efficiency_pct'])}",
            )
    if metrics["llm_required_total"]:
        headline.append("\n")
        headline.append("LLM-required reached Tier 4: ", style="white")
        headline.append(
            f"{metrics['llm_required_routed']}/{metrics['llm_required_total']}",
            style="bold",
        )
    if metrics["confident_accuracy_pct"] is not None:
        headline.append("  •  Confident-call accuracy: ", style="white")
        headline.append(
            f"{metrics['confident_accuracy_pct']}%",
            style=f"bold {_color_pct(metrics['confident_accuracy_pct'])}",
        )
        headline.append(f" ({metrics['confident_total']} calls)", style="dim")
    if metrics["failures_wrong_hex"] or metrics["failures_no_result"] or metrics["errored"]:
        headline.append("\n")
        headline.append("Failures: ", style="white")
        headline.append(f"{metrics['failures_wrong_hex']} wrong hex", style="red")
        headline.append(" • ", style="dim")
        headline.append(f"{metrics['failures_no_result']} no result", style="red")
        headline.append(" • ", style="dim")
        headline.append(f"{metrics['errored']} errors", style="red")
    headline.append("\n")
    headline.append("Estimated LLM cost: ", style="white")
    headline.append(f"${metrics['estimated_cost_usd']:.4f}", style="yellow")
    if metrics.get("models_used"):
        headline.append(f"  (model: {', '.join(metrics['models_used'])})",
                         style="dim")

    console.print(Panel(headline, title="[bold]eval summary[/]",
                         title_align="left", border_style="cyan", box=ROUNDED))

    # --- By-category table -----------------------------------------------------
    cat_table = Table(title="[bold]By category[/]", box=ROUNDED,
                       border_style="grey50",
                       header_style="bold white on grey23",
                       title_style="white", title_justify="left")
    cat_table.add_column("category", style="white")
    cat_table.add_column("passed", justify="right")
    cat_table.add_column("total", justify="right")
    cat_table.add_column("rate", justify="right")
    for cat, rs in sorted(by_cat.items()):
        passed = sum(r["passed"] for r in rs)
        rate = passed / len(rs) if rs else 0
        rate_color = "green" if rate >= 0.85 else "yellow" if rate >= 0.6 else "red"
        cat_table.add_row(
            cat,
            Text(str(passed), style="bold"),
            str(len(rs)),
            Text(f"{rate*100:.0f}%", style=rate_color),
        )
    console.print(cat_table)

    # --- Tier mix + per-tier accuracy ----------------------------------------
    tier_table = Table(title="[bold]By tier[/]", box=ROUNDED,
                        border_style="grey50",
                        header_style="bold white on grey23",
                        title_style="white", title_justify="left")
    tier_table.add_column("tier")
    tier_table.add_column("calls", justify="right")
    tier_table.add_column("share", justify="right")
    tier_table.add_column("accuracy", justify="right")
    for tier_name in sorted(metrics["tier_mix"]):
        n = metrics["tier_mix"][tier_name]
        share = n / metrics["total"] * 100
        slot = metrics["per_tier_accuracy"].get(tier_name, {"passed": 0, "total": 0})
        acc = slot["passed"] / slot["total"] * 100 if slot["total"] else 0
        tier_color = TIER_COLOR.get(tier_name, "white")
        tier_table.add_row(
            Text(tier_name, style=f"bold {tier_color}"),
            str(n),
            Text(f"{share:.0f}%", style="dim"),
            Text(f"{slot['passed']}/{slot['total']} ({acc:.0f}%)",
                  style=("green" if acc >= 85
                          else "yellow" if acc >= 60 else "red")),
        )
    console.print(tier_table)

    # --- Per-case detail table -------------------------------------------------
    detail = Table(title="[bold]Per-case[/]", box=ROUNDED,
                    border_style="grey50",
                    header_style="bold white on grey23",
                    title_style="white", title_justify="left",
                    show_lines=False)
    detail.add_column("status", justify="center", no_wrap=True)
    detail.add_column("query", overflow="fold", max_width=40)
    detail.add_column("tier", no_wrap=True)
    detail.add_column("expected", no_wrap=True)
    detail.add_column("got", no_wrap=True)
    detail.add_column("dist", justify="right")
    detail.add_column("latency", justify="right", style="cyan")

    for r in results:
        dist_val = r.get("dist", "-")
        dist_text = (Text(str(dist_val))
                      if isinstance(dist_val, (int, float))
                      else Text(str(dist_val), style="dim"))
        if isinstance(dist_val, (int, float)):
            tol = r.get("tol", float("inf"))
            dist_text.style = "green" if dist_val <= tol else "red"
        latency = r.get("latency_ms", 0)
        lat_text = Text(f"{latency} ms",
                         style=("dim" if latency < 100 else
                                "cyan" if latency < 1000 else
                                "yellow" if latency < 10_000 else "red"))
        row = [
            _format_pass(bool(r.get("passed"))),
            r["q"],
            _format_tier(r.get("tier")),
            _swatch(r.get("expected")),
            _swatch(r.get("got")),
            dist_text,
            lat_text,
        ]
        detail.add_row(*row)
        if r.get("error"):
            detail.add_row("", Text(f"  err: {r['error']}", style="red dim"),
                            "", "", "", "", "")
    console.print(detail)


def report_plain(results: list[dict]) -> None:
    """Plain-text report. Wraps compute_metrics for --quiet / scripted use."""
    m = compute_metrics(results)
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)

    print(f"\nAccuracy: {m['passed']}/{m['total']} ({m['accuracy_pct']}%)")
    print(f"Top-K hit rate: top1={m['top1_pct']}%  top3={m['top3_pct']}%  top5={m['top5_pct']}%")
    print(f"Distance: mean={m['mean_dist']}  median={m['median_dist']}  max={m['max_dist']}")
    print(f"Latency: p50={m['latency_p50_ms']}ms  p95={m['latency_p95_ms']}ms  mean={m['latency_mean_ms']:.0f}ms")
    if m["routing_accuracy_pct"] is not None:
        kept = m["lookup_resolvable_total"] - m["routing_unnecessary_t4"]
        print(f"Routing accuracy: {kept}/{m['lookup_resolvable_total']} ({m['routing_accuracy_pct']}%)")
    if m["tier4_efficiency_pct"] is not None:
        print(f"Tier-4 necessity: {m['tier4_necessary']}/{m['tier4_calls']} ({m['tier4_efficiency_pct']}%)")
    if m["llm_required_total"]:
        print(f"LLM-required reached Tier 4: {m['llm_required_routed']}/{m['llm_required_total']}")
    if m["confident_accuracy_pct"] is not None:
        print(f"Confident-call accuracy: {m['confident_accuracy_pct']}% ({m['confident_total']} calls)")
    if m["failures_wrong_hex"] or m["failures_no_result"] or m["errored"]:
        print(f"Failures: {m['failures_wrong_hex']} wrong hex, "
              f"{m['failures_no_result']} no result, {m['errored']} errors")
    models = ", ".join(m.get("models_used") or []) or "n/a"
    print(f"Estimated cost: ${m['estimated_cost_usd']:.4f}  (model: {models})")

    print("\nBy category:")
    for cat, rs in sorted(by_cat.items()):
        passed = sum(r["passed"] for r in rs)
        print(f"  {cat:14s} {passed}/{len(rs)}")

    print("\nBy tier:")
    for tier_name in sorted(m["tier_mix"]):
        slot = m["per_tier_accuracy"].get(tier_name, {"passed": 0, "total": 0})
        share = m["tier_mix"][tier_name] / m["total"] * 100
        print(f"  {tier_name:14s} calls={m['tier_mix'][tier_name]:<3} "
              f"share={share:>4.0f}%  accuracy={slot['passed']}/{slot['total']}")

    print("\nPer-case:")
    for r in results:
        mark = "OK" if r.get("passed") else "FAIL"
        line = (f"  [{mark}] {r['q']:42s} tier={r.get('tier','-'):<5} "
                f"got={r.get('got','-')} dist={r.get('dist','-'):>6} "
                f"lat={r['latency_ms']}ms")
        if r.get("error"):
            line += f"  err={r['error']}"
        print(line)
