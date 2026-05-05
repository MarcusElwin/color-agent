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


def _evaluate_case(case: dict) -> dict:
    """Run one case end-to-end. Pure function — no I/O beyond to_hex itself."""
    t0 = time.time()
    try:
        r = to_hex(case["q"])
        top = r.candidates[0] if r.candidates else None
        d = rgb_distance(top.hex, case["expected"]) if top else float("inf")
        return {
            **case,
            "got": top.hex if top else None,
            "tier": r.tier,
            "dist": round(d, 1),
            "passed": d <= case["tol"],
            "latency_ms": int((time.time() - t0) * 1000),
            "spread": r.spread,
            "k": len(r.candidates),
        }
    except Exception as e:
        return {
            **case, "error": repr(e), "passed": False,
            "latency_ms": int((time.time() - t0) * 1000),
        }


def run(path: Path = DATASET_PATH,
        on_case_done: Callable[[dict], None] | None = None) -> list[dict]:
    """Run every case in the dataset, optionally invoking on_case_done after
    each so callers can render progress."""
    cases = json.loads(Path(path).read_text())
    results: list[dict] = []
    for c in cases:
        result = _evaluate_case(c)
        results.append(result)
        if on_case_done is not None:
            on_case_done(result)
    return results


def run_with_progress(path: Path = DATASET_PATH,
                      console: Console | None = None) -> list[dict]:
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

    by_cat: dict[str, list[dict]] = defaultdict(list)
    by_split: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)
        by_split[r["split"]].append(r)

    total = len(results)
    passed_total = sum(r["passed"] for r in results)
    accuracy = passed_total / total if total else 0.0

    latencies = [r["latency_ms"] for r in results]
    p50 = statistics.median(latencies) if latencies else 0
    p95 = (sorted(latencies)[int(len(latencies) * 0.95)]
           if len(latencies) > 1 else (latencies[0] if latencies else 0))

    lookup = by_split.get("lookup_resolvable", [])
    routing_kept_local = (
        sum(1 for r in lookup if not str(r.get("tier", "")).startswith("4"))
        if lookup else 0
    )
    routing_pct = (routing_kept_local / len(lookup) * 100) if lookup else None

    llm_req = by_split.get("llm_required", [])
    llm_used = sum(1 for r in llm_req if str(r.get("tier", "")).startswith("4"))

    # --- Top headline panel ----------------------------------------------------
    headline = Text.assemble(
        ("Accuracy: ", "white"),
        (f"{passed_total}/{total} ", "bold cyan"),
        (f"({accuracy*100:.0f}%)", "bold cyan"),
        "\n",
        ("Latency: ", "white"),
        (f"p50 {int(p50)}ms ", "magenta"),
        ("• ", "dim"),
        (f"p95 {int(p95)}ms", "magenta"),
    )
    if routing_pct is not None:
        routing_color = ("green" if routing_pct >= 95
                          else "yellow" if routing_pct >= 70
                          else "red")
        headline.append("\n")
        headline.append("Routing accuracy: ", style="white")
        headline.append(
            f"{routing_kept_local}/{len(lookup)} ({routing_pct:.0f}%)",
            style=f"bold {routing_color}",
        )
        headline.append("  (target ≥95%)", style="dim")
    if llm_req:
        headline.append("\n")
        headline.append("LLM-required reached Tier 4: ", style="white")
        headline.append(f"{llm_used}/{len(llm_req)}", style="bold")

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
    """Original plain-text report, kept for --quiet / --json modes."""
    by_cat: dict[str, list[dict]] = defaultdict(list)
    by_split: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)
        by_split[r["split"]].append(r)

    print(f"\nOverall: {sum(r['passed'] for r in results)}/{len(results)}")
    print("\nBy category:")
    for cat, rs in sorted(by_cat.items()):
        passed = sum(r["passed"] for r in rs)
        print(f"  {cat:14s} {passed}/{len(rs)}")

    print("\nRouting accuracy (lookup_resolvable that stayed out of Tier 4):")
    lookup = by_split.get("lookup_resolvable", [])
    if lookup:
        non_llm = sum(1 for r in lookup if not str(r.get("tier", "")).startswith("4"))
        print(f"  {non_llm}/{len(lookup)} = {non_llm/len(lookup)*100:.0f}%")
    llm_req = by_split.get("llm_required", [])
    if llm_req:
        llm_used = sum(1 for r in llm_req if str(r.get("tier", "")).startswith("4"))
        print(f"  llm_required correctly routed to Tier 4: {llm_used}/{len(llm_req)}")

    latencies = [r["latency_ms"] for r in results]
    if latencies:
        latencies.sort()
        p50 = statistics.median(latencies)
        p95 = latencies[int(len(latencies) * 0.95)] if len(latencies) > 1 else latencies[-1]
        print(f"\nLatency p50={p50}ms p95={p95}ms")

    print("\nPer-case:")
    for r in results:
        mark = "OK" if r.get("passed") else "FAIL"
        line = (f"  [{mark}] {r['q']:42s} tier={r.get('tier','-'):<5} "
                f"got={r.get('got','-')} dist={r.get('dist','-'):>6} "
                f"lat={r['latency_ms']}ms")
        if r.get("error"):
            line += f"  err={r['error']}"
        print(line)
