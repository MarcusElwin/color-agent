"""Click CLI. Run:
  color-agent "cobalt blue"
  color-agent "Pantone 1837" --json
  color-agent "crimson" -k 8 --force tier1
  color-agent-eval                # runs the eval harness
"""

from __future__ import annotations

import json as jsonlib
from dataclasses import asdict
from pathlib import Path

import threading
import time

import click
from rich.box import ROUNDED
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from color_agent.agent import DEFAULT_MODEL, FAST_MODEL
from color_agent.router import to_hex


# ASCII banner shown above --help. Each row of the wordmark uses a different
# hue from a rainbow gradient so the tool's job (colour) is obvious instantly.
BANNER_LINES = [
    ("  ▄████▄   ▒█████   ██▓     ▒█████   █    ██  ██▀███  ", "bright_red"),
    (" ▒██▀ ▀█  ▒██▒  ██▒▓██▒    ▒██▒  ██▒ ██  ▓██▒▓██ ▒ ██▒", "bright_yellow"),
    (" ▒▓█    ▄ ▒██░  ██▒▒██░    ▒██░  ██▒▓██  ▒██░▓██ ░▄█ ▒", "bright_green"),
    (" ▒▓▓▄ ▄██▒▒██   ██░▒██░    ▒██   ██░▓▓█  ░██░▒██▀▀█▄  ", "bright_cyan"),
    (" ▒ ▓███▀ ░░ ████▓▒░░██████▒░ ████▓▒░▒▒█████▓ ░██▓ ▒██▒", "bright_blue"),
    (" ░ ░▒ ▒  ░░ ▒░▒░▒░ ░ ▒░▓  ░░ ▒░▒░▒░ ░▒▓▒ ▒ ▒ ░ ▒▓ ░▒▓░", "bright_magenta"),
    ("       text  →  hex  •  tiered lookup  +  LLM fallback", "white"),
]


def _print_banner(console: Console | None = None) -> None:
    console = console or Console()
    for line, style in BANNER_LINES:
        console.print(line, style=style, highlight=False)


class _BannerCommand(click.Command):
    """Click Command that prints the Rich ASCII banner above --help output."""

    def format_help(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        # Render the banner straight to stdout via Rich, then let Click write
        # the standard help body underneath.
        _print_banner()
        click.echo()
        super().format_help(ctx, formatter)


# Foreground color (for badges / accents) chosen per tier so the user can spot
# at a glance which path the answer came from.
TIER_THEME = {
    "1":             ("bold green",   "Tier 1: CSS named (in-process)"),
    "2":             ("bold cyan",    "Tier 2: color.pizza exact"),
    "3":             ("bold yellow",  "Tier 3: color.pizza fuzzy"),
    "4-base":        ("bold magenta", "Tier 4: LLM base"),
    "4-reflect":     ("bold magenta", "Tier 4: LLM reflected"),
    "4-consistent":  ("bold magenta", "Tier 4: LLM self-consistency"),
    "hex":           ("bold blue",    "Bare-hex passthrough"),
    "miss":          ("bold red",     "No match"),
    "1-miss":        ("bold red",     "No match"),
}


def _hex_to_rgb_tuple(hex_: str) -> tuple[int, int, int]:
    h = hex_.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _luminance(r: int, g: int, b: int) -> float:
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _swatch_text(hex_: str, width: int = 6) -> Text:
    """Solid block colored with the hex itself, plus the hex value layered on
    top in a contrast-aware foreground."""
    r, g, b = _hex_to_rgb_tuple(hex_)
    fg = "black" if _luminance(r, g, b) > 140 else "white"
    bg = f"on rgb({r},{g},{b})"
    label = f" {hex_} ".center(width + len(hex_))
    return Text(label, style=f"{fg} {bg}")


def _score_bar(score: float, width: int = 12) -> Text:
    score = max(0.0, min(1.0, score))
    filled = int(round(score * width))
    bar = "█" * filled + "░" * (width - filled)
    color = "green" if score >= 0.85 else "yellow" if score >= 0.6 else "red"
    return Text.assemble((bar, color), " ", (f"{score:.2f}", "dim"))


def _render_human(result, console: Console) -> None:
    tier_color, tier_label = TIER_THEME.get(result.tier, ("white", result.tier))
    confidence_label = (
        Text("confident", style="bold green") if result.confident
        else Text("uncertain", style="bold yellow")
    )

    header = Text.assemble(
        "query: ", (f"{result.query!r}", "bold white"),
        "   normalized: ", (f"{result.normalized!r}", "white"),
    )
    meta = Text.assemble(
        (tier_label, tier_color),
        "  •  ", confidence_label,
        "  •  latency ", (f"{result.latency_ms} ms", "cyan"),
    )
    if result.spread is not None:
        meta.append("  •  spread ")
        meta.append(f"{result.spread}", style="magenta")

    console.print(Panel(
        Text.assemble(header, "\n", meta),
        title=Text("colour-agent", style="bold white on blue"),
        title_align="left",
        border_style=tier_color.split()[-1],  # plain color for the border
        box=ROUNDED,
    ))

    if not result.candidates:
        console.print("[red](no candidates)[/red]")
        return

    table = Table(box=ROUNDED, show_lines=False, expand=False,
                  border_style="grey50", header_style="bold white on grey23")
    table.add_column("#", justify="right", style="dim", width=2)
    table.add_column("swatch", no_wrap=True)
    table.add_column("score", no_wrap=True)
    table.add_column("name", overflow="fold", max_width=32)
    table.add_column("source", style="dim")

    for i, c in enumerate(result.candidates, 1):
        rank_style = "bold white on green" if i == 1 else "white"
        table.add_row(
            Text(str(i), style=rank_style),
            _swatch_text(c.hex),
            _score_bar(c.score),
            c.name or "—",
            c.source,
        )

    console.print(table)


def _render_plain(result) -> str:
    lines: list[str] = []
    lines.append(
        f"query={result.query!r}  normalized={result.normalized!r}  "
        f"tier={result.tier}  confident={result.confident}  "
        f"latency={result.latency_ms}ms"
        + (f"  spread={result.spread}" if result.spread is not None else "")
    )
    if not result.candidates:
        lines.append("  (no candidates)")
        return "\n".join(lines)
    lines.append("")
    header = f"  {'#':<2} {'hex':<8} {'score':<6} {'name':<30} source"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for i, c in enumerate(result.candidates, 1):
        lines.append(
            f"  {i:<2} {c.hex:<8} {c.score:<6.3f} {c.name[:30]:<30} {c.source}"
        )
    return "\n".join(lines)


class _LiveSpinner:
    """Rich Status spinner that re-renders every ~120ms so the elapsed counter
    keeps ticking while we're blocked on a single long call (LLM step)."""

    def __init__(self, console: Console):
        self.console = console
        self.phase = "Starting"
        self.t0 = time.monotonic()
        self._status = console.status(self._render(), spinner="dots")
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _render(self) -> Text:
        elapsed = time.monotonic() - self.t0
        return Text.assemble(
            (self.phase, "bold cyan"),
            "  ",
            (f"({elapsed:5.1f}s)", "dim"),
        )

    def _ticker(self) -> None:
        while not self._stop.is_set():
            self._status.update(self._render())
            self._stop.wait(0.12)

    def update(self, phase: str) -> None:
        self.phase = phase
        self._status.update(self._render())

    def __enter__(self) -> "_LiveSpinner":
        self._status.__enter__()
        self._thread = threading.Thread(target=self._ticker, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=0.5)
        self._status.__exit__(*exc)


@click.command(cls=_BannerCommand,
               context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("query", required=True)
@click.option("-k", "--top-k", default=5, show_default=True,
              help="Number of candidates to return.")
@click.option("--force", type=click.Choice([
    "tier1", "tier2_3", "tier4_base", "tier4_reflect", "tier4_consistent",
]), default=None, help="Bypass auto-routing and force a specific layer.")
@click.option("--model", default=DEFAULT_MODEL, show_default=True,
              help="Model used for Tier 4 LLM calls (overridden by --fast).")
@click.option("--fast", is_flag=True,
              help=f"Route Tier 4 to {FAST_MODEL} (~3x cheaper, faster, "
                   "weaker on brand reasoning).")
@click.option("--json", "as_json", is_flag=True,
              help="Emit machine-readable JSON instead of a table.")
@click.option("--no-color", is_flag=True, help="Disable ANSI color swatches.")
@click.option("--quiet", is_flag=True, help="Suppress the progress spinner.")
def cli(query: str, top_k: int, force: str | None, model: str,
        fast: bool, as_json: bool, no_color: bool, quiet: bool) -> None:
    """Convert a color description to ranked hex candidates."""
    if fast:
        model = FAST_MODEL

    show_spinner = not (as_json or quiet or no_color)

    if show_spinner:
        console = Console()
        with _LiveSpinner(console) as spin:
            result = to_hex(query, k=top_k, force=force, model=model,
                             on_progress=spin.update)
    else:
        result = to_hex(query, k=top_k, force=force, model=model)

    if as_json:
        payload = {
            "query": result.query,
            "normalized": result.normalized,
            "tier": result.tier,
            "confident": result.confident,
            "spread": result.spread,
            "latency_ms": result.latency_ms,
            "candidates": [asdict(c) for c in result.candidates],
        }
        click.echo(jsonlib.dumps(payload, indent=2))
    elif no_color:
        click.echo(_render_plain(result))
    else:
        Console().print()
        _render_human(result, Console())


class _EvalBannerCommand(click.Command):
    def format_help(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        _print_banner()
        click.echo()
        super().format_help(ctx, formatter)


@click.command(cls=_EvalBannerCommand,
               context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--dataset", type=click.Path(exists=True, dir_okay=False),
              default=None, help="Override the default eval dataset path.")
@click.option("--model", default=DEFAULT_MODEL, show_default=True,
              help="Model used for Tier 4 LLM calls (overridden by --fast).")
@click.option("--fast", is_flag=True,
              help=f"Route Tier 4 to {FAST_MODEL} (~3x cheaper, faster, "
                   "weaker on brand reasoning).")
@click.option("--json", "as_json", is_flag=True,
              help="Emit raw JSON results instead of styled report.")
@click.option("--quiet", is_flag=True,
              help="Plain-text output, no progress bar.")
def eval_cli(dataset: str | None, model: str, fast: bool,
             as_json: bool, quiet: bool) -> None:
    """Run the eval harness across the dataset."""
    from color_agent.eval import (
        DATASET_PATH, report, report_plain, run, run_with_progress,
    )

    if fast:
        model = FAST_MODEL

    path = Path(dataset) if dataset else DATASET_PATH

    if as_json:
        from color_agent.eval import compute_metrics
        results = run(path, model=model)
        click.echo(jsonlib.dumps(
            {"metrics": compute_metrics(results), "results": results},
            indent=2, default=str,
        ))
        return

    if quiet:
        results = run(path, model=model)
        report_plain(results)
        return

    console = Console()
    _print_banner(console)
    console.print()
    results = run_with_progress(path, console=console, model=model)
    console.print()
    report(results, console=console)


if __name__ == "__main__":
    cli()
