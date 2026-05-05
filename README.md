# color-agent

Convert natural-language color descriptions to ranked hex candidates with a tiered lookup that only spends LLM calls when it has to.

```
$ color-agent crimson
Tier 1: CSS named (in-process)  •  confident  •  latency 0 ms
 #  hex      score          name              source
 1  #DC143C  ████████████   crimson           css
 2  #B22222  ███████████░   firebrick         css
 3  #A52A2A  ██████████░░   brown             css
 4  #FF0000  ██████████░░   red               css
 5  #C71585  ██████████░░   mediumvioletred   css
```

## Why

Most "color name → hex" requests are deterministic — `crimson` should never hit an LLM. But a long tail (brand colors, descriptive phrases, multilingual queries) genuinely needs reasoning + web search. This tool routes accordingly:

```
query → normalize
  ├─ Tier 1: CSS named colors dict (in-process, ~0ms)
  ├─ Tier 2: color.pizza exact match  (~50ms, free)
  ├─ Tier 3: color.pizza fuzzy match  (~50ms, free)
  └─ Tier 4: LLM agent (~3-30s)
       ├─ base       (high confidence → return)
       ├─ reflect    (medium → second pass)
       └─ consistent (low / brand-y → N=5 sample medoid)
```

Cost shift: if 80% of queries are lookup-resolvable, you go from `1.0 × $X` to `~0.2 × $X` — ~5× cheaper than an LLM-only design at the same output quality.

## Install

```bash
pip install color-agent
```

Or from source:

```bash
git clone https://github.com/MarcusElwin/color-agent
cd color-agent
pip install -e ".[dev]"
```

### API key

Tier 4 (the LLM fallback) needs an Anthropic API key. Tiers 1–3 work without one — `crimson`, `cobalt blue`, `#0047AB` all resolve offline-ish.

Two ways to provide it:

```bash
# 1. Export it in your shell
export ANTHROPIC_API_KEY=sk-ant-...

# 2. Or drop a .env file in the directory you run from
cp .env.example .env
# edit .env and add your key
```

The CLI auto-loads a `.env` from the current working directory (or any parent) on first Tier 4 call. Existing env vars take precedence over `.env` — the shell wins ties. Get a key at https://console.anthropic.com/settings/keys.

## Usage

```bash
color-agent crimson                          # tier 1, instant
color-agent "cobalt blue"                    # tier 2/3 via color.pizza
color-agent "Pantone 1837 Tiffany blue"      # tier 4 (LLM)
color-agent "muted forest green" --fast      # tier 4 with Haiku 4.5 (~3× cheaper)

color-agent crimson --json                   # machine-readable output
color-agent crimson -k 8                     # 8 candidates instead of 5
color-agent x --force tier4_consistent       # bypass auto-routing
```

The CLI shows a live spinner with phase names (`Tier 4 base • web_search step • claude-sonnet-4-6  (8.3s)`) so you can tell it's alive during the slower LLM paths.

### Eval harness

```bash
color-agent-eval                             # styled report with progress bar
color-agent-eval --quiet                     # plain-text
color-agent-eval --json | jq                 # JSON for piping
color-agent-eval --dataset path/to/cases.json
```

Metrics computed (all in `compute_metrics(results)`, pure-functional):

| Metric | What it tells you |
|---|---|
| `accuracy_pct` | Top-1 candidate within the case's tolerance |
| `top1_pct` / `top3_pct` / `top5_pct` | How often the right answer is in the first K candidates — proves the K=5 list earns its place |
| `mean_dist` / `median_dist` / `max_dist` | RGB-distance distribution across cases — accuracy hides the gap between dist=0 and dist=51 |
| `latency_p50_ms` / `latency_p95_ms` / `latency_mean_ms` | Tier 1 is sub-millisecond; Tier 4 is 20–40s. Median tracks escalation rate. |
| `routing_accuracy_pct` | % of `lookup_resolvable` cases that did NOT escape to Tier 4. **Target ≥95%** — the single most important metric. |
| `tier4_efficiency_pct` | Inverse view: of all Tier 4 calls actually made, how many were necessary. |
| `tier_mix` / `per_tier_accuracy` | Calls per tier + accuracy at each tier — find weak tiers in isolation. |
| `confident_accuracy_pct` | When the system flags `confident=True`, how often is it actually right? Calibration check. |
| `failures_wrong_hex` / `failures_no_result` / `errored` | Failure breakdown — wrong answer vs. nothing returned vs. exception are different bugs. |
| `estimated_cost_usd` | Rough $ for the run (Tier 4 only; Sonnet 4.6 pricing). Approximation, useful for relative comparison across runs. |

Each category has its own RGB-distance tolerance: tight (≤5 units) for canonical CSS names, generous (≤80) for descriptive/multilingual where multiple plausible hexes are all reasonable.

The dataset (`evals/dataset.json`) covers 34 cases across 7 categories: `css_named`, `standard`, `fuzzy_name`, `brand`, `descriptive`, `multilingual`, `disambiguation`, `bare_hex`.

## Python API

```python
from color_agent.router import to_hex

result = to_hex("cobalt blue")
# Result(
#   query='cobalt blue', tier='2', confident=True, latency_ms=42,
#   candidates=[Candidate(hex='#0047AB', name='Cobalt Blue', score=1.0, source='color_pizza_exact'), ...]
# )

# Optional progress callback for long-running Tier 4 paths
def log(phase: str) -> None:
    print(phase)

result = to_hex("Pantone 1837", on_progress=log)
```

`Result.candidates` is always a length-K (default 5) list ranked best-first.

## Architecture notes

- **148 CSS named colors** embedded in-process (`color_agent/css_colors.py`); the 7 British "grey" spellings are aliased to American forms in `normalize.py`, so the dict holds 141 canonical keys.
- **color.pizza** is queried with a 30-day SQLite cache (`color_agent/color_pizza.py`). Failures fail-soft to the LLM tier.
- **Tier 4 base agent** uses a deliberate **two-step Anthropic call**: the first with `tool_choice=auto` so `web_search` can run, the second with `tool_choice` forced to `return_hex_list` so we get structured candidates. Forcing the tool in step 1 prefills the assistant turn and prevents `web_search` from running first — the single biggest gotcha if you build something similar.
- **Self-consistency** runs N=5 parallel samples, picks the medoid (robust to outliers — one sample saying `#FF0000` against four cobalt samples loses), and uses pairwise spread as the confidence signal.
- **Web-search tool version**: Sonnet 4.6 / Opus 4.7 use `web_search_20260209`; Haiku 4.5 uses `web_search_20250305`. Selected automatically per `MODEL_CONFIGS` in `agent.py`.

## Output contract

```python
@dataclass
class Candidate:
    hex: str        # "#0047AB"
    name: str       # canonical from source
    score: float    # 0..1, semantics depend on source
    source: str     # "css" | "color_pizza_exact" | "color_pizza_fuzzy" | "llm_*"

@dataclass
class Result:
    query: str
    normalized: str
    candidates: list[Candidate]   # length >= K, ranked best-first
    confident: bool               # caller can ignore the list if True
    tier: str                     # "1" | "2" | "3" | "4-base" | "4-reflect" | "4-consistent" | "hex"
    spread: float | None          # only set on tier 4-consistent
    latency_ms: int
```

Score semantics intentionally vary by source — within a single result the list is ranked best-first, but absolute scores are NOT comparable across sources.

## Development

```bash
pip install -e ".[dev]"
pytest                       # 75 mocked + unit tests
pytest -m live               # live tests (needs ANTHROPIC_API_KEY)
```

## Credits

- [color.pizza](https://github.com/meodai/color-name-api) — the ~32k named-color database that powers Tiers 2/3.
- [CSS Color Module Level 4 §6.4](https://www.w3.org/TR/css-color-4/#named-colors) — the 148 CSS named colors.

## License

MIT — see [LICENSE](./LICENSE).
