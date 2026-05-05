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
  ├─ Tier 1:   CSS named colors dict (in-process, ~0ms)
  ├─ Tier 2.5: local 32k color-name dict (in-process, ~50-300ms)
  ├─ Tier 2:   color.pizza exact match  (~50ms, free, network)
  ├─ Tier 3:   color.pizza fuzzy match  (~50ms, free, network)
  └─ Tier 4:   LLM agent (~3-30s)
       ├─ base       (high confidence → return)
       ├─ reflect    (medium → second pass)
       └─ consistent (low / brand-y → N=5 sample medoid)
```

The local 32k dictionary (Tier 2.5) is mirrored from [meodai/color-name-list](https://github.com/meodai/color-names) — MIT-licensed, ships with the package, ~650 KB. Critical because color.pizza's free public API rate-limits or 403s commercial-feel traffic, which used to push every "cobalt blue" / "burnt sienna" query to the LLM (~30 seconds).

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

#### Latest run

13 cases, all passing accuracy thresholds:

```
╭─ eval summary ──────────────────────────────────────────────────────────────╮
│ Accuracy: 13/13 (100%)                                                      │
│ Latency: p50 25268ms • p95 40837ms                                          │
│ Routing accuracy: 4/7 (57%)  (target ≥95%)                                  │
│ LLM-required reached Tier 4: 6/6                                            │
╰─────────────────────────────────────────────────────────────────────────────╯
By category
╭──────────────┬────────┬───────┬──────╮
│ category     │ passed │ total │ rate │
├──────────────┼────────┼───────┼──────┤
│ brand        │      3 │     3 │ 100% │
│ css_named    │      4 │     4 │ 100% │
│ descriptive  │      2 │     2 │ 100% │
│ multilingual │      1 │     1 │ 100% │
│ standard     │      3 │     3 │ 100% │
╰──────────────┴────────┴───────┴──────╯
Per-case (selected)
  PASS  crimson                           tier=1         #DC143C  → #DC143C    0.0      0 ms
  PASS  rebeccapurple                     tier=1         #663399  → #663399    0.0      0 ms
  PASS  Cobalt-Blue!                      tier=4-base    #0047AB  → #0047AB    0.0  23918 ms  ← should be tier 2/3
  PASS  burnt sienna                      tier=4-base    #E97451  → #E97451    0.0  25855 ms  ← should be tier 2/3
  PASS  salmon pink                       tier=4-reflect #FF91A4  → #FF91A4    0.0  32429 ms  ← should be tier 2/3
  PASS  International Klein Blue          tier=4-base    #002FA7  → #002FA7    0.0  30535 ms
  PASS  Hermès orange                     tier=4-base    #FF7900  → #F37021   36.2  40837 ms
  PASS  Tiffany blue                      tier=4-base    #0ABAB5  → #0ABAB5    0.0  31555 ms
  PASS  the color of a flamingo at sunset tier=4-base    #FC8EAC  → #FF8C8C   32.2  28541 ms
  PASS  muted forest green                tier=4-base    #4F7942  → #4A6741   18.7  25268 ms
  PASS  rött koppar                       tier=4-base    #B87333  → #B5461B   51.1  22824 ms
```

> Note: this output is from the original 13-case dataset before [PR #3](https://github.com/MarcusElwin/color-agent/pull/3) grew it to 34. Re-run `color-agent-eval` for the current numbers — the new metrics (top-K hit rate, tier mix, confident-call accuracy, $ cost) only show up on the new dataset.

**Honest take:** accuracy is 13/13, but **routing accuracy is 57%** — three `lookup_resolvable` queries (`Cobalt-Blue!`, `burnt sienna`, `salmon pink`) escaped to Tier 4 and burned ~25–32s of LLM time when they should have been served by color.pizza in <500ms. The router is failing soft to the LLM when color.pizza returns errors (we've seen 403s from some networks). Two follow-ups are needed: improve the color.pizza error handling so we don't bypass Tier 3 on transient failures, and lower the Tier 3 fuzzy threshold so partial-match standard queries land there. Tracking this in [TODO](#todo--known-issues).

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

## Architecture

The whole point is to keep cheap deterministic lookups in front of expensive LLM calls. The router walks the tiers in order; only queries the dictionaries can't handle reach the agent.

### Full pipeline

```
                       ┌──────────────────────┐
   raw query  ────────▶│  normalize.py         │  lower-case, strip punct,
                       │  parse_hex            │  collapse ws, grey→gray,
                       └────────┬──────────────┘  detect bare hex
                                │
                                ▼
            ┌───────────────────────────────────────────┐
            │   bare hex?                               │  yes → color.pizza
            │   (#0047AB / 0047AB)                      │       hex_lookup
            └────────────┬──────────────────────────────┘       (named neighbors)
                         │ no
                         ▼
            ┌───────────────────────────────────────────┐
            │  Tier 1 — css_colors.py (in-process)      │  ~0 ms, free
            │  141 canonical CSS names; KNN-pad to K    │
            └────────────┬──────────────────────────────┘
                         │ miss
                         ▼
            ┌───────────────────────────────────────────┐
            │  Tier 2/3 — color_pizza.py + tier23.py    │  ~50 ms, free
            │  SQLite-cached (30-day TTL)               │
            │   sim == 1.0     → Tier 2 (exact)         │
            │   sim ≥  0.85    → Tier 3 (fuzzy)         │
            │   sim ≥  0.92    → confident=True         │
            └────────────┬──────────────────────────────┘
                         │ miss / not confident / API error
                         ▼
       ┌────────────────────────────────────────────────────────────┐
       │                    Tier 4 — LLM agent                      │
       │                                                            │
       │   query has brand/Pantone/PMS hints?                       │
       │           ├── yes ──▶ consistent (skip base + reflect)     │
       │           └── no  ──▶ base                                 │
       │                                                            │
       │   ┌──────────────────────────────────────────────────────┐ │
       │   │  base — agent.py (two-step!)                         │ │
       │   │  step 1: tool_choice=auto, web_search may run        │ │
       │   │  step 2: tool_choice=force return_hex_list           │ │
       │   │  ↓ overall_confidence                                │ │
       │   │  high   → return                                     │ │
       │   │  medium → reflect                                    │ │
       │   │  low    → consistent                                 │ │
       │   └──────────────────────────────────────────────────────┘ │
       │                                                            │
       │   ┌──────────────────────────────────────────────────────┐ │
       │   │  reflect — reflect.py                                │ │
       │   │  Single critique pass; same-model Sonnet by default, │ │
       │   │  reviewer kwarg lets you A/B Opus 4.7.               │ │
       │   └──────────────────────────────────────────────────────┘ │
       │                                                            │
       │   ┌──────────────────────────────────────────────────────┐ │
       │   │  consistent — consistency.py                         │ │
       │   │  N=5 parallel samples (ThreadPool default; Batches   │ │
       │   │  API path available for −50% non-latency-sensitive). │ │
       │   │  medoid winner (robust to outliers); pairwise spread │ │
       │   │  overrides model self-reported confidence.           │ │
       │   └──────────────────────────────────────────────────────┘ │
       └────────────────────────────────────────────────────────────┘
                         │
                         ▼
                Result(candidates=[...]≥K, tier, confident, spread, latency_ms)
```

### Module map

| File                       | Responsibility                                             |
|----------------------------|------------------------------------------------------------|
| `types.py`                 | `Candidate`, `Result` dataclasses + score-semantics doc    |
| `normalize.py`             | Single canonical key for cache hits + bare-hex detection   |
| `css_colors.py` + `tier1.py` | 141 CSS named colors + KNN-pad to K candidates           |
| `distance.py`              | RGB distance, hex↔rgb, medoid, KNN                          |
| `color_pizza.py`           | HTTP client + SQLite write-through cache (30-day TTL)      |
| `tier23.py`                | Maps color.pizza response → ranked `Candidate` list        |
| `agent.py`                 | Two-step Tier 4 base; `MODEL_CONFIGS`; `return_hex_list`   |
| `reflect.py`               | Single-pass critique; same-model or reviewer override      |
| `consistency.py`           | N=5 sampling, medoid winner, spread-derived confidence     |
| `router.py`                | Orchestrates the tiers + Tier 4 sub-routing + `--force`    |
| `prompts.py`               | All system prompts in one place                            |
| `cli.py`                   | Click + Rich CLI (banner, table, swatches, live spinner)   |
| `eval.py`                  | Eval runner + reporter; `--json` / `--quiet` / Rich modes  |

### Things worth knowing if you build something similar

- **The two-step Tier 4 call is non-negotiable.** Forcing `tool_choice={type:"tool",name:"return_hex_list"}` prefills the assistant turn and prevents the model from calling `web_search` first. So step 1 runs with `tool_choice=auto` (search may run, citations come back), step 2 appends that turn and forces the structured return. Merging them was my first attempt and it silently never searched.
- **Web-search tool version is model-coupled.** Sonnet 4.6 / Opus 4.7 use `web_search_20260209` (dynamic filtering, server auto-injects `code_execution`). Haiku 4.5 uses the older `web_search_20250305`. Adding `code_execution` manually to the tools array on the new version returns `400 invalid_request_error: tool name conflict` — the API auto-injects it.
- **Medoid > centroid for self-consistency.** With samples `[#0047AB, #0050B0, #0045A8, #0047AB, #FF0000]`, the centroid would be `~#332E89` — a never-output color. The medoid picks an actual sample and is robust to the one outlier.
- **British greys are aliased before lookup.** The CSS spec defines 148 names but 7 are British spellings of greys. `normalize.py` collapses them so the in-process dict holds 141 canonical American keys.
- **Tier 3 fail-soft escalates to Tier 4 on color.pizza errors.** Currently this is too aggressive — see eval routing accuracy above.

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

## TODO / known issues

- [x] **Routing accuracy fix** ([PR](https://github.com/MarcusElwin/color-agent/pulls?q=is%3Apr+routing-accuracy)): typed `ColorPizzaTransientError` / `ColorPizzaPermanentError`, retry-with-backoff for 403/429/5xx, fuzzy floor lowered from 0.85 → 0.65, RGB-cluster-tightness rescues mid-similarity matches that converge on a single hex, transient errors now downgrade `confident=False` instead of silently masking. Re-run `color-agent-eval` to see the new routing accuracy.
- [ ] Grow the eval dataset: more multilingual cases (currently 1/13), disambiguation pairs ("the green Stripe used in 2023" vs current), bare-hex inputs (`#0047AB`), and intentional negative cases (`xyzzy`).
- [ ] Cache the Tier 4 LLM responses too — repeat queries shouldn't pay the 25-second cost twice.
- [ ] Add a `--top-k` aware Tier 4 prompt so the model can return fewer candidates when the user explicitly asked for fewer.

## Credits

- [color.pizza](https://github.com/meodai/color-name-api) — the ~32k named-color database that powers Tiers 2/3.
- [CSS Color Module Level 4 §6.4](https://www.w3.org/TR/css-color-4/#named-colors) — the 148 CSS named colors.

## License

MIT — see [LICENSE](./LICENSE).
