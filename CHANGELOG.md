# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of the tiered text-to-hex color agent.
- Tier 1: in-process CSS named colors lookup (141 American spellings, 7 British greys aliased).
- Tier 2/3: `color.pizza` name search with SQLite cache (30-day TTL).
- Tier 4: Anthropic LLM agent with two-step `web_search` + forced `return_hex_list` tool.
- Self-reflection layer (same-model Sonnet by default; Opus 4.7 reviewer A/B via kwarg).
- Self-consistency layer (N=5 parallel samples, medoid winner, pairwise-spread confidence).
- Router with auto-routing across tiers; `--force` flag for evals.
- `color-agent` CLI with Rich-rendered output, ANSI swatches, score bars, live progress spinner.
- `color-agent-eval` CLI with progress bar, styled summary panel, by-category and per-case tables.
- 75 mocked + unit tests. 8 live tests gated by the `live` marker and `ANTHROPIC_API_KEY`.
- Eval dataset with 13 cases across 5 categories, split into `lookup_resolvable` / `llm_required`.
