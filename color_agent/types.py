"""Public dataclasses returned by the router.

Score semantics intentionally vary by source — within a single Result the list
is ranked best-first; absolute scores are NOT comparable across sources.

  css                    1.0 for exact match; 1 - rgb_distance/441.67 for KNN pads
  color_pizza_exact      color.pizza similarity field (1.0 for exact)
  color_pizza_fuzzy      color.pizza similarity field (Levenshtein-derived 0..1)
  llm                    {high:0.9, medium:0.7, low:0.5} from model self-report
  llm_consistent         1 - spread/441.67 clipped to [0,1] (overrides self-report)
"""

from dataclasses import dataclass, field


@dataclass
class Candidate:
    hex: str
    name: str
    score: float
    source: str


@dataclass
class Result:
    query: str
    normalized: str
    candidates: list[Candidate]
    confident: bool
    tier: str
    spread: float | None = None
    latency_ms: int = 0
    notes: list[str] = field(default_factory=list)
