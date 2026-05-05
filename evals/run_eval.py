"""Thin shim — keeps `python evals/run_eval.py` working from the repo root.
Real implementation lives in color_agent.eval."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from color_agent.eval import DATASET_PATH, report, run  # noqa: E402

if __name__ == "__main__":
    report(run(DATASET_PATH))
