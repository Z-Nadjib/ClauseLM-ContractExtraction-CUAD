"""Build result tables and figures from the evaluation metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

from contract_extractor.evaluation.report import build_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="reports/results")
    parser.add_argument("--figures-dir", default="reports/figures")
    parser.add_argument("--outputs-dir", default="outputs", help="Training runs holding train_summary.json files")
    args = parser.parse_args()
    for name, path in build_report(Path(args.results_dir), Path(args.figures_dir), Path(args.outputs_dir)).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
