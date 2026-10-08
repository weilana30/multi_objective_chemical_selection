#!/usr/bin/env python3
"""Rebuild end-of-run Pareto and candidate-flow figures from saved run files."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "iterative_pareto_docking"))
from iterative_dock_pareto import write_run_summary_figures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="A saved run directory containing workflow_summary.json")
    args = parser.parse_args()
    summary_path = args.run_dir / "workflow_summary.json"
    if not summary_path.is_file():
        parser.error(f"No workflow_summary.json found in {args.run_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    evolution, metrics = write_run_summary_figures(args.run_dir, summary)
    print(f"Saved {evolution}")
    print(f"Saved {metrics}")


if __name__ == "__main__":
    main()
