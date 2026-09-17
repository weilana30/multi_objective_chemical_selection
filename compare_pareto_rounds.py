#!/usr/bin/env python3
"""Plot the Pareto fronts from rounds 00 and 01 of an iterative run."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = SCRIPT_DIR / "pareto_results" / "run_02"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--round-00",
        type=Path,
        default=DEFAULT_RUN_DIR / "round_00_pareto.csv",
        help="Round 00 Pareto CSV.",
    )
    parser.add_argument(
        "--round-01",
        type=Path,
        default=DEFAULT_RUN_DIR / "round_01_pareto.csv",
        help="Round 01 Pareto CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RUN_DIR / "round_00_01_pareto_comparison.png",
        help="Output PNG path.",
    )
    return parser.parse_args()


def load_front(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"lipophilicity_score", "solubility_score"}
    missing = required.difference(frame.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"{path} is missing required columns: {missing_text}")
    return frame.dropna(subset=required)


def plot_fronts(round_00: pd.DataFrame, round_01: pd.DataFrame, output: Path) -> None:
    with TemporaryDirectory(prefix="compare-pareto-matplotlib-") as config_dir:
        previous_config_dir = os.environ.get("MPLCONFIGDIR")
        os.environ["MPLCONFIGDIR"] = config_dir
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            figure, axis = plt.subplots(figsize=(10, 7))
            for frame, label, color, marker in (
                (round_00, "Round 00", "tab:blue", "o"),
                (round_01, "Round 01", "tab:orange", "s"),
            ):
                ordered = frame.sort_values("lipophilicity_score")
                axis.plot(
                    ordered["lipophilicity_score"],
                    ordered["solubility_score"],
                    color=color,
                    linestyle="--",
                    linewidth=1,
                    alpha=0.8,
                )
                axis.scatter(
                    ordered["lipophilicity_score"],
                    ordered["solubility_score"],
                    color=color,
                    marker=marker,
                    edgecolor="black",
                    label=f"{label} ({len(ordered)} molecules)",
                    zorder=3,
                )

            axis.set_xlabel("Lipophilicity score (higher is better)")
            axis.set_ylabel("Solubility score (higher is better)")
            axis.set_title("Pareto Front Comparison: Round 00 vs Round 01")
            axis.grid(alpha=0.2)
            axis.legend()
            figure.tight_layout()
            output.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(output, dpi=200)
            plt.close(figure)
        finally:
            if previous_config_dir is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = previous_config_dir


def main() -> None:
    args = parse_args()
    round_00 = load_front(args.round_00)
    round_01 = load_front(args.round_01)
    if round_00.empty or round_01.empty:
        raise ValueError("Both Pareto CSVs must contain at least one complete row")
    plot_fronts(round_00, round_01, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
