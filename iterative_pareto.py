#!/usr/bin/env python3
"""Iteratively generate, score, and refine molecules on a Pareto front."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
REMASK_SCRIPT = PROJECT_DIR.parent / "genmol_molecule_generator" / "remask_generated_molecule.py"
AGGREGATE_SCRIPT = PROJECT_DIR / "aggregate_pareto.py"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "pareto_results" / "iterative"

LIPOPHILICITY_PROPERTIES = [
    "Admet-AI:logP", "Admet-AI:Lipophilicity_AstraZeneca",
    "Admetica:lipophilicity", "Admethyst:Lipophilicity_AstraZeneca",
    "BoiledEgg:WLogP", "ClassicalGSG:LogP_MMFF", "MolProperty:logP",
]
SOLUBILITY_PROPERTIES = [
    "Admet-AI:Solubility_AqSolDB", "Admet-AI:HydrationFreeEnergy_FreeSolv",
    "Admetica:solubility", "Admethyst:Solubility_AqSolDB",
    "Admethyst:HydrationFreeEnergy_FreeSolv", "MolProperty:logS",
]
PORT_ARGS = [
    "--admet-ai-port", "18019", "--admetica-port", "8000",
    "--admethyst-port", "18001", "--boiled-egg-port", "18002",
    "--classical-gsg-port", "10001", "--mol-property-port", "18003",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", "-n", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples-per-mask", type=int, default=10)
    parser.add_argument("--min-mask-tokens", type=int, default=5)
    parser.add_argument("--max-mask-tokens", type=int, default=15)
    return parser.parse_args()


def run_remask(smiles: str | None, output_json: Path, args: argparse.Namespace) -> list[str]:
    project_root = REMASK_SCRIPT.parent
    bundled_python = project_root / ".conda-env" / "bin" / "python"
    interpreter = str(bundled_python) if bundled_python.is_file() else sys.executable
    command = [interpreter, str(REMASK_SCRIPT), "--output", str(output_json)]
    if args.model_path:
        command += ["--model-path", str(args.model_path)]
    command += ["--device", args.device, "--seed", str(args.seed)]
    command += [
        "--samples-per-mask", str(args.samples_per_mask),
        "--min-mask-tokens", str(args.min_mask_tokens),
        "--max-mask-tokens", str(args.max_mask_tokens),
    ]
    if smiles is not None:
        command += ["--smiles", smiles]
    environment = None
    if bundled_python.is_file():
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(project_root / "genmol" / "src")
    completed = subprocess.run(
        command, check=True, text=True, stdout=subprocess.PIPE, env=environment
    )
    try:
        result_text = output_json.read_text(encoding="utf-8")
        payload = json.loads(result_text)
    except json.JSONDecodeError as exc:
        if completed.stdout.strip():
            payload = json.loads(completed.stdout)
        else:
            raise RuntimeError(f"Remasking output is not valid JSON: {exc}") from exc
    return list(dict.fromkeys(payload.get("all_unique_generated_variants", [])))


def write_candidates(smiles: list[str], path: Path, round_index: int) -> None:
    pd.DataFrame({
        "candidate_id": [f"R{round_index:02d}_{i:05d}" for i in range(len(smiles))],
        "preferred_name": [f"generated_{round_index}_{i}" for i in range(len(smiles))],
        "smiles": smiles,
    }).to_csv(path, index=False)


def run_aggregate(input_csv: Path, output_csv: Path, output_image: Path) -> pd.DataFrame:
    command = [
        sys.executable, str(AGGREGATE_SCRIPT),
        "--input-csv", str(input_csv), "--output-csv", str(output_csv),
        "--output-image", str(output_image), *PORT_ARGS,
        "--lipophilicity-properties", *LIPOPHILICITY_PROPERTIES,
        "--solubility-properties", *SOLUBILITY_PROPERTIES,
    ]
    subprocess.run(command, check=True)
    return pd.read_csv(output_csv)


def main() -> None:
    args = parse_args()
    if args.iterations < 0:
        raise ValueError("--iterations must be non-negative")
    if args.samples_per_mask < 1:
        raise ValueError("--samples-per-mask must be positive")
    if args.min_mask_tokens < 1:
        raise ValueError("--min-mask-tokens must be positive")
    if args.max_mask_tokens < args.min_mask_tokens:
        raise ValueError("--max-mask-tokens must be >= --min-mask-tokens")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    current_csv = args.output_dir / "round_00_candidates.csv"
    initial = run_remask(None, args.output_dir / "round_00_remask.json", args)
    if not initial:
        raise RuntimeError("Initial remasking produced no valid molecules")
    write_candidates(initial, current_csv, 0)

    for round_index in range(args.iterations + 1):
        pareto = run_aggregate(
            current_csv,
            args.output_dir / f"round_{round_index:02d}_pareto.csv",
            args.output_dir / f"round_{round_index:02d}_pareto.png",
        )
        print(f"Round {round_index}: {len(pareto)} Pareto molecules", flush=True)
        if round_index == args.iterations or pareto.empty:
            break

        next_smiles: list[str] = []
        for index, smiles in enumerate(pareto["smiles"].dropna().astype(str), start=1):
            next_smiles.extend(run_remask(
                smiles,
                args.output_dir / f"round_{round_index + 1:02d}_source_{index:04d}.json",
                args,
            ))
        next_smiles = list(dict.fromkeys(next_smiles))
        if not next_smiles:
            print("No valid remasked molecules remain; stopping.", file=sys.stderr)
            break
        current_csv = args.output_dir / f"round_{round_index + 1:02d}_candidates.csv"
        write_candidates(next_smiles, current_csv, round_index + 1)


if __name__ == "__main__":
    main()
