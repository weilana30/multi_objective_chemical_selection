#!/usr/bin/env python
"""Evaluate two Docker microservices and identify a two-objective Pareto front."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


MOO_SELECTION_DIR = Path(__file__).resolve().parent
INPUT_CSV = MOO_SELECTION_DIR / "curated_200_approved_drug_molecules.csv"
RESULTS_DIR = MOO_SELECTION_DIR / "pareto_results"
ALL_RESULTS_CSV = RESULTS_DIR / "all_predictions.csv"
PARETO_RESULTS_CSV = RESULTS_DIR / "pareto_front.csv"
RUN_SUMMARY_JSON = RESULTS_DIR / "run_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Call two SMILES microservices, collect their requested properties, "
            "and calculate a Pareto front. Port and property order must match."
        )
    )
    parser.add_argument(
        "--ports",
        nargs=2,
        type=int,
        required=True,
        metavar=("PORT_1", "PORT_2"),
        help="Host ports for the two Docker microservices, in order.",
    )
    parser.add_argument(
        "--properties",
        nargs=2,
        required=True,
        metavar=("PROPERTY_1", "PROPERTY_2"),
        help="One requested property per microservice, in matching order.",
    )
    parser.add_argument(
        "--directions",
        nargs=2,
        choices=("minimize", "maximize"),
        default=("minimize", "maximize"),
        metavar=("DIRECTION_1", "DIRECTION_2"),
        help="Objective directions; defaults to: minimize maximize.",
    )
    return parser.parse_args()


def check_service(session: requests.Session, port: int) -> dict:
    url = f"http://localhost:{port}/health"
    response = session.get(url, timeout=10)
    response.raise_for_status()
    return response.json()


def extract_prediction(body: dict, property_name: str) -> float:
    if str(body.get("status", "")).lower() != "success":
        raise RuntimeError(f"Microservice returned an error: {body}")

    result = body.get("results", {}).get(property_name)
    if result is None:
        available = list(body.get("results", {}))
        raise KeyError(
            f"Property {property_name!r} was not returned; available: {available}"
        )

    value = result.get("results")
    if value is None:
        value = result.get("prediction")
    if value is None:
        raise ValueError(f"No numeric value was returned for {property_name!r}")
    return float(value)


def query_property(
    session: requests.Session,
    port: int,
    property_name: str,
    smiles: str,
) -> float:
    response = session.post(
        f"http://localhost:{port}/smi",
        json={"smiles": smiles, "property": [property_name]},
        timeout=(10, 300),
    )

    response.raise_for_status()
    return extract_prediction(response.json(), property_name)


def make_objective_matrix(
    predictions: pd.DataFrame,
    properties: list[str],
    directions: list[str],
) -> np.ndarray:
    columns = []
    for property_name, direction in zip(properties, directions):
        values = predictions[property_name].to_numpy(dtype=float)
        columns.append(-values if direction == "maximize" else values)
    return np.column_stack(columns)


def main() -> None:
    args = parse_args()
    ports = list(args.ports)
    properties = list(args.properties)
    directions = list(args.directions)

    if properties[0] == properties[1]:
        raise ValueError("The two properties must have different names.")
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    compounds = pd.read_csv(INPUT_CSV)
    if "smiles" not in compounds.columns:
        raise ValueError(f"Missing required 'smiles' column in {INPUT_CSV}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    health = {
        str(port): check_service(session, port)
        for port in ports
    }

    rows = []
    for position, (_, compound) in enumerate(compounds.iterrows(), start=1):
        smiles = str(compound["smiles"])
        print(f"{position}/{len(compounds)}: {smiles}", flush=True)
        output = compound.to_dict()
        errors = []

        for port, property_name in zip(ports, properties):
            try:
                output[property_name] = query_property(
                    session, port, property_name, smiles
                )
            except Exception as exc:
                output[property_name] = np.nan
                errors.append(f"port {port}, {property_name}: {exc}")

        output["status"] = "success" if not errors else "error"
        output["error"] = " | ".join(errors) if errors else None
        if errors:
            print(f"  ERROR: {output['error']}", flush=True)
        rows.append(output)

    all_results = pd.DataFrame(rows)
    all_results.to_csv(ALL_RESULTS_CSV, index=False)

    valid = all_results.dropna(subset=properties).copy()
    if valid.empty:
        raise RuntimeError(
            f"No valid prediction pairs were returned. Inspect {ALL_RESULTS_CSV} "
            "and its error column."
        )

    objectives = make_objective_matrix(valid, properties, directions)
    front_indices = NonDominatedSorting().do(
        objectives,
        only_non_dominated_front=True,
    )
    pareto = valid.iloc[front_indices].copy()
    pareto.to_csv(PARETO_RESULTS_CSV, index=False)

    identifying_columns = [
        column
        for column in (
            "candidate_id",
            "chembl_id",
            "preferred_name",
            "smiles",
            *properties,
        )
        if column in pareto.columns
    ]
    pareto_records = json.loads(
        pareto[identifying_columns].to_json(orient="records")
    )
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_csv": str(INPUT_CSV),
        "results_directory": str(RESULTS_DIR),
        "configuration": [
            {
                "port": port,
                "property": property_name,
                "direction": direction,
                "endpoint": f"http://localhost:{port}/smi",
                "health": health[str(port)],
            }
            for port, property_name, direction in zip(
                ports, properties, directions
            )
        ],
        "total_compounds": int(len(all_results)),
        "successful_compounds": int(len(valid)),
        "failed_compounds": int(len(all_results) - len(valid)),
        "pareto_compound_count": int(len(pareto)),
        "pareto_compounds": pareto_records,
    }
    RUN_SUMMARY_JSON.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"All predictions: {ALL_RESULTS_CSV}")
    print(f"Pareto front: {PARETO_RESULTS_CSV}")
    print(f"Run summary: {RUN_SUMMARY_JSON}")


if __name__ == "__main__":
    main()
