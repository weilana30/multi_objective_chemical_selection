#!/usr/bin/env python
"""Aggregate registry-selected property predictions into a Pareto front.

The script reads the two objective definitions from ``property_output_registry.json``,
queries only the selected model properties, sums each objective's member values,
and identifies compounds for which no other compound is both less lipophilic and
more soluble.  Model services are expected to expose a ``POST /smi`` endpoint on
localhost; their ports are supplied on the command line.

The aggregate scores are intentionally raw sums.  The registry documents the
normalization and weighting assumptions that should be applied before combining
heterogeneous properties; this script does not infer or add those transforms.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, TypedDict

import numpy as np
import pandas as pd
import requests

SELECTION_DIR = Path(__file__).resolve().parent
INPUT_CSV = SELECTION_DIR / "curated_200_approved_drug_molecules.csv"
REGISTRY_JSON = SELECTION_DIR / "property_output_registry.json"
OUTPUT_CSV = SELECTION_DIR / "pareto_results" / "aggregate_pareto_front.csv"
OUTPUT_IMAGE = SELECTION_DIR / "pareto_results" / "aggregate_pareto_front.png"

MODEL_PORT_ARGUMENTS = {
    "Admet-AI": "admet_ai_port",
    "Admetica": "admetica_port",
    "Admethyst": "admethyst_port",
    "BoiledEgg": "boiled_egg_port",
    "LogPPrepML": "logp_prep_ml_port",
    "ClassicalGSG": "classical_gsg_port",
    "MolProperty": "mol_property_port",
}


class PredictionFailure(RuntimeError):
    """A prediction service failed for a specific molecule."""


class ObjectiveMember(TypedDict):
    model: str
    property: str
    transform: str
    weight: float


def valid_port(value: str) -> int:
    """Parse a TCP port and reject values outside the valid port range."""
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for model ports and objective members."""
    parser = argparse.ArgumentParser(
        description=(
            "Sum selected lipophilicity and solubility predictions and calculate "
            "a Pareto front that minimizes lipophilicity and maximizes solubility."
        )
    )
    parser.add_argument("--input-csv", type=Path, default=INPUT_CSV)
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--output-image", type=Path, default=OUTPUT_IMAGE)
    for model, destination in MODEL_PORT_ARGUMENTS.items():
        parser.add_argument(
            "--" + destination.replace("_", "-"),
            dest=destination,
            type=valid_port,
            metavar="PORT",
            help=f"Host port for {model}; required when that model is selected.",
        )
    parser.add_argument(
        "--lipophilicity-properties",
        nargs="+",
        metavar="MODEL:PROPERTY",
        help=(
            "Lipophilicity members to include, written as MODEL:PROPERTY. "
            "Defaults to every lipophilicity member in the registry."
        ),
    )
    parser.add_argument(
        "--solubility-properties",
        nargs="+",
        metavar="MODEL:PROPERTY",
        help=(
            "Solubility members to include, written as MODEL:PROPERTY. "
            "Defaults to every solubility member in the registry."
        ),
    )
    return parser


def load_objective_members() -> dict[str, list[ObjectiveMember]]:
    """Load objective members and their transforms and weights from the registry.

    The registry is the source of truth for the default member selection.  A
    ``ValueError`` is raised when its ``moo_objectives`` section is missing or
    has an unexpected shape.
    """
    with REGISTRY_JSON.open(encoding="utf-8") as registry_file:
        registry = json.load(registry_file)

    try:
        objectives = registry["moo_objectives"]
        return {
            objective: [
                {
                    "model": member["model"],
                    "property": member["property"],
                    "transform": member.get("transform", "identity"),
                    "weight": float(member.get("weight", 1.0)),
                }
                for member in objectives[objective]["members"]
            ]
            for objective in ("lipophilicity", "solubility")
        }
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"Invalid moo_objectives structure in {REGISTRY_JSON}: {exc}"
        ) from exc


def member_name(member: ObjectiveMember) -> str:
    """Return the stable dataframe column name for a registry member."""
    return f"{member['model']}:{member['property']}"


def resolve_members(
    parser: argparse.ArgumentParser,
    objective: str,
    requested: list[str] | None,
    available: list[ObjectiveMember],
) -> list[ObjectiveMember]:
    """Resolve requested member names against one objective's registry entries.

    With no explicit request, all available members are returned.  Explicit
    names must be unique and use the ``MODEL:PROPERTY`` spelling shown by the
    command-line help.
    """
    if requested is None:
        return available

    available_by_name = {member_name(member): member for member in available}
    unknown = [name for name in requested if name not in available_by_name]
    if unknown:
        choices = ", ".join(available_by_name)
        parser.error(
            f"invalid {objective} member(s): {', '.join(unknown)}; "
            f"valid choices are: {choices}"
        )

    if len(requested) != len(set(requested)):
        parser.error(f"duplicate {objective} members are not allowed")
    return [available_by_name[name] for name in requested]


def selected_properties_by_model(
    objectives: dict[str, list[ObjectiveMember]],
) -> dict[str, list[str]]:
    """Group selected properties by model while preserving selection order.

    A property selected by both objectives is queried once and reused in both
    aggregate scores.
    """
    grouped: dict[str, list[str]] = defaultdict(list)
    for members in objectives.values():
        for member in members:
            model = member["model"]
            property_name = member["property"]
            if property_name not in grouped[model]:
                grouped[model].append(property_name)
    return dict(grouped)


def require_selected_ports(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    properties_by_model: dict[str, list[str]],
) -> dict[str, int]:
    """Validate and return ports for exactly the models used by the objectives."""
    unknown_models = sorted(set(properties_by_model) - set(MODEL_PORT_ARGUMENTS))
    if unknown_models:
        parser.error(
            "no port argument is configured for registry model(s): "
            + ", ".join(unknown_models)
        )

    ports: dict[str, int] = {}
    for model in properties_by_model:
        destination = MODEL_PORT_ARGUMENTS[model]
        port = getattr(args, destination)
        if port is None:
            option = "--" + destination.replace("_", "-")
            parser.error(f"{option} is required for selected model {model}")
        ports[model] = port
    return ports


def extract_predictions(
    body: dict[str, Any],
    property_names: list[str],
) -> dict[str, float]:
    """Extract finite numeric predictions from a model service response.

    Services may return a value under either ``results`` or ``prediction``.
    Missing, failed, non-numeric, and non-finite property values are rejected
    so an invalid response cannot silently affect the Pareto calculation.
    """
    if str(body.get("status", "")).lower() != "success":
        raise ValueError(body.get("error") or body)

    results = body.get("results")
    if not isinstance(results, dict):
        raise ValueError("response does not contain a results mapping")

    predictions = {}
    for property_name in property_names:
        result = results.get(property_name)
        if not isinstance(result, dict):
            raise ValueError(f"{property_name}: property was not returned")
        if str(result.get("status", "")).lower() != "success":
            raise ValueError(
                f"{property_name}: {result.get('error') or 'prediction failed'}"
            )

        value = result.get("results")
        if value is None:
            value = result.get("prediction")
        if value is None:
            raise ValueError(f"{property_name}: no numeric value was returned")

        try:
            prediction = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{property_name}: returned non-numeric value {value!r}"
            ) from exc
        if not math.isfinite(prediction):
            raise ValueError(f"{property_name}: returned non-finite value {value!r}")
        predictions[property_name] = prediction

    return predictions


def query_model(
    session: requests.Session,
    model: str,
    port: int,
    property_names: list[str],
    smiles: str,
    molecule: str,
) -> dict[str, float]:
    """Request several properties for one molecule from a model service.

    Network, HTTP, JSON, and response-validation failures are wrapped in
    ``PredictionFailure`` with the model, molecule, and requested properties
    included for diagnosis.
    """
    url = f"http://localhost:{port}/smi"
    try:
        response = session.post(
            url,
            json={"smiles": smiles, "property": property_names},
            timeout=(10, 300),
        )
        if not response.ok:
            detail = response.text.strip() or response.reason
            raise ValueError(f"HTTP {response.status_code}: {detail}")
        body = response.json()
        if not isinstance(body, dict):
            raise ValueError("response body is not a JSON object")
        return extract_predictions(body, property_names)
    except (requests.RequestException, ValueError) as exc:
        properties = ", ".join(property_names)
        raise PredictionFailure(
            f"{model} failed for {molecule} (SMILES: {smiles}); "
            f"requested properties: {properties}; {exc}"
        ) from exc


def molecule_name(compound: pd.Series, position: int) -> str:
    """Build a readable molecule identifier for progress and error messages."""
    identifiers = [
        str(compound[column])
        for column in ("candidate_id", "chembl_id", "preferred_name")
        if column in compound and pd.notna(compound[column])
    ]
    return " / ".join(identifiers) if identifiers else f"row {position}"


def calculate_predictions(
    compounds: pd.DataFrame,
    objectives: dict[str, list[ObjectiveMember]],
    ports: dict[str, int],
) -> pd.DataFrame:
    """Query selected models and append member and aggregate score columns.

    The input compound columns are preserved.  Each member prediction is stored
    under ``MODEL:PROPERTY``; each objective score is the sum of its selected
    member columns. Compounds with a failed property request are logged and
    skipped because a partial score is not comparable with other compounds.
    """
    properties_by_model = selected_properties_by_model(objectives)
    rows = []
    with requests.Session() as session:
        for position, (_, compound) in enumerate(compounds.iterrows(), start=1):
            smiles = str(compound["smiles"])
            molecule = molecule_name(compound, position)
            print(f"{position}/{len(compounds)}: {molecule}", flush=True)

            output = compound.to_dict()
            failed = False
            for model, property_names in properties_by_model.items():
                try:
                    predictions = query_model(
                        session,
                        model,
                        ports[model],
                        property_names,
                        smiles,
                        molecule,
                    )
                except PredictionFailure as exc:
                    print(f"WARNING: skipping {exc}", file=sys.stderr, flush=True)
                    failed = True
                    break
                for property_name, value in predictions.items():
                    output[f"{model}:{property_name}"] = value

            if failed:
                continue

            for objective, members in objectives.items():
                weighted_values = []
                for member in members:
                    value = float(output[member_name(member)])
                    if member["transform"] == "negate":
                        value *= -1
                    elif member["transform"] != "identity":
                        raise ValueError(
                            f"Unsupported transform {member['transform']!r} "
                            f"for {member_name(member)}"
                        )
                    weighted_values.append(member["weight"] * value)

                output[f"{objective}_score"] = sum(weighted_values)
            rows.append(output)
    return pd.DataFrame(rows)


def find_pareto_front(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return compounds not dominated under the two aggregate objectives.

    Lipophilicity and solubility are maximized.  A compound is
    dominated when another compound is no worse on both scores and strictly
    better on at least one.  Equal-score compounds therefore remain on the
    front.
    """
    lipophilicity = predictions["lipophilicity_score"].to_numpy(dtype=float)
    solubility = predictions["solubility_score"].to_numpy(dtype=float)
    is_pareto = np.ones(len(predictions), dtype=bool)

    for index in range(len(predictions)):
        at_least_as_good = (lipophilicity >= lipophilicity[index]) & (
            solubility >= solubility[index]
        )
        strictly_better = (lipophilicity > lipophilicity[index]) | (
            solubility > solubility[index]
        )
        if np.any(at_least_as_good & strictly_better):
            is_pareto[index] = False

    return predictions.loc[is_pareto].copy()


def plot_pareto_front(
    predictions: pd.DataFrame, pareto: pd.DataFrame, output_image: Path
) -> None:
    """Save a scatter plot of all compounds and the selected Pareto front."""
    with TemporaryDirectory(prefix="aggregate-pareto-matplotlib-") as config_dir:
        previous_config_dir = os.environ.get("MPLCONFIGDIR")
        os.environ["MPLCONFIGDIR"] = config_dir
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            figure, axis = plt.subplots(figsize=(10, 7))
            axis.scatter(
                predictions["lipophilicity_score"],
                predictions["solubility_score"],
                color="0.65",
                alpha=0.7,
                label="All molecules",
            )

            ordered_front = pareto.sort_values("lipophilicity_score")
            axis.plot(
                ordered_front["lipophilicity_score"],
                ordered_front["solubility_score"],
                color="tab:red",
                linestyle="--",
                linewidth=1,
            )
            axis.scatter(
                pareto["lipophilicity_score"],
                pareto["solubility_score"],
                color="tab:red",
                edgecolor="black",
                label="Pareto front",
                zorder=3,
            )

            label_column = next(
                (
                    column
                    for column in ("preferred_name", "candidate_id", "chembl_id")
                    if column in pareto.columns
                ),
                None,
            )
            if label_column:
                for _, compound in pareto.iterrows():
                    axis.annotate(
                        str(compound[label_column]),
                        (
                            compound["lipophilicity_score"],
                            compound["solubility_score"],
                        ),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=8,
                    )

            axis.set_xlabel("Lipophilicity score (higher is better)")
            axis.set_ylabel("Solubility score (higher is better)")
            axis.set_title("Aggregate Lipophilicity–Solubility Pareto Front")
            axis.grid(alpha=0.2)
            axis.legend()
            figure.tight_layout()
            figure.savefig(output_image, dpi=200)
            plt.close(figure)
        finally:
            if previous_config_dir is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = previous_config_dir


def main() -> None:
    """Run the registry-driven prediction, Pareto selection, and export steps."""
    parser = build_parser()
    args = parser.parse_args()
    available = load_objective_members()

    objectives = {
        "lipophilicity": resolve_members(
            parser,
            "lipophilicity",
            args.lipophilicity_properties,
            available["lipophilicity"],
        ),
        "solubility": resolve_members(
            parser,
            "solubility",
            args.solubility_properties,
            available["solubility"],
        ),
    }
    properties_by_model = selected_properties_by_model(objectives)
    ports = require_selected_ports(parser, args, properties_by_model)

    input_csv = args.input_csv.expanduser().resolve()
    output_csv = args.output_csv.expanduser().resolve()
    output_image = args.output_image.expanduser().resolve()
    compounds = pd.read_csv(input_csv)
    if "smiles" not in compounds.columns:
        raise ValueError(f"Missing required 'smiles' column in {input_csv}")
    if compounds.empty:
        raise ValueError(f"Input CSV contains no molecules: {input_csv}")

    predictions = calculate_predictions(compounds, objectives, ports)
    if predictions.empty:
        raise ValueError("No compounds produced complete predictions")
    pareto = find_pareto_front(predictions)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_image.parent.mkdir(parents=True, exist_ok=True)
    pareto.to_csv(output_csv, index=False)
    plot_pareto_front(predictions, pareto, output_image)

    print(f"Pareto compounds: {len(pareto)}")
    print(f"Pareto CSV: {output_csv}")
    print(f"Pareto image: {output_image}")


if __name__ == "__main__":
    try:
        main()
    except PredictionFailure as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
