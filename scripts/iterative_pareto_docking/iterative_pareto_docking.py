#!/usr/bin/env python3
"""Iteratively optimize lipophilicity, solubility, and docking score.

Each round generates candidates with GenMol, scores them with the available
property services and AutoDock Vina, writes the complete score table, and
remasks the Pareto-front molecules for the next round.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aggregate_pareto import (
    ObjectiveMember,
    PredictionFailure,
    extract_predictions,
    load_objective_members,
    member_name,
)


PROJECT_DIR = Path(__file__).resolve().parents[2]
REMASK_SCRIPT = PROJECT_DIR.parent / "genmol_molecule_generator" / "remask_generated_molecule.py"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "pareto_results" / "caffeine_docking"

CURRENT_LIPOPHILICITY_PROPERTIES = [
    "Admet-AI:logP",
    "Admet-AI:Lipophilicity_AstraZeneca",
    "Admetica:lipophilicity",
    "Admethyst:Lipophilicity_AstraZeneca",
    "BoiledEgg:WLogP",
    "ClassicalGSG:LogP_MMFF",
    "MolProperty:logP",
]
CURRENT_SOLUBILITY_PROPERTIES = [
    "Admet-AI:Solubility_AqSolDB",
    "Admet-AI:HydrationFreeEnergy_FreeSolv",
    "Admetica:solubility",
    "Admethyst:Solubility_AqSolDB",
    "Admethyst:HydrationFreeEnergy_FreeSolv",
    "MolProperty:logS",
]

SERVICE_DEFAULT_PORTS = {
    "Admet-AI": 18019,
    "Admetica": 18000,
    "Admethyst": 18001,
    "BoiledEgg": 18002,
    "ClassicalGSG": 10001,
    "MolProperty": 18003,
}
OBJECTIVE_COLUMNS = [
    "lipophilicity_score",
    "solubility_score",
    "docking_score",
]
CAFFEINE_SMILES = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
CAFFEINE_SMILES_ALIASES = {
    CAFFEINE_SMILES.lower(),
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C".lower(),
}
DEFAULT_SAMPLES_SCHEDULE = (10, 5, 2, 1)
DEFAULT_MIN_SOLUBILITY_LOGS = -4.0
DEFAULT_MIN_DOCKING_SCORE = 5.0
SCREENING_CONSTRAINTS = [
    {
        "name": "molecular_weight_oral",
        "model": "Admet-AI",
        "property": "molecular_weight",
        "operator": "<=",
        "value": 500.0,
        "unit": "Da",
        "source": "property_output_registry.json:molecular_weight_oral",
    },
    {
        "name": "hba_oral",
        "model": "Admet-AI",
        "property": "hydrogen_bond_acceptors",
        "operator": "<=",
        "value": 10.0,
        "unit": "count",
        "source": "property_output_registry.json:hba_oral",
    },
    {
        "name": "hbd_oral",
        "model": "Admet-AI",
        "property": "hydrogen_bond_donors",
        "operator": "<=",
        "value": 5.0,
        "unit": "count",
        "source": "property_output_registry.json:hbd_oral",
    },
    {
        "name": "tpsa_oral",
        "model": "Admet-AI",
        "property": "tpsa",
        "operator": "<=",
        "value": 140.0,
        "unit": "angstrom^2",
        "source": "property_output_registry.json:tpsa_oral",
    },
    {
        "name": "lipinski_rule_of_five",
        "model": "Admet-AI",
        "property": "Lipinski",
        "operator": "==",
        "value": 4.0,
        "unit": "rules_satisfied_out_of_4",
        "source": "property_output_registry.json:Admet-AI.Lipinski",
    },
    {
        "name": "qed_druglike",
        "model": "Admet-AI",
        "property": "QED",
        "operator": ">=",
        "value": 0.5,
        "unit": "unitless_0_to_1",
        "source": "property_output_registry.json:qed_druglike",
    },
    {
        "name": "rotatable_bonds_oral",
        "model": "MolProperty",
        "property": "RotatableBonds",
        "operator": "<=",
        "value": 10.0,
        "unit": "count",
        "source": "property_output_registry.json:rotatable_bonds_oral",
    },
]
GENMOL_PROGRESS_PATTERN = re.compile(
    r"fragment (?P<fragment>\d+)/(?P<fragments>\d+), mask length (?P<mask>\d+)"
)


def valid_port(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def parse_samples_schedule(value: str) -> list[int]:
    try:
        schedule = [int(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "sample schedule must be comma-separated positive integers"
        ) from exc
    if not schedule or any(samples < 1 for samples in schedule):
        raise argparse.ArgumentTypeError(
            "sample schedule must contain positive integers"
        )
    return schedule


def show_progress(label: str, fraction: float, detail: str = "") -> None:
    fraction = min(max(fraction, 0.0), 1.0)
    width = 28
    filled = int(round(width * fraction))
    bar = "=" * filled + "-" * (width - filled)
    line = f"{label} [{bar}] {fraction * 100:6.1f}%"
    if detail:
        line += f" | {detail}"
    sys.stdout.write(f"\r\033[K{line}")
    sys.stdout.flush()


def finish_progress() -> None:
    sys.stdout.write("\n")
    sys.stdout.flush()


def build_screening_constraints(args: argparse.Namespace) -> list[dict[str, Any]]:
    constraints = [dict(constraint) for constraint in SCREENING_CONSTRAINTS]
    constraints.extend(
        [
            {
                "name": "admet_ai_solubility_screen",
                "model": "Admet-AI",
                "property": "Solubility_AqSolDB",
                "operator": ">=",
                "value": args.min_solubility_log_s,
                "unit": "log10_mol_per_liter",
                "source": "project screening threshold; registry identifies this as a direct solubility objective",
            },
            {
                "name": "molproperty_solubility_screen",
                "model": "MolProperty",
                "property": "logS",
                "operator": ">=",
                "value": args.min_solubility_log_s,
                "unit": "presumed_log10_mol_per_liter",
                "source": "project screening threshold; registry identifies this as a direct solubility objective",
            },
        ]
    )
    return constraints


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--starting-smiles",
        help="Seed molecule for the first GenMol round; omit for de novo generation.",
    )
    parser.add_argument(
        "--iterations",
        "-n",
        type=int,
        default=None,
        help="Number of transitions; defaults to the full sample schedule.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Parent directory for timestamped run directories.",
    )
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--samples-schedule",
        type=parse_samples_schedule,
        default=list(DEFAULT_SAMPLES_SCHEDULE),
        help="GenMol samples per fragment for successive rounds (default: 10,5,2,1).",
    )
    parser.add_argument("--min-mask-tokens", type=int, default=5)
    parser.add_argument("--max-mask-tokens", type=int, default=5)
    parser.add_argument(
        "--vina-url",
        default="http://localhost:18025",
        help="Base URL for the AutoDock Vina service.",
    )
    parser.add_argument("--receptor", default="A2A_3RFM")
    parser.add_argument("--vina-exhaustiveness", type=int, default=8)
    parser.add_argument("--vina-poses", type=int, default=5)
    parser.add_argument(
        "--min-solubility-log-s",
        type=float,
        default=DEFAULT_MIN_SOLUBILITY_LOGS,
        help="Minimum ADMET-AI and MolProperty logS screening value.",
    )
    parser.add_argument(
        "--min-docking-score",
        type=float,
        default=DEFAULT_MIN_DOCKING_SCORE,
        help="Minimum Vina docking score retained after docking.",
    )
    parser.add_argument(
        "--no-screening-constraints",
        action="store_true",
        help="Disable registry-based property gates and solubility/docking screening.",
    )
    parser.add_argument(
        "--service-probe-smiles",
        default=CAFFEINE_SMILES,
        help="SMILES used once at startup to remove unhealthy or unsupported property members.",
    )
    parser.add_argument(
        "--require-all-services",
        action="store_true",
        help="Fail instead of skipping a requested property service/member that does not work.",
    )
    for model, port in SERVICE_DEFAULT_PORTS.items():
        destination = model.lower().replace("-", "_") + "_port"
        parser.add_argument(
            "--" + destination.replace("_", "-"),
            dest=destination,
            type=valid_port,
            default=port,
            metavar="PORT",
            help=f"Host port for {model} (default: {port}).",
        )
    return parser


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def create_run_output_dir(args: argparse.Namespace) -> Path:
    starting_smiles = (args.starting_smiles or "").replace(" ", "").lower()
    label = "caffeine" if starting_smiles in CAFFEINE_SMILES_ALIASES else "de_novo"
    mask_range = f"{args.min_mask_tokens}_to_{args.max_mask_tokens}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{label}_{mask_range}_{timestamp}"
    run_dir = args.output_dir / stem
    suffix = 2
    while run_dir.exists():
        run_dir = args.output_dir / f"{stem}_run_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True)
    return run_dir


def run_remask(
    smiles: str | None,
    output_json: Path,
    args: argparse.Namespace,
    samples_per_mask: int,
    round_index: int,
    total_rounds: int,
    source_index: int,
    source_total: int,
) -> list[str]:
    project_root = REMASK_SCRIPT.parent
    bundled_python = project_root / ".conda-env" / "bin" / "python"
    interpreter = str(bundled_python) if bundled_python.is_file() else sys.executable
    command = [interpreter, str(REMASK_SCRIPT), "--output", str(output_json)]
    if args.model_path:
        command += ["--model-path", str(args.model_path)]
    command += ["--device", args.device, "--seed", str(args.seed)]
    command += [
        "--samples-per-mask",
        str(samples_per_mask),
        "--min-mask-tokens",
        str(args.min_mask_tokens),
        "--max-mask-tokens",
        str(args.max_mask_tokens),
    ]
    if smiles is not None:
        command += ["--smiles", smiles]

    environment = None
    if bundled_python.is_file():
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(project_root / "genmol" / "src")

    process = subprocess.Popen(
        command,
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        bufsize=1,
        env=environment,
    )
    stderr_lines: list[str] = []
    condition_count = args.max_mask_tokens - args.min_mask_tokens + 1
    label = f"Round {round_index + 1}/{total_rounds} | generating batch"
    while process.stderr is not None:
        line = process.stderr.readline()
        if not line:
            if process.poll() is not None:
                break
            continue
        stderr_lines.append(line)
        match = GENMOL_PROGRESS_PATTERN.search(line)
        if not match:
            continue
        fragment = int(match.group("fragment"))
        fragments = int(match.group("fragments"))
        mask_length = int(match.group("mask"))
        total_conditions = fragments * condition_count
        completed_conditions = (fragment - 1) * condition_count + max(
            0, mask_length - args.min_mask_tokens
        )
        source_fraction = completed_conditions / total_conditions
        show_progress(
            label,
            ((source_index - 1) + source_fraction) / source_total,
            f"source {source_index}/{source_total}, "
            f"fragment {fragment}/{fragments}, mask {mask_length}",
        )

    return_code = process.wait()
    if return_code != 0:
        finish_progress()
        details = "".join(stderr_lines[-8:]).strip()
        raise RuntimeError(f"GenMol failed with exit code {return_code}: {details}")
    show_progress(
        label,
        source_index / source_total,
        f"source {source_index}/{source_total} complete",
    )
    finish_progress()

    try:
        payload = json.loads(output_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"GenMol output is not valid JSON: {exc}") from exc

    return list(dict.fromkeys(payload.get("all_unique_generated_variants", [])))


def write_candidates(smiles: list[str], path: Path, round_index: int) -> None:
    pd.DataFrame(
        {
            "candidate_id": [f"R{round_index:02d}_{i:05d}" for i in range(len(smiles))],
            "preferred_name": [f"generated_{round_index}_{i}" for i in range(len(smiles))],
            "smiles": smiles,
        }
    ).to_csv(path, index=False)


def request_property(
    session: requests.Session,
    model: str,
    port: int,
    property_names: list[str],
    smiles: str,
    molecule: str,
) -> dict[str, float]:
    try:
        response = session.post(
            f"http://localhost:{port}/smi",
            json={"smiles": smiles, "property": property_names},
            timeout=(10, 300),
        )
        if not response.ok:
            raise ValueError(f"HTTP {response.status_code}: {response.text.strip()}")
        body = response.json()
        if not isinstance(body, dict):
            raise ValueError("response body is not a JSON object")
        return extract_predictions(body, property_names)
    except (requests.RequestException, ValueError) as exc:
        properties = ", ".join(property_names)
        raise PredictionFailure(
            f"{model} failed for {molecule}; requested properties: {properties}; {exc}"
        ) from exc


def check_property_services(
    args: argparse.Namespace,
    requested: dict[str, list[ObjectiveMember]],
    constraints: list[dict[str, Any]],
) -> tuple[
    dict[str, list[ObjectiveMember]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    active: dict[str, list[ObjectiveMember]] = {"lipophilicity": [], "solubility": []}
    active_constraints: list[dict[str, Any]] = []
    requested_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for members in requested.values():
        for member in members:
            requested_by_model[member["model"]].append(
                {"kind": "objective", "member": member}
            )
    for constraint in constraints:
        requested_by_model[constraint["model"]].append(
            {"kind": "constraint", "constraint": constraint}
        )

    statuses: list[dict[str, Any]] = []
    with requests.Session() as session:
        model_total = len(requested_by_model)
        for model_index, (model, members) in enumerate(
            requested_by_model.items(), start=1
        ):
            show_progress(
                "Preflight | property services",
                (model_index - 1) / model_total,
                model,
            )
            port = getattr(args, model.lower().replace("-", "_") + "_port")
            status: dict[str, Any] = {
                "model": model,
                "port": port,
                "health": False,
                "active_members": [],
                "disabled_members": [],
            }
            try:
                health = session.get(
                    f"http://localhost:{port}/health", timeout=(5, 15)
                )
                body = health.json()
                if not isinstance(body, dict):
                    raise ValueError("health response is not a JSON object")
                if not health.ok or str(body.get("status", "")).lower() != "healthy":
                    raise ValueError(f"health response: {body}")
                status["health"] = True
            except (requests.RequestException, ValueError) as exc:
                reason = f"health check failed: {exc}"
                for check in members:
                    name = (
                        member_name(check["member"])
                        if check["kind"] == "objective"
                        else f"{check['constraint']['model']}:{check['constraint']['property']}"
                    )
                    status["disabled_members"].append(
                        {"member": name, "reason": reason}
                    )
                statuses.append(status)
                show_progress(
                    "Preflight | property services",
                    model_index / model_total,
                    f"{model} unavailable",
                )
                continue

            for check in members:
                member = check.get("member")
                constraint = check.get("constraint")
                name = (
                    member_name(member)
                    if check["kind"] == "objective"
                    else f"{constraint['model']}:{constraint['property']}"
                )
                try:
                    request_property(
                        session,
                        model,
                        port,
                        [member["property"] if member else constraint["property"]],
                        args.service_probe_smiles,
                        "service probe",
                    )
                except PredictionFailure as exc:
                    status["disabled_members"].append(
                        {"member": name, "reason": str(exc)}
                    )
                    continue
                status["active_members"].append(name)
                if check["kind"] == "objective":
                    for objective, objective_members in requested.items():
                        if member in objective_members:
                            active[objective].append(member)
                else:
                    active_constraints.append(constraint)
            statuses.append(status)
            show_progress(
                "Preflight | property services",
                model_index / model_total,
                f"{model} checked",
            )
        finish_progress()

    disabled = [item for status in statuses for item in status["disabled_members"]]
    for item in disabled:
        print(f"Skipping {item['member']}: {item['reason']}", file=sys.stderr, flush=True)
    if args.require_all_services and disabled:
        raise RuntimeError("Requested property services failed; see service_status.json")
    if not active["lipophilicity"] or not active["solubility"]:
        raise RuntimeError("No working members remain for both objectives")
    return active, active_constraints, statuses


def check_vina(args: argparse.Namespace) -> dict[str, Any]:
    base_url = args.vina_url.rstrip("/")
    try:
        health_response = requests.get(f"{base_url}/health", timeout=(5, 30))
        health = health_response.json()
        if not isinstance(health, dict):
            raise ValueError("health response is not a JSON object")
        if not health_response.ok or str(health.get("status", "")).lower() != "healthy":
            raise ValueError(f"health response: {health}")
        receptors_response = requests.get(f"{base_url}/receptors", timeout=(5, 30))
        receptor_body = receptors_response.json()
        if not receptors_response.ok or not isinstance(receptor_body, dict):
            raise ValueError(f"receptor response: {receptor_body}")
        receptors = receptor_body.get("receptors", [])
        if not isinstance(receptors, list) or not all(isinstance(item, dict) for item in receptors):
            raise ValueError(f"invalid receptor list: {receptors}")
        names = [item.get("name") for item in receptors]
        if args.receptor not in names:
            raise ValueError(f"receptor {args.receptor!r} is not available; found {names}")
    except (requests.RequestException, ValueError) as exc:
        raise RuntimeError(f"AutoDock Vina preflight failed: {exc}") from exc
    return {
        "url": base_url,
        "receptor": args.receptor,
        "exhaustiveness": args.vina_exhaustiveness,
        "poses": args.vina_poses,
        "health": health,
    }


def request_docking(
    session: requests.Session,
    args: argparse.Namespace,
    smiles: str,
    molecule: str,
) -> dict[str, float]:
    try:
        response = session.post(
            f"{args.vina_url.rstrip('/')}/smi",
            json={
                "smiles": smiles,
                "receptor": args.receptor,
                "exhaustiveness": args.vina_exhaustiveness,
                "poses": args.vina_poses,
                "seed": args.seed,
            },
            timeout=(10, 600),
        )
        if not response.ok:
            raise ValueError(f"HTTP {response.status_code}: {response.text.strip()}")
        body = response.json()
        if not isinstance(body, dict):
            raise ValueError("response body is not a JSON object")
        if str(body.get("status", "")).lower() != "success":
            raise ValueError(body)
        affinity = float(body["docking_affinity"])
        score = float(body.get("docking_score", -affinity))
        if not math.isfinite(affinity) or not math.isfinite(score):
            raise ValueError("docking result was not finite")
        return {"docking_affinity": affinity, "docking_score": score}
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        raise PredictionFailure(
            f"AutoDock Vina failed for {molecule} (SMILES: {smiles}): {exc}"
        ) from exc


def selected_properties_by_model(
    objectives: dict[str, list[ObjectiveMember]],
    constraints: list[dict[str, Any]],
) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for members in objectives.values():
        for member in members:
            if member["property"] not in grouped[member["model"]]:
                grouped[member["model"]].append(member["property"])
    for constraint in constraints:
        if constraint["property"] not in grouped[constraint["model"]]:
            grouped[constraint["model"]].append(constraint["property"])
    return dict(grouped)


def evaluate_constraints(
    output: dict[str, Any], constraints: list[dict[str, Any]]
) -> list[str]:
    failures = []
    for constraint in constraints:
        key = f"{constraint['model']}:{constraint['property']}"
        try:
            actual = float(output[key])
        except (KeyError, TypeError, ValueError) as exc:
            failures.append(f"{constraint['name']} (missing/non-numeric: {exc})")
            continue
        if not math.isfinite(actual):
            failures.append(f"{constraint['name']} (non-finite: {actual})")
            continue

        expected = float(constraint["value"])
        operator = constraint["operator"]
        passed = {
            "<=": actual <= expected,
            ">=": actual >= expected,
            "==": math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-6),
        }.get(operator)
        if passed is None:
            raise ValueError(f"Unsupported screening operator: {operator}")
        if not passed:
            failures.append(
                f"{constraint['name']} ({actual:.4g} {operator} {expected:.4g} failed)"
            )
    return failures


def molecule_name(compound: pd.Series, position: int) -> str:
    identifiers = [
        str(compound[column])
        for column in ("candidate_id", "chembl_id", "preferred_name")
        if column in compound and pd.notna(compound[column])
    ]
    return " / ".join(identifiers) if identifiers else f"row {position}"


def calculate_scores(
    compounds: pd.DataFrame,
    objectives: dict[str, list[ObjectiveMember]],
    constraints: list[dict[str, Any]],
    args: argparse.Namespace,
    round_index: int,
    total_rounds: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    properties_by_model = selected_properties_by_model(objectives, constraints)
    screened_rows: list[dict[str, Any]] = []
    scored_rows: list[dict[str, Any]] = []
    with requests.Session() as session:
        for position, (_, compound) in enumerate(compounds.iterrows(), start=1):
            smiles = str(compound["smiles"])
            molecule = molecule_name(compound, position)
            show_progress(
                f"Round {round_index + 1}/{total_rounds} | scoring molecules",
                (position - 1) / len(compounds),
                f"molecule {position}/{len(compounds)}",
            )
            output = compound.to_dict()
            try:
                for model, property_names in properties_by_model.items():
                    predictions = request_property(
                        session,
                        model,
                        getattr(args, model.lower().replace("-", "_") + "_port"),
                        property_names,
                        smiles,
                        molecule,
                    )
                    for property_name, value in predictions.items():
                        output[f"{model}:{property_name}"] = value
            except PredictionFailure as exc:
                print(f"WARNING: skipping {exc}", file=sys.stderr, flush=True)
                continue

            failures = evaluate_constraints(output, constraints)
            if failures:
                output["docked"] = False
                output["screening_pass"] = False
                output["screening_failures"] = "; ".join(failures)
                screened_rows.append(output)
                continue

            try:
                output.update(request_docking(session, args, smiles, molecule))
            except PredictionFailure as exc:
                print(f"WARNING: skipping {exc}", file=sys.stderr, flush=True)
                continue

            if not args.no_screening_constraints and output["docking_score"] < args.min_docking_score:
                failures.append(
                    f"docking_score ({output['docking_score']:.4g} < {args.min_docking_score:.4g})"
                )
            output["docked"] = True
            output["screening_pass"] = not failures
            output["screening_failures"] = "; ".join(failures)
            screened_rows.append(output)
            if failures:
                continue

            for objective, members in objectives.items():
                total = 0.0
                for member in members:
                    value = float(output[member_name(member)])
                    transform = member.get("transform", "identity")
                    if transform == "negate":
                        value *= -1
                    elif transform != "identity":
                        raise ValueError(
                            f"Unsupported transform {transform!r} for {member_name(member)}"
                        )
                    total += float(member.get("weight", 1.0)) * value
                output[f"{objective}_score"] = total
            scored_rows.append(output)
    show_progress(
        f"Round {round_index + 1}/{total_rounds} | scoring molecules",
        1.0,
        f"{len(compounds)}/{len(compounds)} complete",
    )
    finish_progress()
    return pd.DataFrame(screened_rows), pd.DataFrame(scored_rows)


def find_pareto_front(predictions: pd.DataFrame) -> pd.DataFrame:
    values = predictions[OBJECTIVE_COLUMNS].to_numpy(dtype=float)
    is_pareto = np.ones(len(predictions), dtype=bool)
    for index, current in enumerate(values):
        no_worse = np.all(values >= current, axis=1)
        strictly_better = np.any(values > current, axis=1)
        if np.any(no_worse & strictly_better):
            is_pareto[index] = False
    return predictions.loc[is_pareto].copy()


def plot_pareto_front(
    predictions: pd.DataFrame,
    pareto: pd.DataFrame,
    pairwise_path: Path,
) -> None:
    with TemporaryDirectory(prefix="iterative-docking-matplotlib-") as config_dir:
        previous_config_dir = os.environ.get("MPLCONFIGDIR")
        os.environ["MPLCONFIGDIR"] = config_dir
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            pairs = [
                (
                    "lipophilicity_score",
                    "solubility_score",
                    "Lipophilicity vs. Solubility",
                ),
                (
                    "lipophilicity_score",
                    "docking_score",
                    "Lipophilicity vs. Docking",
                ),
                ("solubility_score", "docking_score", "Solubility vs. Docking"),
            ]
            figure, axes = plt.subplots(
                1, 3, figsize=(20, 7), constrained_layout=True
            )
            for axis, (x_column, y_column, title) in zip(axes, pairs):
                axis.scatter(
                    predictions[x_column],
                    predictions[y_column],
                    color="0.65",
                    alpha=0.65,
                    s=42,
                    label="All molecules",
                )
                ordered_front = pareto.sort_values(x_column)
                axis.plot(
                    ordered_front[x_column],
                    ordered_front[y_column],
                    color="tab:red",
                    linewidth=2.5,
                    marker="o",
                    markersize=7,
                    markeredgecolor="black",
                    label="Pareto front",
                )
                axis.scatter(
                    pareto[x_column],
                    pareto[y_column],
                    color="tab:red",
                    edgecolor="black",
                    s=55,
                    zorder=3,
                )
                axis.set_title(title, fontsize=14, pad=12)
                axis.set_xlabel(x_column.replace("_score", "").title(), fontsize=12)
                axis.set_ylabel(y_column.replace("_score", "").title(), fontsize=12)
                axis.tick_params(axis="both", labelsize=10)
                axis.grid(alpha=0.25)
                axis.legend(fontsize=10)
            figure.suptitle(
                "Pairwise projections of the three-objective Pareto front",
                fontsize=17,
            )
            figure.savefig(pairwise_path, dpi=250, bbox_inches="tight")
            plt.close(figure)
        finally:
            if previous_config_dir is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = previous_config_dir


def resolve_requested_members() -> dict[str, list[ObjectiveMember]]:
    available = load_objective_members()
    available_by_name = {
        objective: {member_name(member): member for member in members}
        for objective, members in available.items()
    }
    selected_names = {
        "lipophilicity": CURRENT_LIPOPHILICITY_PROPERTIES,
        "solubility": CURRENT_SOLUBILITY_PROPERTIES,
    }
    requested: dict[str, list[ObjectiveMember]] = {}
    for objective, names in selected_names.items():
        missing = [name for name in names if name not in available_by_name[objective]]
        if missing:
            raise RuntimeError(
                f"The registry is missing {objective} member(s): {', '.join(missing)}"
            )
        requested[objective] = [available_by_name[objective][name] for name in names]
    return requested


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.iterations is None:
        args.iterations = len(args.samples_schedule) - 1
    if args.iterations < 0:
        parser.error("--iterations must be non-negative")
    if args.iterations >= len(args.samples_schedule):
        parser.error("--iterations cannot exceed the sample schedule length minus one")
    if args.min_mask_tokens < 1 or args.max_mask_tokens < args.min_mask_tokens:
        parser.error("mask token bounds are invalid")
    if args.vina_exhaustiveness < 1 or args.vina_poses < 1:
        parser.error("Vina exhaustiveness and poses must be positive")
    if args.min_solubility_log_s > 0:
        parser.error("--min-solubility-log-s should normally be <= 0")
    if args.starting_smiles is not None and not args.starting_smiles.strip():
        parser.error("--starting-smiles cannot be empty")

    args.output_dir = create_run_output_dir(args)
    requested = resolve_requested_members()
    requested_constraints = (
        [] if args.no_screening_constraints else build_screening_constraints(args)
    )
    objectives, constraints, service_status = check_property_services(
        args, requested, requested_constraints
    )
    vina_status = check_vina(args)
    write_json(
        args.output_dir / "service_status.json",
        {
            "requested_objectives": {
                objective: [member_name(member) for member in members]
                for objective, members in requested.items()
            },
            "active_objectives": {
                objective: [member_name(member) for member in members]
                for objective, members in objectives.items()
            },
            "active_constraints": constraints,
            "property_services": service_status,
            "vina": vina_status,
        },
    )

    print(
        f"GenMol round 0: {args.samples_schedule[0]} samples per fragment",
        flush=True,
    )
    initial = run_remask(
        args.starting_smiles,
        args.output_dir / "round_00_remask.json",
        args,
        args.samples_schedule[0],
        round_index=0,
        total_rounds=args.iterations + 1,
        source_index=1,
        source_total=1,
    )
    if not initial:
        raise RuntimeError("Initial GenMol run produced no valid molecules")
    current_csv = args.output_dir / "round_00_candidates.csv"
    write_candidates(initial, current_csv, 0)

    completed_rounds: list[int] = []
    for round_index in range(args.iterations + 1):
        screened, predictions = calculate_scores(
            pd.read_csv(current_csv),
            objectives,
            constraints,
            args,
            round_index=round_index,
            total_rounds=args.iterations + 1,
        )
        screened_path = args.output_dir / f"round_{round_index:02d}_screened.csv"
        screened.to_csv(screened_path, index=False)
        if predictions.empty:
            raise RuntimeError(
                f"No candidates passed screening in round {round_index}; "
                f"see {screened_path}"
            )
        pareto = find_pareto_front(predictions)
        scored_path = args.output_dir / f"round_{round_index:02d}_scored.csv"
        pareto_path = args.output_dir / f"round_{round_index:02d}_pareto.csv"
        pairwise_path = args.output_dir / f"round_{round_index:02d}_pareto_pairwise.png"
        predictions.to_csv(scored_path, index=False)
        pareto.to_csv(pareto_path, index=False)
        plot_pareto_front(predictions, pareto, pairwise_path)
        completed_rounds.append(round_index)
        print(f"Round {round_index}: {len(pareto)} Pareto molecules", flush=True)

        if round_index == args.iterations:
            break

        next_smiles: list[str] = []
        next_round = round_index + 1
        print(
            f"GenMol round {next_round}: {args.samples_schedule[next_round]} samples per fragment",
            flush=True,
        )
        source_total = len(pareto)
        for source_index, smiles in enumerate(
            pareto["smiles"].dropna().astype(str), start=1
        ):
            next_smiles.extend(
                run_remask(
                    smiles,
                    args.output_dir
                    / f"round_{round_index + 1:02d}_source_{source_index:04d}.json",
                    args,
                    args.samples_schedule[next_round],
                    round_index=next_round,
                    total_rounds=args.iterations + 1,
                    source_index=source_index,
                    source_total=source_total,
                )
            )
        next_smiles = list(dict.fromkeys(next_smiles))
        if not next_smiles:
            print("No valid remasked molecules remain; stopping.", file=sys.stderr)
            break
        current_csv = args.output_dir / f"round_{round_index + 1:02d}_candidates.csv"
        write_candidates(next_smiles, current_csv, round_index + 1)

    write_json(
        args.output_dir / "workflow_summary.json",
        {
            "completed_rounds": completed_rounds,
            "active_objectives": {
                objective: [member_name(member) for member in members]
                for objective, members in objectives.items()
            },
            "objective_columns": OBJECTIVE_COLUMNS,
            "screening_constraints": constraints,
            "starting_smiles": args.starting_smiles,
            "receptor": args.receptor,
            "samples_schedule": args.samples_schedule,
        },
    )


if __name__ == "__main__":
    try:
        main()
    except (PredictionFailure, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
