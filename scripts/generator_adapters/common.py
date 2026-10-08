#!/usr/bin/env python3
"""Shared adapter layer for generator-specific iterative dock/Pareto runs."""
from __future__ import annotations

import csv
import importlib.util
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

PROJECT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = PROJECT / "scripts" / "iterative_dock_pareto" / "iterative_dock_pareto.py"


def _base_module():
    sys.path.insert(0, str(BASE_SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("iterative_dock_pareto_base", BASE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _toml_string(value: str) -> str:
    return json.dumps(str(value))


def install(kind: str) -> None:
    """Add one generator adapter to the established workflow and run its CLI."""
    base = _base_module()
    # Keep each generator's global archive and default cache independent.
    base.OUTPUT_ROOT = PROJECT / "outputs" / f"{kind}_dock_pareto"
    original_parser = base.build_parser
    original_property_preflight = base.legacy.check_property_services

    def parser_with_adapter_args():
        parser = original_parser()
        parser.add_argument("--generator-max-output", type=int, default=100)
        if kind == "crem":
            parser.add_argument("--crem-db", type=Path, default=os.getenv("CREM_DB"))
            parser.add_argument("--crem-radius", type=int, default=1)
            parser.add_argument("--crem-max-size", type=int, default=8)
            parser.add_argument("--crem-growth-max-atoms", type=int, default=4)
            parser.add_argument("--crem-mode", choices=("mutate", "grow", "both"), default="both")
        elif kind == "molmim":
            parser.add_argument("--molmim-url", default=os.getenv("MOLMIM_URL", "http://localhost:8008"))
            parser.add_argument("--molmim-radius", type=float, default=0.7)
            parser.add_argument("--molmim-beam-size", type=int, default=1)
        elif kind == "reinvent4_mol2mol":
            parser.add_argument("--reinvent-executable", default=os.getenv("REINVENT", "reinvent"))
            parser.add_argument("--reinvent-model", type=Path, default=os.getenv("REINVENT_MOL2MOL_PRIOR"))
            parser.add_argument("--reinvent-device", default="cpu")
        return parser

    base.build_parser = parser_with_adapter_args

    def validate_backend(args):
        if not args.starting_smiles:
            raise RuntimeError(f"{kind} analog generation requires --starting-smiles (for example, caffeine)")
        if kind == "crem":
            if not args.crem_db or not Path(args.crem_db).is_file():
                raise RuntimeError("CReM requires --crem-db or CREM_DB pointing to a prepared fragment database")
            try:
                import crem  # noqa: F401
            except ImportError as exc:
                raise RuntimeError("CReM is not installed in this Python environment; install it with pip install crem") from exc
        elif kind == "molmim":
            parsed = urlparse(args.molmim_url)
            host, port = parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)
            try:
                with socket.create_connection((host, port), timeout=2):
                    pass
            except OSError as exc:
                raise RuntimeError(f"MolMIM service is not reachable at {args.molmim_url}") from exc
        else:
            executable = shlex.split(args.reinvent_executable)[0]
            if not (Path(executable).is_file() or shutil.which(executable)):
                raise RuntimeError(f"REINVENT4 executable not found: {args.reinvent_executable}")
            if not args.reinvent_model or not Path(args.reinvent_model).is_file():
                raise RuntimeError("REINVENT4 Mol2Mol requires --reinvent-model pointing to a downloaded Mol2Mol prior")

    def property_preflight_with_generator_check(args, requested, constraints):
        validate_backend(args)
        return original_property_preflight(args, requested, constraints)

    base.legacy.check_property_services = property_preflight_with_generator_check

    def generate(parent_smiles, output_path, args, samples_per_mask, *_metadata):
        started = time.perf_counter()
        requested = min(
            args.generator_max_output,
            max(1, int(samples_per_mask) * max(1, args.max_mask_tokens - args.min_mask_tokens + 1)),
        )
        if kind == "crem":
            try:
                from rdkit import Chem
                from crem.crem import grow_mol, mutate_mol
            except ImportError as exc:
                raise RuntimeError("Install CReM and RDKit in the Python environment used to run this workflow") from exc
            mol = Chem.MolFromSmiles(parent_smiles) if parent_smiles else None
            if mol is None:
                raise RuntimeError("CReM analog generation requires a valid --starting-smiles seed")
            products = []
            if args.crem_mode in {"mutate", "both"}:
                products.extend(mutate_mol(
                    mol, db_name=str(args.crem_db), radius=args.crem_radius,
                    max_size=args.crem_max_size, max_replacements=requested, seed=int(args.seed),
                ))
            if args.crem_mode in {"grow", "both"} and len(products) < requested:
                products.extend(grow_mol(
                    mol, db_name=str(args.crem_db), radius=args.crem_radius,
                    max_atoms=args.crem_growth_max_atoms,
                    max_replacements=requested, seed=int(args.seed) + 1,
                ))
            products = list(dict.fromkeys(products))[:requested]
        elif kind == "molmim":
            products = []
            with requests.Session() as session:
                for _ in range((requested + 9) // 10):
                    try:
                        response = session.post(
                            args.molmim_url.rstrip("/") + "/sampling",
                            json={"sequences": [parent_smiles], "num_molecules": min(10, requested-len(products)),
                                  "beam_size": args.molmim_beam_size, "scaled_radius": args.molmim_radius},
                            timeout=(5, 300),
                        )
                    except requests.RequestException as exc:
                        raise RuntimeError(f"MolMIM is not reachable at {args.molmim_url}: {exc}") from exc
                    if not response.ok:
                        raise RuntimeError(f"MolMIM /sampling returned HTTP {response.status_code}: {response.text[:300]}")
                    body = response.json()
                    generated = body.get("generated") if isinstance(body, dict) else None
                    if not isinstance(generated, list) or not generated or not isinstance(generated[0], list):
                        raise RuntimeError(f"Unexpected MolMIM response shape: {body}")
                    products.extend(str(s) for s in generated[0] if s)
                    if len(products) >= requested:
                        break
        else:
            with tempfile.TemporaryDirectory(prefix="reinvent4-mol2mol-") as temp_name:
                temp = Path(temp_name)
                seed_path, csv_path, config_path = temp / "seed.smi", temp / "generated.csv", temp / "sampling.toml"
                seed_path.write_text(parent_smiles + "\n", encoding="utf-8")
                config_path.write_text(
                    'run_type = "sampling"\n' + f'device = {_toml_string(args.reinvent_device)}\n' +
                    "[parameters]\n" + f'model_file = {_toml_string(str(Path(args.reinvent_model).resolve()))}\n' +
                    f'smiles_file = {_toml_string(str(seed_path))}\n' +
                    'sample_strategy = "multinomial"\n' + f'output_file = {_toml_string(str(csv_path))}\n' +
                    f'num_smiles = {requested}\nunique_molecules = true\nrandomize_smiles = true\n', encoding="utf-8")
                command = shlex.split(args.reinvent_executable) + [str(config_path)]
                try:
                    process = subprocess.run(command, cwd=temp, text=True, capture_output=True, timeout=600)
                except FileNotFoundError as exc:
                    raise RuntimeError(f"REINVENT4 executable not found: {args.reinvent_executable}") from exc
                if process.returncode:
                    raise RuntimeError(f"REINVENT4 failed ({process.returncode}): {(process.stderr or process.stdout)[-1200:]}")
                if not csv_path.is_file():
                    raise RuntimeError("REINVENT4 completed but did not create its sampling CSV")
                with csv_path.open(newline="", encoding="utf-8") as stream:
                    rows = list(csv.DictReader(stream))
                smiles_col = next((key for key in ("SMILES", "smiles") if rows and key in rows[0]), None)
                if not smiles_col:
                    raise RuntimeError("REINVENT4 sampling CSV has no SMILES column")
                products = [row[smiles_col] for row in rows if row.get(smiles_col)]

        # Keep small per-call generation metadata alongside the legacy workflow artifacts.
        metadata_path = Path(output_path).with_suffix(".adapter.json")
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps({"generator": kind, "requested": requested,
            "generated": len(products), "generation_seconds": round(time.perf_counter()-started, 4)}, indent=2), encoding="utf-8")
        return products

    base.legacy.run_remask = generate
    try:
        base.main()
    except (base.legacy.PredictionFailure, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
