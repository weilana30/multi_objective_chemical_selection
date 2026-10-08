"""Options exposed by the dashboard for new iterative workflows."""
from __future__ import annotations

import json
import math
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
CREM_DB = PROJECT / "assets" / "crem" / "chembl33_sa2_f5.db"
PORTS = {
    "admet_ai_port": ("Admet-AI", 18019),
    "admetica_port": ("Admetica", 8000),
    "admethyst_port": ("Admethyst", 18001),
    "boiled_egg_port": ("BoiledEgg", 18002),
    "classical_gsg_port": ("ClassicalGSG", 10001),
    "mol_property_port": ("MolProperty", 18003),
    "logpprepml_port": ("LogPPrepML", None),
}

NUMBERS = {
    "seed": ("Random seed", 0, "int", 0, None, "Both"),
    "samples_per_mask": ("Minimum samples per mask", 2, "int", 1, None, "Both"),
    "initial_mask_count": ("Initial mask sizes", 11, "int", 1, None, "Both"),
    "min_mask_tokens": ("Minimum mask tokens", 5, "int", 1, None, "Both"),
    "max_mask_tokens": ("Maximum mask tokens", 15, "int", 1, None, "Both"),
    "candidate_decay": ("Candidate decay between rounds", 0.5, "float", 0, 1, "Both"),
    "max_pareto_parents": ("Maximum Pareto parents", 3, "int", 1, None, "Both"),
    "max_generation_batches": ("Maximum generation batches", 5, "int", 1, None, "Both"),
    "temperature": ("GenMol temperature", 0.8, "float", 0, None, "GenMol"),
    "randomness": ("GenMol randomness", 0.5, "float", 0, None, "GenMol"),
    "gamma": ("GenMol gamma", 0.0, "float", 0, 1, "GenMol"),
    "crem_radius": ("CReM replacement radius", 1, "int", 0, None, "CReM"),
    "crem_max_size": ("CReM maximum fragment size", 8, "int", 1, None, "CReM"),
    "crem_growth_max_atoms": ("CReM growth atoms", 4, "int", 1, None, "CReM"),
    "generator_max_output": ("CReM products per generation call", 100, "int", 1, None, "CReM"),
    "vina_exhaustiveness": ("Vina exhaustiveness", 8, "int", 1, None, "Both"),
    "vina_poses": ("Vina poses", 5, "int", 1, None, "Both"),
}

TEXT = {
    "reference_smiles": ("Docking reference SMILES", "", "Both"),
    "service_probe_smiles": ("Prediction service probe SMILES", "", "Both"),
    "receptor": ("Receptor", "A2A_3RFM", "Both"),
    "vina_url": ("Vina service URL", "http://localhost:18025", "Both"),
    "cache_db": ("Cache database path", "", "Both"),
    "model_path": ("GenMol checkpoint path", "", "GenMol"),
    "crem_db": ("CReM fragment database path", str(CREM_DB), "CReM"),
}

CHOICES = {
    "device": ("GenMol device", "auto", ["auto", "cpu", "mps", "cuda"], "GenMol"),
    "crem_mode": ("CReM editing mode", "both", ["both", "mutate", "grow"], "CReM"),
}


def model_options() -> dict[str, list[str]]:
    registry = json.loads((PROJECT / "property_output_registry.json").read_text(encoding="utf-8"))
    return {objective: [f"{member['model']}:{member['property']}"
                        for member in registry["moo_objectives"][objective]["members"]]
            for objective in ("lipophilicity", "solubility")}


def normalize(raw: dict, generator: str) -> dict:
    """Use workflow defaults for blank fields and reject invalid overrides."""
    options = {}
    for key, (label, default, kind, minimum, maximum, applies_to) in NUMBERS.items():
        if applies_to not in ("Both", generator):
            continue
        value = raw.get(key)
        value = default if value in (None, "") else value
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{label} must be a number.") from None
        if not math.isfinite(number) or (kind == "int" and not number.is_integer()):
            raise ValueError(f"{label} must be a finite {'whole number' if kind == 'int' else 'number'}.")
        if number < minimum or (maximum is not None and number > maximum):
            raise ValueError(f"{label} must be between {minimum} and {maximum if maximum is not None else 'above'}.")
        options[key] = int(number) if kind == "int" else number
    if not 0 < options["candidate_decay"] < 1:
        raise ValueError("Candidate decay must be greater than 0 and less than 1.")
    if options["max_mask_tokens"] < options["min_mask_tokens"]:
        raise ValueError("Maximum mask tokens must be at least the minimum.")
    if generator == "GenMol" and options["temperature"] <= 0:
        raise ValueError("GenMol temperature must be greater than 0.")
    for key, (_, default, applies_to) in TEXT.items():
        if applies_to in ("Both", generator):
            options[key] = str(raw.get(key) or default).strip()
    for key, (label, default, choices, applies_to) in CHOICES.items():
        if applies_to in ("Both", generator):
            value = raw.get(key) or default
            if value not in choices:
                raise ValueError(f"Choose a valid {label}.")
            options[key] = value
    available = model_options()
    for objective in ("lipophilicity", "solubility"):
        key = objective + "_members"
        selected = raw.get(key) or []
        if any(name not in available[objective] for name in selected) or len(selected) != len(set(selected)):
            raise ValueError(f"Choose valid, unique {objective} predictions.")
        options[key] = selected
    for key, (label, default) in PORTS.items():
        value = raw.get(key)
        value = default if value in (None, "") else value
        if value is not None:
            try:
                port = int(value)
            except (TypeError, ValueError):
                raise ValueError(f"{label} port must be a whole number.") from None
            if not 1 <= port <= 65535:
                raise ValueError(f"{label} port must be between 1 and 65535.")
            value = port
        options[key] = value
    if any(name.startswith("LogPPrepML:") for name in options["lipophilicity_members"]):
        if options["logpprepml_port"] is None:
            raise ValueError("Enter the LogPPrepML port when selecting its predictions.")
    options["retry_failed_cache"] = bool(raw.get("retry_failed_cache"))
    options["require_all_services"] = bool(raw.get("require_all_services"))
    return options
