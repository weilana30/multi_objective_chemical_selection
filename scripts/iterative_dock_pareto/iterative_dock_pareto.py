#!/usr/bin/env python3
"""Iteratively dock generated molecules, cache predictions, and maintain cumulative Pareto fronts."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "outputs" / "iterative_dock_pareto"
GENMOL_ROOT = ROOT.parent / "genmol_molecule_generator"
GENMOL_PYTHON = GENMOL_ROOT / ".conda-env" / "bin" / "python"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "iterative_pareto_docking"))
import iterative_pareto_docking as legacy
from aggregate_pareto import member_name
from molecule_cache import MoleculeCache, canonicalize_smiles

SERVICE_PORTS = {**legacy.SERVICE_DEFAULT_PORTS, "Admetica": 8000}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iterations", "-n", type=int, default=5, help="Refinement rounds after the initial generation.")
    p.add_argument("--output-dir", type=Path, help="Run output directory; default is a timestamped subfolder under outputs/iterative_dock_pareto.")
    p.add_argument("--resume", action="store_true", help="Reuse saved round candidate CSVs in --output-dir and continue with cached predictions.")
    p.add_argument("--cache-db", type=Path, default=OUTPUT_ROOT / "molecule_cache.sqlite3")
    p.add_argument("--starting-smiles", help="Optional seed molecule; omit for de novo initial generation.")
    p.add_argument("--model-path", type=Path)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--samples-per-mask", type=int, default=2, help="Minimum samples per mask in the initial round; the total-budget planner can raise it.")
    p.add_argument("--initial-mask-count", type=int, default=11, help="Maximum mask sizes used initially; the budget planner selects the needed number.")
    p.add_argument("--min-mask-tokens", type=int, default=5)
    p.add_argument("--max-mask-tokens", type=int, default=15)
    p.add_argument("--candidate-decay", type=float, default=0.5, help="Maximum next-round dock pool as a fraction of the prior round's pool.")
    p.add_argument("--total-candidate-budget", type=int, default=1000, help="Target number of candidates to dock across all rounds in this run.")
    p.add_argument("--max-pareto-parents", type=int, default=3, help="Upper limit for Pareto parents; the workflow lowers this as the candidate budget shrinks.")
    p.add_argument("--max-generation-batches", type=int, default=5, help="Maximum differently seeded GenMol batches used to fill a round's candidate budget.")
    p.add_argument("--retry-failed-cache", action="store_true", help="Retry cached docking/property failures.")
    p.add_argument("--caffeine-smiles", default=legacy.CAFFEINE_SMILES)
    p.add_argument("--vina-url", default="http://localhost:18025")
    p.add_argument("--receptor", default="A2A_3RFM")
    p.add_argument("--vina-exhaustiveness", type=int, default=8)
    p.add_argument("--vina-poses", type=int, default=5)
    p.add_argument("--service-probe-smiles", default=legacy.CAFFEINE_SMILES)
    p.add_argument("--require-all-services", action="store_true")
    for model, default_port in SERVICE_PORTS.items():
        dest = model.lower().replace("-", "_") + "_port"
        p.add_argument("--" + dest.replace("_", "-"), dest=dest, type=int, default=default_port)
    return p


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def round_dir(output_dir: Path, round_index: int) -> Path:
    path = output_dir / f"round_{round_index:02d}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def round_file(output_dir: Path, round_index: int, suffix: str) -> Path:
    """Prefer organized round paths while still resuming old flat run folders."""
    organized = output_dir / f"round_{round_index:02d}" / f"round_{round_index:02d}_{suffix}"
    legacy = output_dir / f"round_{round_index:02d}_{suffix}"
    return organized if organized.is_file() or not legacy.is_file() else legacy


def write_round_figures(
    round_path: Path,
    docked: pd.DataFrame,
    scored: pd.DataFrame,
    pareto: pd.DataFrame,
    global_front: pd.DataFrame,
    caffeine_score: float,
    round_index: int,
) -> None:
    """Save readable docking and property/Pareto figures for this round."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, (dock_axis, property_axis) = plt.subplots(
        2, 1, figsize=(10, 9), constrained_layout=True, gridspec_kw={"height_ratios": [1, 1.2]}
    )
    if not docked.empty and "docking_score" in docked:
        values = pd.to_numeric(docked["docking_score"], errors="coerce")
        passed = docked.get("beats_caffeine", pd.Series(False, index=docked.index)).fillna(False).astype(bool)
        dock_axis.scatter(range(1, len(values) + 1), values, c=passed.map({True: "#218c74", False: "#8492a6"}), s=34, alpha=.8)
        dock_axis.axhline(caffeine_score, color="#c0392b", linestyle="--", linewidth=1.6, label=f"Caffeine ({caffeine_score:.2f})")
        dock_axis.set_xlim(0, max(2, len(values) + 1))
        dock_axis.legend(frameon=False, loc="best")
        dock_axis.set_ylabel("Docking score (higher is better)")
    else:
        dock_axis.text(.5, .5, "No docked candidates", ha="center", va="center", transform=dock_axis.transAxes)
    dock_axis.set_title(f"Round {round_index:02d}: docking results")
    dock_axis.set_xlabel("Candidate")
    for spine in ("top", "right"):
        dock_axis.spines[spine].set_visible(False)

    has_property_data = not scored.empty or not pareto.empty
    if not scored.empty:
        property_axis.scatter(scored["lipophilicity_score"], scored["solubility_score"], color="#8492a6", s=36, alpha=.7, label="Scored this round")
    if not pareto.empty:
        ordered = pareto.sort_values("lipophilicity_score")
        property_axis.plot(ordered["lipophilicity_score"], ordered["solubility_score"], color="#c0392b", linewidth=1.7, alpha=.75)
        property_axis.scatter(ordered["lipophilicity_score"], ordered["solubility_score"], color="#c0392b", edgecolor="white", linewidth=.5, s=54, label="Cumulative run Pareto")
        property_axis.legend(frameon=False, loc="best")
    else:
        message = "No molecules passed caffeine cutoff" if scored.empty else "No cumulative Pareto points"
        property_axis.set_axis_off()
        property_axis.text(.5, .5, message, ha="center", va="center", transform=property_axis.transAxes)
    property_axis.set_title("Lipophilicity vs. solubility")
    if has_property_data:
        property_axis.set_xlabel("Lipophilicity score (higher is better)")
        property_axis.set_ylabel("Solubility score (higher is better)")
        for spine in ("top", "right"):
            property_axis.spines[spine].set_visible(False)
    figure.savefig(round_path / f"round_{round_index:02d}_summary.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    global_figure, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    if not global_front.empty:
        ordered = global_front.sort_values("lipophilicity_score")
        axis.plot(ordered["lipophilicity_score"], ordered["solubility_score"], color="#c0392b", linewidth=2, marker="o", markersize=5)
        axis.scatter(ordered["lipophilicity_score"], ordered["solubility_score"], color="#c0392b", edgecolor="white", linewidth=.5, s=50, label="Global Pareto")
        axis.legend(frameon=False)
    else:
        axis.set_axis_off()
        axis.text(.5, .5, "No scored candidates yet", ha="center", va="center", transform=axis.transAxes)
    axis.set_title(f"Global Pareto after round {round_index:02d}")
    if not global_front.empty:
        axis.set_xlabel("Lipophilicity score (higher is better)")
        axis.set_ylabel("Solubility score (higher is better)")
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    global_figure.savefig(round_path.parent / "global_pareto.png", dpi=200, bbox_inches="tight")
    plt.close(global_figure)


def write_run_summary_figures(run_dir: Path, summary: dict[str, Any] | None = None) -> tuple[Path, Path]:
    """Write end-of-run Pareto evolution and candidate-flow figures from saved artifacts."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if summary is None:
        summary = json.loads((run_dir / "workflow_summary.json").read_text(encoding="utf-8"))
    rounds = summary.get("rounds", [])
    completed = set(summary.get("completed_rounds", []))
    cmap = plt.get_cmap("viridis")
    round_colors = {int(item["round"]): cmap(i / max(1, len(rounds) - 1)) for i, item in enumerate(rounds)}

    evolution_path = run_dir / "pareto_evolution.png"
    figure, (front_axis, count_axis) = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    for item in rounds:
        round_index = int(item["round"])
        if round_index not in completed:
            continue
        pareto_path = round_file(run_dir, round_index, "pareto.csv")
        if not pareto_path.is_file():
            continue
        frame = pd.read_csv(pareto_path)
        if not {"lipophilicity_score", "solubility_score"} <= set(frame.columns) or frame.empty:
            continue
        frame = frame.sort_values("lipophilicity_score")
        color = round_colors[round_index]
        front_axis.plot(frame["lipophilicity_score"], frame["solubility_score"], color=color, alpha=.45, linewidth=1.2)
        front_axis.scatter(frame["lipophilicity_score"], frame["solubility_score"], color=[color], s=22, alpha=.6, label=f"Round {round_index:02d}")

    global_path = run_dir / "global_pareto.csv"
    if global_path.is_file():
        final = pd.read_csv(global_path)
        if {"lipophilicity_score", "solubility_score"} <= set(final.columns) and not final.empty:
            final = final.sort_values("lipophilicity_score")
            front_axis.plot(final["lipophilicity_score"], final["solubility_score"], color="#c0392b", linewidth=2.5, zorder=4, label="Final global front")
            front_axis.scatter(final["lipophilicity_score"], final["solubility_score"], marker="*", color="#c0392b", edgecolor="white", linewidth=.5, s=100, zorder=5)
    front_axis.set_title("Lipophilicity–solubility front by round")
    front_axis.set_xlabel("Lipophilicity score (higher is better)")
    front_axis.set_ylabel("Solubility score (higher is better)")
    front_axis.grid(alpha=.18)
    front_axis.legend(frameon=False, fontsize=8, ncol=2)

    round_numbers = [int(item["round"]) for item in rounds if int(item["round"]) in completed]
    run_front_counts = [int(item.get("cumulative_run_pareto_count", 0)) for item in rounds if int(item["round"]) in completed]
    global_front_counts = [int(item.get("global_pareto_count", 0)) for item in rounds if int(item["round"]) in completed]
    count_axis.plot(round_numbers, run_front_counts, marker="o", linewidth=2, color="#2471a3", label="Current-run front")
    count_axis.plot(round_numbers, global_front_counts, marker="s", linewidth=2, color="#c0392b", label="Global front incl. archive")
    count_axis.set_title("Pareto set size over rounds")
    count_axis.set_xlabel("Round")
    count_axis.set_ylabel("Number of molecules")
    count_axis.set_xticks(round_numbers)
    count_axis.grid(alpha=.2)
    count_axis.legend(frameon=False)
    figure.suptitle("Pareto evolution", fontsize=16)
    figure.savefig(evolution_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    metrics_path = run_dir / "run_metrics.png"
    figure, (flow_axis, pie_axis) = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True, gridspec_kw={"width_ratios": [1.5, 1]})
    metric_names = ["Submitted to docking", "Docked", "Beat caffeine", "Property scored"]
    metric_keys = ["candidate_count", "docked_successfully", "beats_caffeine", "property_scored"]
    x = np.arange(len(round_numbers))
    width = .18
    for metric_index, (metric_name, key) in enumerate(zip(metric_names, metric_keys)):
        vals = []
        for round_index in round_numbers:
            item = next(entry for entry in rounds if int(entry["round"]) == round_index)
            value = item.get(key, 0)
            vals.append(int(value or 0))
        flow_axis.bar(x + (metric_index - 1.5) * width, vals, width, label=metric_name)
    generated = []
    for round_index in round_numbers:
        generation_path = round_file(run_dir, round_index, "generation_summary.json")
        generation = json.loads(generation_path.read_text()) if generation_path.is_file() else {}
        raw_count = generation.get("generated_raw")
        if raw_count is None:
            source_files = sorted(run_dir.glob(f"round_{round_index:02d}_source_*.json"))
            if not source_files:
                legacy_remask = run_dir / f"round_{round_index:02d}_remask.json"
                source_files = [legacy_remask] if legacy_remask.is_file() else []
            raw_count = sum(
                int(json.loads(source.read_text()).get("total_valid_samples_before_deduplication", 0) or 0)
                for source in source_files
            )
        generated.append(int(raw_count or 0))
    raw_axis = flow_axis.twinx()
    raw_axis.plot(x, generated, color="#2c3e50", marker="D", linestyle="--", linewidth=1.6, label="Generated raw")
    flow_axis.set_title("Candidate counts by round")
    flow_axis.set_xlabel("Round")
    flow_axis.set_ylabel("Molecules")
    flow_axis.set_xticks(x, [f"{i:02d}" for i in round_numbers])
    flow_axis.grid(axis="y", alpha=.2)
    raw_axis.set_ylabel("Raw generator proposals")
    handles, labels = flow_axis.get_legend_handles_labels()
    raw_handles, raw_labels = raw_axis.get_legend_handles_labels()
    flow_axis.legend(handles + raw_handles, labels + raw_labels, frameon=False, fontsize=8, ncol=2, loc="upper right")

    final = pd.read_csv(global_path) if global_path.is_file() else pd.DataFrame()
    final_smiles = set(final.get("canonical_smiles", pd.Series(dtype=str)).fillna("").astype(str))
    first_round: dict[str, int] = {}
    for round_index in round_numbers:
        candidate_path = round_file(run_dir, round_index, "candidates.csv")
        if not candidate_path.is_file():
            continue
        candidate_frame = pd.read_csv(candidate_path)
        smiles_column = "canonical_smiles" if "canonical_smiles" in candidate_frame else "smiles"
        for smiles in candidate_frame[smiles_column].fillna("").astype(str):
            first_round.setdefault(smiles, round_index)
    contribution: dict[int | str, int] = {}
    for smiles in final_smiles:
        round_index = first_round.get(smiles, "Earlier archive")
        contribution[round_index] = contribution.get(round_index, 0) + 1
    contribution = dict(sorted(contribution.items(), key=lambda item: (-1 if item[0] == "Earlier archive" else item[0])))
    labels = [f"Round {key:02d}" if isinstance(key, int) else key for key in contribution]
    sizes = list(contribution.values())
    colors = [round_colors.get(key, "#9e9e9e") if isinstance(key, int) else "#9e9e9e" for key in contribution]
    if sizes:
        wedges, _, _ = pie_axis.pie(sizes, colors=colors, autopct=lambda pct: f"{pct:.0f}%" if pct >= 5 else "", startangle=90, counterclock=False, wedgeprops={"edgecolor": "white", "linewidth": 1})
        legend_labels = [f"{label}: {count}" for label, count in zip(labels, sizes)]
        pie_axis.legend(wedges, legend_labels, title="First appearance", frameon=False, fontsize=8, title_fontsize=9, loc="center left", bbox_to_anchor=(.92, .5))
    else:
        pie_axis.text(.5, .5, "No final Pareto members", ha="center", va="center", transform=pie_axis.transAxes)
    pie_axis.set_title(f"Final global Pareto members by first appearance (n={len(final_smiles)})")
    figure.suptitle("Run funnel and final Pareto contribution", fontsize=16)
    figure.savefig(metrics_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return evolution_path, metrics_path


def round_csv_paths(directory: Path, pattern: str) -> list[Path]:
    """Find round artifacts from both organized runs and the previous flat layout."""
    return sorted(set(directory.glob(pattern)) | {
        path for round_folder in directory.glob("round_[0-9][0-9]")
        for path in round_folder.glob(pattern)
    })


def build_protocol(args: argparse.Namespace, objectives) -> str:
    protocol = {
        "caffeine": canonicalize_smiles([args.caffeine_smiles], GENMOL_PYTHON)[0],
        "vina": {
            "receptor": args.receptor,
            "exhaustiveness": args.vina_exhaustiveness,
            "poses": args.vina_poses,
            "seed": args.seed,
        },
        "objective_directions": {"lipophilicity_score": "maximize", "solubility_score": "maximize"},
        "objectives": {
            name: sorted(
                [
                    {
                        "member": member_name(member),
                        "transform": member.get("transform", "identity"),
                        "weight": float(member.get("weight", 1.0)),
                    }
                    for member in members
                ],
                key=lambda item: item["member"],
            )
            for name, members in objectives.items()
        },
    }
    encoded = json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def find_pareto_front(scored: pd.DataFrame) -> pd.DataFrame:
    """Maximize aggregate lipophilicity and aggregate solubility."""
    if scored.empty:
        return scored.copy()
    lipo = scored["lipophilicity_score"].to_numpy(dtype=float)
    sol = scored["solubility_score"].to_numpy(dtype=float)
    keep = []
    for index in range(len(scored)):
        dominated = (
            (lipo >= lipo[index])
            & (sol >= sol[index])
            & ((lipo > lipo[index]) | (sol > sol[index]))
        ).any()
        if not dominated:
            keep.append(index)
    return scored.iloc[keep].sort_values(
        ["lipophilicity_score", "solubility_score"], ascending=[False, False]
    ).copy()


def cache_row(entry: dict[str, Any], candidate: dict[str, Any], baseline: float) -> dict[str, Any]:
    row = dict(candidate)
    row.update({
        "canonical_smiles": entry["canonical_smiles"],
        "docking_affinity": entry.get("docking_affinity"),
        "docking_score": entry.get("docking_score"),
        "caffeine_docking_score": baseline,
        "beats_caffeine": entry.get("docking_score") is not None and entry["docking_score"] > baseline,
        "lipophilicity_score": entry.get("lipophilicity_score"),
        "solubility_score": entry.get("solubility_score"),
        "cache_status": entry.get("status"),
        "cache_hit": True,
        "error": entry.get("error"),
    })
    return row


def score_properties(compounds: pd.DataFrame, objectives, args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, str]]:
    properties_by_model: dict[str, set[str]] = {}
    for members in objectives.values():
        for member in members:
            properties_by_model.setdefault(member["model"], set()).add(member["property"])
    scored_rows = []
    failures: dict[str, str] = {}
    with requests.Session() as session:
        for position, (_, compound) in enumerate(compounds.iterrows(), 1):
            row = compound.to_dict()
            name = str(row.get("preferred_name", f"candidate_{position}"))
            canonical = str(row["canonical_smiles"])
            try:
                for model, properties in properties_by_model.items():
                    predictions = legacy.request_property(
                        session,
                        model,
                        getattr(args, model.lower().replace("-", "_") + "_port"),
                        sorted(properties),
                        str(row["smiles"]),
                        name,
                    )
                    row.update({f"{model}:{key}": value for key, value in predictions.items()})
                for objective, members in objectives.items():
                    total = 0.0
                    for member in members:
                        value = float(row[member_name(member)])
                        transform = member.get("transform", "identity")
                        if transform == "negate":
                            value = -value
                        elif transform != "identity":
                            raise ValueError(f"Unsupported transform {transform!r}")
                        total += float(member.get("weight", 1.0)) * value
                    row[objective + "_score"] = total
                row["cache_hit"] = False
                row["cache_status"] = "scored"
                scored_rows.append(row)
            except Exception as exc:
                failures[canonical] = str(exc)
                print(f"WARNING: property scoring failed for {name}: {exc}", file=sys.stderr, flush=True)
            print(f"Property scoring {position}/{len(compounds)}", flush=True)
    return pd.DataFrame(scored_rows), failures


def candidate_budget_schedule(total: int, round_count: int, decay: float) -> list[int]:
    weights = [decay**index for index in range(round_count)]
    weight_sum = sum(weights)
    exact = [total * weight / weight_sum for weight in weights]
    budgets = [math.floor(value) for value in exact]
    remainder = total - sum(budgets)
    order = sorted(range(round_count), key=lambda index: exact[index] - budgets[index], reverse=True)
    for index in order[:remainder]:
        budgets[index] += 1
    return budgets


def reallocate_budget_shortfall(
    budgets: list[int], completed_round: int, actual_candidates: int, decay: float
) -> tuple[list[int], int]:
    """Carry unused unique-candidate budget forward to rounds that can still generate."""
    updated_budgets = budgets.copy()
    shortfall = max(0, budgets[completed_round] - actual_candidates)
    remaining_rounds = len(updated_budgets) - completed_round - 1
    if shortfall and remaining_rounds:
        additions = candidate_budget_schedule(shortfall, remaining_rounds, decay)
        for offset, addition in enumerate(additions, start=completed_round + 1):
            updated_budgets[offset] += addition
    return updated_budgets, shortfall


def estimate_fragment_count(smiles: str) -> int:
    """Approximate GenMol SAFE fragments from molecule length for budget planning."""
    return max(4, min(8, round(len(smiles) / 10)))


def plan_generation(
    budget: int,
    parent_count: int,
    max_mask_count: int,
    expected_fragments: int,
    minimum_samples: int = 1,
) -> dict[str, int | float]:
    """Choose mask/sample settings whose expected valid output is near the dock budget."""
    valid_rate = 0.65
    options = []
    for mask_count in range(1, max_mask_count + 1):
        samples = max(
            minimum_samples,
            math.ceil(budget / (parent_count * mask_count * expected_fragments * valid_rate)),
        )
        expected = parent_count * mask_count * expected_fragments * samples * valid_rate
        options.append((expected - budget, -mask_count, mask_count, samples, expected))
    _, _, masks, samples, expected = min(options)
    return {
        "parents": parent_count,
        "mask_count": masks,
        "samples_per_mask": samples,
        "expected_valid_candidates": expected,
    }


def generation_args_for_round(args: argparse.Namespace, generation_round: int, mask_count: int) -> argparse.Namespace:
    """Restrict mask conditions before invoking GenMol for this generation."""
    generation_args = copy.copy(args)
    available = args.max_mask_tokens - args.min_mask_tokens + 1
    if generation_round == 0:
        start_index = 0
    else:
        start_index = (generation_round - 1) % max(1, available - mask_count + 1)
    generation_args.min_mask_tokens = args.min_mask_tokens + start_index
    generation_args.max_mask_tokens = min(args.max_mask_tokens, generation_args.min_mask_tokens + mask_count - 1)
    return generation_args


def generate_to_budget(
    parents: list[str | None],
    generation_args: argparse.Namespace,
    output_dir: Path,
    round_index: int,
    total_rounds: int,
    samples_per_mask: int,
    budget: int,
    seen_smiles: set[str],
    seed: int,
    max_batches: int,
) -> tuple[dict[str, str], int, int]:
    generated: dict[str, str] = {}
    generated_raw = 0
    stagnant_batches = 0
    batch_count = 0
    while len(generated) < budget and batch_count < max_batches:
        before_batch = len(generated)
        for parent_index, smiles in enumerate(parents):
            batch_args = copy.copy(generation_args)
            batch_args.seed = seed + batch_count * len(parents) + parent_index
            source_number = batch_count * len(parents) + parent_index + 1
            output_path = round_dir(output_dir, round_index) / (
                f"round_{round_index:02d}_source_{source_number:04d}_batch_{batch_count:02d}.json"
            )
            variants = legacy.run_remask(
                smiles,
                output_path,
                batch_args,
                samples_per_mask,
                round_index,
                total_rounds,
                source_number,
                len(parents) * max_batches,
            )
            generated_raw += len(variants)
            canonical = canonicalize_smiles(variants, GENMOL_PYTHON)
            for variant, key in zip(variants, canonical):
                if key not in seen_smiles and key not in generated:
                    generated[key] = key
        batch_count += 1
        stagnant_batches = stagnant_batches + 1 if len(generated) == before_batch else 0
        if stagnant_batches >= 2:
            break
    return generated, batch_count, generated_raw


def select_pool(smiles: list[str], budget: int, seed: int) -> list[str]:
    if len(smiles) <= budget:
        return smiles
    indexes = sorted(random.Random(seed).sample(range(len(smiles)), budget))
    return [smiles[index] for index in indexes]


def select_parents(front: pd.DataFrame, limit: int) -> list[str]:
    smiles = front["smiles"].dropna().astype(str).tolist()
    if len(smiles) <= limit:
        return smiles
    indexes = [round(i * (len(smiles) - 1) / (limit - 1)) for i in range(limit)] if limit > 1 else [0]
    return [smiles[index] for index in indexes]


def import_previous_runs(
    cache: MoleculeCache,
    protocol: str,
    output_root: Path,
    current_run_id: str,
    objectives,
    args: argparse.Namespace,
) -> dict[str, int]:
    expected_members = {
        key: sorted(member_name(member) for member in members)
        for key, members in objectives.items()
    }
    counts = {"runs": 0, "docked": 0, "scored": 0, "pareto_memberships": 0}
    round_pattern = re.compile(r"round_(\d+)_")
    for directory in output_root.iterdir() if output_root.exists() else []:
        if not directory.is_dir() or directory.name == current_run_id or cache.was_imported(protocol, directory.name):
            continue
        status_path = directory / "service_status.json"
        reference_path = directory / "caffeine_reference.json"
        if not status_path.is_file() or not reference_path.is_file():
            continue
        try:
            service_status = json.loads(status_path.read_text(encoding="utf-8"))
            reference = json.loads(reference_path.read_text(encoding="utf-8"))
            vina = service_status["vina"]
            active = service_status.get("active_objectives", {})
            old_members = {key: sorted(values) for key, values in active.items()}
            if old_members != expected_members:
                continue
            if (
                vina.get("receptor") != args.receptor
                or int(vina.get("exhaustiveness", -1)) != args.vina_exhaustiveness
                or int(vina.get("poses", -1)) != args.vina_poses
                or reference.get("receptor") != args.receptor
            ):
                continue
        except (OSError, ValueError, KeyError, TypeError):
            continue

        run_id = directory.name
        caffeine_key = canonicalize_smiles([str(reference["smiles"])], GENMOL_PYTHON)[0]
        cache.store(
            protocol, caffeine_key, caffeine_key,
            docking_affinity=reference.get("docking_affinity"),
            docking_score=reference.get("docking_score"),
            status="docked",
        )
        for csv_path in round_csv_paths(directory, "round_*_docked.csv"):
            match = round_pattern.match(csv_path.name)
            if not match:
                continue
            round_index = int(match.group(1))
            frame = read_csv_or_empty(csv_path)
            if "smiles" not in frame:
                continue
            canonical = canonicalize_smiles(frame["smiles"].fillna("").astype(str).tolist(), GENMOL_PYTHON)
            for (_, row), key in zip(frame.iterrows(), canonical):
                cache.store(
                    protocol, key, str(row["smiles"]),
                    docking_affinity=row.get("docking_affinity"),
                    docking_score=row.get("docking_score"),
                    status="docked",
                )
                cache.mark_candidate(protocol, run_id, round_index, key, "docked")
                counts["docked"] += 1
        for csv_path in round_csv_paths(directory, "round_*_property_scored.csv"):
            match = round_pattern.match(csv_path.name)
            if not match:
                continue
            round_index = int(match.group(1))
            frame = read_csv_or_empty(csv_path)
            if not {"smiles", "lipophilicity_score", "solubility_score"} <= set(frame.columns):
                continue
            canonical = canonicalize_smiles(frame["smiles"].fillna("").astype(str).tolist(), GENMOL_PYTHON)
            for (_, row), key in zip(frame.iterrows(), canonical):
                cache.store(
                    protocol, key, str(row["smiles"]),
                    docking_affinity=row.get("docking_affinity"),
                    docking_score=row.get("docking_score"),
                    lipophilicity_score=row["lipophilicity_score"],
                    solubility_score=row["solubility_score"],
                    status="scored",
                )
                cache.mark_candidate(protocol, run_id, round_index, key, "scored")
                counts["scored"] += 1
        for csv_path in round_csv_paths(directory, "round_*_pareto.csv"):
            match = round_pattern.match(csv_path.name)
            if not match:
                continue
            round_index = int(match.group(1))
            frame = read_csv_or_empty(csv_path)
            if "smiles" not in frame:
                continue
            canonical = canonicalize_smiles(frame["smiles"].fillna("").astype(str).tolist(), GENMOL_PYTHON)
            cache.mark_pareto(protocol, run_id, round_index, canonical)
            counts["pareto_memberships"] += len(canonical)
        cache.record_imported_run(protocol, run_id)
        counts["runs"] += 1
    return counts


def write_global_outputs(cache: MoleculeCache, protocol: str, output_root: Path, run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    archive = pd.DataFrame(cache.global_scored(protocol))
    front = find_pareto_front(archive)
    output_root.mkdir(parents=True, exist_ok=True)
    archive.to_csv(output_root / "global_scored_archive.csv", index=False)
    front.to_csv(output_root / "global_pareto.csv", index=False)
    pd.DataFrame(cache.report(protocol)).to_csv(output_root / "molecule_cache.csv", index=False)
    front.to_csv(run_dir / "global_pareto.csv", index=False)
    return archive, front


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def update_summary(path: Path, summary: dict[str, Any]) -> None:
    write_json(path, summary)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.iterations < 0:
        parser.error("--iterations must be non-negative")
    if args.samples_per_mask < 1 or args.initial_mask_count < 1 or args.max_pareto_parents < 1 or args.max_generation_batches < 1:
        parser.error("samples per mask, initial mask count, max Pareto parents, and max generation batches must be positive")
    if args.total_candidate_budget < args.iterations + 1:
        parser.error("total candidate budget must allow at least one candidate per requested round")
    if args.min_mask_tokens < 1 or args.max_mask_tokens < args.min_mask_tokens:
        parser.error("mask bounds are invalid")
    if not 0 < args.candidate_decay < 1:
        parser.error("--candidate-decay must be greater than 0 and less than 1")
    if args.vina_exhaustiveness < 1 or args.vina_poses < 1:
        parser.error("Vina exhaustiveness and poses must be positive")
    candidate_budgets = candidate_budget_schedule(
        args.total_candidate_budget, args.iterations + 1, args.candidate_decay
    )
    if args.output_dir is None:
        from datetime import datetime
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = OUTPUT_ROOT / f"caffeine_seeded_{stamp}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.output_dir.name
    cache = MoleculeCache(args.cache_db)
    try:
        requested = legacy.resolve_requested_members()
        objectives, _, property_status = legacy.check_property_services(args, requested, [])
        vina_status = legacy.check_vina(args)
        protocol = build_protocol(args, objectives)
        imported = import_previous_runs(cache, protocol, OUTPUT_ROOT, run_id, objectives, args)
        service_record = {
            "property_services": property_status,
            "vina": vina_status,
            "active_objectives": {
                objective: [member_name(member) for member in members]
                for objective, members in objectives.items()
            },
            "protocol": protocol,
            "historical_runs_imported": imported,
        }
        write_json(args.output_dir / "service_status.json", service_record)

        caffeine_key = canonicalize_smiles([args.caffeine_smiles], GENMOL_PYTHON)[0]
        cached_caffeine = cache.get(protocol, caffeine_key)
        if cached_caffeine and cached_caffeine.get("docking_score") is not None and not (
            args.retry_failed_cache and cached_caffeine["status"] == "docking_failed"
        ):
            caffeine = cached_caffeine
            caffeine_cache_hit = True
        else:
            with requests.Session() as session:
                caffeine = legacy.request_docking(session, args, args.caffeine_smiles, "caffeine reference")
            cache.store(protocol, caffeine_key, args.caffeine_smiles, **caffeine, status="docked")
            caffeine_cache_hit = False
        baseline = float(caffeine["docking_score"])
        write_json(args.output_dir / "caffeine_reference.json", {
            "smiles": args.caffeine_smiles,
            "docking_affinity": caffeine["docking_affinity"],
            "docking_score": baseline,
            "receptor": args.receptor,
            "cache_hit": caffeine_cache_hit,
            "filter": "candidate docking_score must be strictly greater than caffeine docking_score",
        })

        initial_generation_plan = plan_generation(
            candidate_budgets[0],
            1,
            min(args.initial_mask_count, args.max_mask_tokens - args.min_mask_tokens + 1),
            estimate_fragment_count(args.starting_smiles or args.caffeine_smiles),
            args.samples_per_mask,
        )
        initial_generation_args = generation_args_for_round(
            args, 0, int(initial_generation_plan["mask_count"])
        )
        summary: dict[str, Any] = {
            "status": "running",
            "run_id": run_id,
            "completed_rounds": [],
            "iterations_requested": args.iterations,
            "caffeine_docking_score": baseline,
            "candidate_filter": "docking_score > caffeine_docking_score",
            "candidate_decay": args.candidate_decay,
            "total_candidate_budget": args.total_candidate_budget,
            "round_candidate_budgets": candidate_budgets,
            "budget_reallocations": [],
            "max_pareto_parents": args.max_pareto_parents,
            "samples_per_mask_initial": initial_generation_plan["samples_per_mask"],
            "initial_mask_tokens": [initial_generation_args.min_mask_tokens, initial_generation_args.max_mask_tokens],
            "historical_runs_imported": imported,
            "rounds": [],
        }
        summary_path = args.output_dir / "workflow_summary.json"
        update_summary(summary_path, summary)

        initial_candidates_path = round_file(args.output_dir, 0, "candidates.csv")
        if args.resume and initial_candidates_path.is_file():
            candidates = pd.read_csv(initial_candidates_path)
            print(f"Resuming from {initial_candidates_path}: {len(candidates)} saved candidates.", flush=True)
        else:
            first_generated, initial_batches, initial_generated_raw = generate_to_budget(
                [args.starting_smiles] if args.starting_smiles else [None],
                initial_generation_args,
                args.output_dir,
                0,
                args.iterations + 1,
                int(initial_generation_plan["samples_per_mask"]),
                candidate_budgets[0],
                set(),
                args.seed,
                args.max_generation_batches,
            )
            unique_first = first_generated
            if not unique_first:
                raise RuntimeError("Initial GenMol run produced no valid molecules")
            selected_first = select_pool(
                list(unique_first.values()), candidate_budgets[0], args.seed
            )
            write_json(round_dir(args.output_dir, 0) / "round_00_generation_summary.json", {
                "candidate_budget": candidate_budgets[0],
                "parents_used": 1,
                "mask_tokens_used": [initial_generation_args.min_mask_tokens, initial_generation_args.max_mask_tokens],
                "samples_per_mask": initial_generation_plan["samples_per_mask"],
                "estimated_valid_candidates": initial_generation_plan["expected_valid_candidates"],
                "generation_batches": initial_batches,
                "generated_raw": initial_generated_raw,
                "unique_generated": len(unique_first),
                "selected_for_docking": len(selected_first),
            })
            candidates = pd.DataFrame([
                {"candidate_id": f"R00_{index:05d}", "preferred_name": f"generated_0_{index}", "smiles": smiles, "canonical_smiles": canonical}
                for index, smiles in enumerate(selected_first)
                for canonical in [smiles]
            ])
            initial_candidates_path.parent.mkdir(parents=True, exist_ok=True)
            candidates.to_csv(initial_candidates_path, index=False)
        run_smiles_seen = set(candidates["canonical_smiles"].astype(str))

        cache_hits = {"docking": 0, "properties": 0, "failed_skipped": 0}
        for round_index in range(args.iterations + 1):
            current_round_dir = round_dir(args.output_dir, round_index)
            if round_index > 0:
                candidates = pd.read_csv(round_file(args.output_dir, round_index, "candidates.csv"))
            candidate_budgets, shortfall = reallocate_budget_shortfall(
                candidate_budgets, round_index, len(candidates), args.candidate_decay
            )
            if shortfall:
                summary["round_candidate_budgets"] = candidate_budgets
                summary["budget_reallocations"].append({
                    "from_round": round_index,
                    "unused_budget": shortfall,
                    "reassigned_to_rounds": candidate_budgets[round_index + 1:],
                })
                update_summary(summary_path, summary)
                print(
                    f"Round {round_index}: carrying {shortfall} unfilled unique-candidate slots into later rounds.",
                    flush=True,
                )
            dock_rows = []
            with requests.Session() as docking_session:
                for position, (_, candidate) in enumerate(candidates.iterrows(), 1):
                    row = candidate.to_dict()
                    canonical = str(row["canonical_smiles"])
                    cached = cache.get(protocol, canonical)
                    retry_failed = args.retry_failed_cache and cached and cached["status"] == "docking_failed"
                    if cached and cached.get("docking_score") is not None and not retry_failed:
                        entry = cached
                        cache_hits["docking"] += 1
                        row = cache_row(entry, row, baseline)
                        row["docking_status"] = "cache_hit"
                    elif cached and cached["status"] == "docking_failed" and not args.retry_failed_cache:
                        row.update({"canonical_smiles": canonical, "caffeine_docking_score": baseline, "beats_caffeine": False, "docking_status": "cached_failure", "error": cached.get("error"), "cache_hit": True})
                        cache_hits["failed_skipped"] += 1
                    else:
                        try:
                            result = legacy.request_docking(docking_session, args, str(row["smiles"]), str(row.get("preferred_name", canonical)))
                            row.update(result)
                            row.update({"canonical_smiles": canonical, "caffeine_docking_score": baseline, "beats_caffeine": result["docking_score"] > baseline, "docking_status": "docked", "cache_hit": False})
                            cache.store(protocol, canonical, str(row["smiles"]), **result, status="docked")
                        except Exception as exc:
                            row.update({"canonical_smiles": canonical, "caffeine_docking_score": baseline, "beats_caffeine": False, "docking_status": "docking_failed", "error": str(exc), "cache_hit": False})
                            cache.store(protocol, canonical, str(row["smiles"]), status="docking_failed", error=str(exc))
                            print(f"WARNING: docking failed for {row.get('preferred_name', canonical)}: {exc}", file=sys.stderr, flush=True)
                    cache.mark_candidate(protocol, run_id, round_index, canonical, str(row["docking_status"]))
                    dock_rows.append(row)
                    if position % 10 == 0:
                        snapshot = pd.DataFrame(dock_rows)
                        snapshot.to_csv(current_round_dir / f"round_{round_index:02d}_docked_partial.csv", index=False)
                        snapshot[snapshot["beats_caffeine"] == True].to_csv(current_round_dir / f"round_{round_index:02d}_docking_passed_partial.csv", index=False)
                    print(f"Round {round_index}: docking {position}/{len(candidates)}", flush=True)

            docked = pd.DataFrame(dock_rows)
            passed = docked[docked["beats_caffeine"] == True].copy()
            docked.to_csv(current_round_dir / f"round_{round_index:02d}_docked.csv", index=False)
            passed.to_csv(current_round_dir / f"round_{round_index:02d}_docking_passed.csv", index=False)
            for suffix in ("docked_partial", "docking_passed_partial"):
                (current_round_dir / f"round_{round_index:02d}_{suffix}.csv").unlink(missing_ok=True)

            needs_properties = []
            scored_rows = []
            property_failures = 0
            for _, candidate in passed.iterrows():
                row = candidate.to_dict()
                canonical = str(row["canonical_smiles"])
                entry = cache.get(protocol, canonical)
                if entry and entry.get("status") == "scored" and not (args.retry_failed_cache and entry.get("error")):
                    row.update({
                        "lipophilicity_score": entry["lipophilicity_score"],
                        "solubility_score": entry["solubility_score"],
                        "cache_status": "scored",
                        "cache_hit": True,
                    })
                    scored_rows.append(row)
                    cache_hits["properties"] += 1
                    cache.mark_candidate(protocol, run_id, round_index, canonical, "scored_cache_hit")
                elif entry and entry.get("status") == "property_failed" and not args.retry_failed_cache:
                    cache_hits["failed_skipped"] += 1
                    property_failures += 1
                    cache.mark_candidate(protocol, run_id, round_index, canonical, "property_failed_cache_hit")
                else:
                    needs_properties.append(row)

            if needs_properties:
                fresh, errors = score_properties(pd.DataFrame(needs_properties), objectives, args)
                successful = set(fresh["canonical_smiles"].astype(str)) if not fresh.empty else set()
                for _, row in fresh.iterrows():
                    item = row.to_dict()
                    canonical = str(item["canonical_smiles"])
                    cache.store(
                        protocol, canonical, str(item["smiles"]),
                        docking_affinity=item.get("docking_affinity"),
                        docking_score=item.get("docking_score"),
                        lipophilicity_score=item["lipophilicity_score"],
                        solubility_score=item["solubility_score"],
                        status="scored",
                    )
                    cache.mark_candidate(protocol, run_id, round_index, canonical, "scored")
                    scored_rows.append(item)
                for row in needs_properties:
                    canonical = str(row["canonical_smiles"])
                    if canonical not in successful:
                        message = errors.get(canonical, "property prediction did not return a complete score")
                        cache.store(protocol, canonical, str(row["smiles"]), status="property_failed", error=message)
                        cache.mark_candidate(protocol, run_id, round_index, canonical, "property_failed")
                        property_failures += 1

            scored = pd.DataFrame(scored_rows)
            scored.to_csv(current_round_dir / f"round_{round_index:02d}_property_scored.csv", index=False)
            cumulative = pd.DataFrame(cache.run_scored(protocol, run_id))
            pareto = find_pareto_front(cumulative)
            pareto_keys = pareto["canonical_smiles"].astype(str).tolist() if not pareto.empty else []
            cache.mark_pareto(protocol, run_id, round_index, pareto_keys)
            pareto.to_csv(current_round_dir / f"round_{round_index:02d}_pareto.csv", index=False)
            archive, global_front = write_global_outputs(cache, protocol, OUTPUT_ROOT, args.output_dir)
            write_round_figures(current_round_dir, docked, scored, pareto, global_front, baseline, round_index)
            round_summary = {
                "round": round_index,
                "candidate_budget": candidate_budgets[round_index],
                "candidate_count": len(candidates),
                "docked_successfully": int(docked["docking_score"].notna().sum()),
                "docking_failures": int((docked["docking_status"] == "docking_failed").sum()),
                "beats_caffeine": len(passed),
                "property_scored": len(scored),
                "property_failures_or_cached_skips": property_failures,
                "cumulative_run_pareto_count": len(pareto),
                "global_archive_count": len(archive),
                "global_pareto_count": len(global_front),
                "docking_cache_hits_total": cache_hits["docking"],
                "property_cache_hits_total": cache_hits["properties"],
                "cache_failed_skips_total": cache_hits["failed_skipped"],
            }
            summary["rounds"].append(round_summary)
            summary["completed_rounds"].append(round_index)
            summary["current_round"] = round_index
            summary["status"] = "running"
            update_summary(summary_path, summary)
            print(
                f"Round {round_index}: {len(passed)}/{len(candidates)} beat caffeine; "
                f"{len(pareto)} cumulative Pareto; {len(global_front)} global Pareto.",
                flush=True,
            )

            if round_index == args.iterations or pareto.empty:
                break
            next_index = round_index + 1
            next_candidates_path = round_dir(args.output_dir, next_index) / f"round_{next_index:02d}_candidates.csv"
            if args.resume and next_candidates_path.is_file():
                candidates = pd.read_csv(next_candidates_path)
                run_smiles_seen.update(candidates["canonical_smiles"].astype(str))
                print(
                    f"Resuming round {next_index} from {next_candidates_path}: "
                    f"{len(candidates)} saved candidates.",
                    flush=True,
                )
                continue
            budget = candidate_budgets[round_index + 1]
            parent_limit = min(args.max_pareto_parents, len(pareto), budget)
            parents = select_parents(pareto, parent_limit)
            expected_fragments = max(
                1,
                round(sum(estimate_fragment_count(smiles) for smiles in parents) / len(parents)),
            )
            generation_plan = plan_generation(
                budget,
                len(parents),
                args.max_mask_tokens - args.min_mask_tokens + 1,
                expected_fragments,
            )
            samples_this_round = int(generation_plan["samples_per_mask"])
            generation_args = generation_args_for_round(
                args, round_index + 1, int(generation_plan["mask_count"])
            )
            unique_generated, generation_batches, generated_raw = generate_to_budget(
                parents,
                generation_args,
                args.output_dir,
                round_index + 1,
                args.iterations + 1,
                samples_this_round,
                budget,
                run_smiles_seen,
                args.seed + (round_index + 1) * 100000,
                args.max_generation_batches,
            )
            selected = select_pool(list(unique_generated.values()), budget, args.seed + round_index + 1)
            write_json(round_dir(args.output_dir, round_index + 1) / f"round_{round_index + 1:02d}_generation_summary.json", {
                "parents_in_global_run_front": len(pareto),
                "parents_used": len(parents),
                "mask_tokens_used": [generation_args.min_mask_tokens, generation_args.max_mask_tokens],
                "mask_conditions_per_parent": generation_args.max_mask_tokens - generation_args.min_mask_tokens + 1,
                "samples_per_mask": samples_this_round,
                "estimated_valid_candidates": generation_plan["expected_valid_candidates"],
                "generation_batches": generation_batches,
                "generated_raw": generated_raw,
                "unique_generated": len(unique_generated),
                "candidate_budget": budget,
                "selected_for_docking": len(selected),
                "cached_molecules_reused": sum(cache.get(protocol, key) is not None for key in selected),
                "candidate_decay": args.candidate_decay,
                "duplicate_generated_skipped": generated_raw - len(unique_generated),
            })
            if not selected:
                summary["status"] = "completed_no_novel_candidates"
                update_summary(summary_path, summary)
                print("No new unique candidates remain for this run; stopping.", file=sys.stderr, flush=True)
                break
            candidates = pd.DataFrame([
                {"candidate_id": f"R{next_index:02d}_{index:05d}", "preferred_name": f"generated_{next_index}_{index}", "smiles": smiles, "canonical_smiles": smiles}
                for index, smiles in enumerate(selected)
            ])
            candidates.to_csv(next_candidates_path, index=False)
            run_smiles_seen.update(selected)
        else:
            summary["status"] = "completed"

        summary["status"] = "completed" if summary["status"] == "running" else summary["status"]
        summary["global_archive_count"] = len(cache.global_scored(protocol))
        _, final_global = write_global_outputs(cache, protocol, OUTPUT_ROOT, args.output_dir)
        summary["final_global_pareto_count"] = len(final_global)
        summary["cache_hits"] = cache_hits
        update_summary(summary_path, summary)
        write_run_summary_figures(args.output_dir, summary)
    finally:
        cache.close()


if __name__ == "__main__":
    try:
        main()
    except (legacy.PredictionFailure, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
