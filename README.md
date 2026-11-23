# Multi-Objective Chemical Selection

An iterative molecular discovery pipeline that generates candidate molecules, docks them, scores the survivors on competing physicochemical objectives, and feeds the best trade-offs back into the next round of generation.

Generative models can propose large numbers of drug-like molecules, but a useful candidate has to satisfy several properties at once. Optimizing a single weighted score hides those trade-offs. This pipeline keeps them explicit by maintaining a Pareto front: the set of molecules that no other molecule beats on every objective at the same time.

## How a run works

Each round executes the same loop:

1. **Generate** candidate molecules, optionally seeded from a starting SMILES.
2. **Dock** every unique candidate with AutoDock Vina.
3. **Filter** to candidates that score strictly better than a fixed caffeine reference, recomputed at the start of each run with the same receptor and Vina settings.
4. **Score** the survivors on aggregate lipophilicity and solubility.
5. **Recompute the Pareto front** across every molecule scored so far in the run, so a molecule can drop off the front when a later candidate dominates it.
6. **Remask** the front members as parents for the next round.

## Generators

Four generators are supported behind a shared adapter, so docking, filtering, scoring, caching, and Pareto logic stay identical across them:

| Generator | Workflow |
| --- | --- |
| NVIDIA GenMol | `scripts/iterative_dock_pareto/` |
| CReM | `scripts/crem_dock_pareto/` |
| NVIDIA MolMIM | `scripts/molmim_dock_pareto/` |
| REINVENT4 Mol2Mol | `scripts/reinvent4_mol2mol_dock_pareto/` |

Shared logic lives in `scripts/generator_adapters/common.py`.

## Dashboard

A Dash application (`app/`) runs locally at `127.0.0.1:8050`. It configures and launches a run, checks that the required prediction services are up, and then shows live progress, per-round results, the cumulative Pareto front and its evolution, 2D structures, and interactive 3D views of the best docked poses. Stopping a run preserves the rounds already finished.

## Engineering for long runs

- **Candidate budget planner** - spreads a fixed docking budget across rounds in geometrically decaying batches, choosing parent count, mask sizes, and samples per mask.
- **Persistent SQLite cache** - records canonical SMILES, scores, failures, and Pareto history so molecules are never re-docked.
- **Resumable runs** - `--resume` continues an interrupted run against its existing output directory.
- **Deduplication** by canonical SMILES across the whole run.

## Requirements

This repository contains the pipeline only. It expects prediction services to be running and reachable on localhost:

- AutoDock Vina for docking
- Property prediction services for lipophilicity, solubility, and related screening properties

Nothing here starts those services; the dashboard checks for them and names any that are missing. Python 3.10 is the validated version.

`property_output_registry.json` records which model outputs may enter the two optimization objectives and which are treated as constraints instead, along with the caveats that apply to each: classifier values are positive-class probabilities, a 0.5 cutoff is a screening default rather than a safety guarantee, and raw values must be normalized within a model before being combined.

## Repository layout

| Path | Contents |
| --- | --- |
| `scripts/` | Pareto workflows, one directory per generator, plus the shared adapter |
| `app/` | Dash dashboard: configuration, run management, reports, figures |
| `outputs/` | Run reports: settings, logs, per-round CSVs, figures |
| `pareto_results/` | Earlier standalone Pareto runs |
| `assets/` | Workflow diagrams and generator assets |
| `aggregate_pareto.py`, `compare_pareto_rounds.py` | Analysis helpers |

`scripts/iterative_pareto/`, `scripts/iterative_pareto_docking/`, and `scripts/live_pareto/` are earlier versions, kept because the current workflow still imports constants from them.
