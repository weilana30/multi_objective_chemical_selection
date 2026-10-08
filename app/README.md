# Local molecule search dashboard

The dashboard runs on this Mac at <http://127.0.0.1:8050>. It starts a GenMol or CReM docking search, shows live progress and completed rounds, and opens old reports. The **total rounds** field includes round 0. The entered SMILES is both the starting molecule and docking reference. You can name each report; the name appears in the report picker and its output folder.

New runs open on **This run**, which stays empty until the first round finishes. Each finished round becomes a tab. The latest-round view shows the docking results, cumulative lipophilicity/solubility Pareto plot, Pareto front evolution within this run, and a fixed-reference coverage trend. Historical reports open individually; there is no cross-report global Pareto plot. The home screen highlights an active search and links straight to it. **Run settings** starts closed and can be opened to see the recorded starting molecule, docking reference, settings, receptor, and property models. Advanced settings are available before a new run for model selection, generation controls, docking settings, and service ports. Structure images in molecule popups are 2D RDKit drawings of the SMILES, not docked 3D poses.

## Setup

From the `multi_objective_chemical_selection` directory:

Use Python 3.10 (the version used for validation):

```bash
python3.10 -m venv .venv-app
.venv-app/bin/pip install -r app/requirements.txt
```

GenMol uses the existing `../genmol_molecule_generator/.conda-env/bin/python`. For CReM, create a persistent environment in this project using the same Python version and packages required by the workflow, including `crem`, RDKit, pandas, requests, matplotlib, and numpy. For example:

```bash
python3.10 -m venv .venv-crem
.venv-crem/bin/pip install -r app/requirements-crem.txt
```

The CReM fragment database defaults to `assets/crem/chembl33_sa2_f5.db`. It is a large local asset and is intentionally excluded from Git. To use a different workflow Python interpreter, set `GENMOL_PYTHON` or `CREM_PYTHON` before starting the app.

Start the prediction services (AutoDock Vina on port 18025 and the six property services named by preflight if missing), then run:

```bash
.venv-app/bin/python app/app.py
```

The app checks those services before each new run and names missing services. It does not start containers. One workflow can run at a time. The Stop button terminates it and preserves completed rounds.

## Saved data

New run reports live under `outputs/iterative_dock_pareto/dashboard_*` for GenMol and `outputs/crem_dock_pareto/dashboard_*` for CReM. Each has `run_settings.json`, `workflow_settings.json` (effective workflow options), `workflow.log`, atomic `progress.json`, and the workflow's per-round CSVs and figures. The dashboard also opens older flat-layout reports and lists a new report as soon as its settings file is saved.

Within a report, scored molecules are deduplicated by canonical SMILES and the displayed cumulative Pareto fronts are recomputed with both scores maximized. The coverage chart uses one fixed reference point for all completed rounds in that report, so its trend is comparable over time.
