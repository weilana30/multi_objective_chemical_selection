# Iterative three-objective workflow

`iterative_pareto_docking.py` connects GenMol to the working property
microservices and the AutoDock Vina microservice. It optimizes these three
aggregate objectives, all treated as “higher is better”:

1. lipophilicity score
2. solubility score
3. docking score

The property member lists are the same lists used by `iterative_pareto.py`.
At startup, each requested member is tested with the probe SMILES. A service
member that is unavailable or returns an invalid prediction is excluded from
the run and recorded in `service_status.json`. This catches services that
report healthy but do not actually expose a requested model. Use
`--require-all-services` if a run should fail instead.

GenMol uses the default per-round schedule `10,5,2,1`: ten samples per
fragment in round 0, five in round 1, two in round 2, and one in round 3.
The workflow stops after round 3 by default. With `--min-mask-tokens 5` and
`--max-mask-tokens 5`, a source molecule with five fragments requests 50
GenMol samples in round 0. Invalid or duplicate generated molecules can make
the retained candidate count smaller than that request.

The Vina service returns a negative `docking_affinity` in kcal/mol and stores
`docking_score = -docking_affinity` as the maximized objective. The workflow
keeps both columns in its output. Solubility uses the registry transforms and
weights, including the negated, half-weighted hydration free-energy members.

By default, candidates must also pass early screening gates before they can
enter the Pareto calculation. These are molecular weight <=500 Da, HBA <=10,
HBD <=5, TPSA <=140 Angstrom^2, all four Lipinski rules satisfied, QED >=0.5,
and rotatable bonds <=10. The two direct solubility predictions must be at
least `--min-solubility-log-s` (default -4.0), and the Vina docking score must
be at least `--min-docking-score` (default 5.0, equivalent to affinity <=-5.0
kcal/mol). The rotatable-bond gate is skipped automatically if the MolProperty
service does not support that property. Use `--no-screening-constraints` to
disable these gates.

## Required running services

Start the property services on their ATLAS ports and the Vina service on port
18025. The workflow defaults are:

| Service | Port |
| --- | ---: |
| Admetica | 18000 |
| ADMEThyst | 18001 |
| BoiledEgg | 18002 |
| MolProperty | 18003 |
| ADMET-AI | 18019 |
| ClassicalGSG | 10001 |
| AutoDock Vina | 18025 |

The workflow does not start containers itself. It probes the services and
skips broken property members. The Vina service must be available because
docking is required for every candidate.

The default parent is `pareto_results/caffeine_docking`. `--output-dir` is a
parent directory, not a file-like run destination. Every
invocation creates a new child directory labeled with the starting molecule,
mask-token range, and timestamp. For example:

```text
pareto_results/caffeine_docking/caffeine_5_to_10_20261001_143015/
```

If two runs start in the same second, the later run receives a `_run_02`
suffix instead of overwriting the first run.

During a run, the terminal shows live progress bars for service preflight,
GenMol batch generation, and candidate scoring. GenMol progress is reported
as the current round, Pareto source, fragment, and mask length.

## Small test run

Run the script with the GenMol environment, which contains the Python
dependencies for the workflow:

```bash
  ../genmol_molecule_generator/.conda-env/bin/python \
  iterative_pareto_docking.py \
  --starting-smiles 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C' \
  --min-mask-tokens 5 \
  --max-mask-tokens 5 \
  --output-dir pareto_results/iterative_docking_test
```

The command runs the full `10,5,2,1` schedule. To stop earlier, add for
example `--iterations 1`, which runs rounds 0 and 1. To use a different
schedule, pass `--samples-schedule`, such as `--samples-schedule 4,2,1`.

## Outputs

Each scored round writes:

- `round_XX_screened.csv`: candidates with valid property predictions, including
  candidates rejected by a screening gate and their failure reasons. Candidates
  rejected before docking do not have docking values.
- `round_XX_scored.csv`: every candidate with all active member predictions,
  the three aggregate scores, docking affinity, and docking score that passed
  screening.
- `round_XX_pareto.csv`: only the non-dominated candidates, with the same
  score columns.
- `round_XX_pareto_pairwise.png`: a readable three-panel figure showing the
  pairwise objective projections, with Pareto-front points connected by a
  red line in each panel.

The directory also contains the GenMol JSON files, candidate CSVs,
`service_status.json`, and `workflow_summary.json`.

For a prediction-by-prediction inventory, including each transform, weight,
threshold, and whether it is used for scoring or screening, see
`ITERATIVE_PARETO_DOCKING_PREDICTIONS.txt`.
