# Caffeine-seeded iterative run status

**Status:** Stopped at the user's request during round 2. The first two Pareto rounds completed. Round 2 generated its candidate set, but only part of its docking batch completed. The script writes docking CSVs after a full batch, so partial round 2 docking results were not saved as a CSV.

## Verification

- Python compilation and CLI startup passed.
- Synthetic two-round integration check passed: strict caffeine cutoff, no top-N cap, property score aggregation, Pareto filtering, and Pareto-parent remasking.
- Real preflight passed for all 13 selected objective predictions across six property services. Vina was healthy and the A2A_3RFM receptor was available.
- Real caffeine reference docking: affinity `-5.347 kcal/mol`; docking score `5.347`.
- Admetica is mapped to host port `8000` in this run; the inherited `18000` default was not the container's published port.

## Completed rounds

| Round | Generated | Docked successfully | Better than caffeine | Property-scored | Pareto |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 70 | 70 | 54 | 54 | 1 |
| 1 | 129 | 129 | 112 | 112 | 2 |
| 2 | 304 | Partial: 91 calls reached before stop | Not complete | Not run | Not run |

The comparison is strict: candidate `docking_score` must be greater than `5.347`. Some round 2 candidates were skipped because RDKit could not generate a 3-D conformer.

## Pareto results

Round 0 retained one zwitterionic caffeine analogue with docking score `6.318`, aggregate lipophilicity score `-24.324`, and aggregate solubility score `18.661`.

Round 1 retained two non-dominated candidates:

- `generated_1_84`: docking `6.815`, lipophilicity `-29.554`, solubility `18.185`.
- `generated_1_126`: docking `5.853`, lipophilicity `-26.452`, solubility `22.743`.

Within the workflow's predicted scores, these show the intended trade-off: one candidate has the lower lipophilicity score, while the other has the higher solubility score. Both docked better than caffeine. These are model predictions and docking scores, not experimental validation. The aggregate scores combine multiple model outputs, so interpret comparisons within this workflow rather than as calibrated physical units.

Round-level candidate, docking, scored, Pareto, and GenMol files are in this folder. The caffeine reference and service preflight are recorded in `caffeine_reference.json` and `service_status.json`.
