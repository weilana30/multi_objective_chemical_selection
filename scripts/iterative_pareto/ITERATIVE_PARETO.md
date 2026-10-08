# Iterative Pareto workflow

Run five refinement rounds (the default):

```bash
python3 iterative_pareto.py
```

The workflow generates one de novo source molecule, remasks its generated variants,
scores them with the configured working services, and then remasks every Pareto
molecule in the next round. Use `-n` or `--iterations` to change the number of
refinement rounds:

```bash
python3 iterative_pareto.py --iterations 2 --output-dir pareto_results/run_02
```

Outputs include a candidate CSV, Pareto CSV, Pareto PNG, and remasking JSON for
each round. The driver uses Admet-AI (18019), Admetica (8000), ADMEThyst (18001),
BoiledEgg (18002), ClassicalGSG (10001), and MolProperty (18003). It excludes
LogPPrepML and ClassicalGSG CGenFF because those services are unavailable for
the SMILES-only workflow.

