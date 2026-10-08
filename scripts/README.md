# Pareto workflows

- `iterative_dock_pareto/` — current dock-first iterative workflow. It keeps every candidate scoring better than caffeine, scores survivors by solubility and lipophilicity, then remasks Pareto-front molecules.
- `crem_dock_pareto/` — new CReM analog-generator variant of the same iterative workflow; requires a fragment database.
- `molmim_dock_pareto/` — new NVIDIA MolMIM variant; requires a running local NIM service.
- `reinvent4_mol2mol_dock_pareto/` — new REINVENT4 Mol2Mol variant; requires its executable and pretrained prior.
- `generator_adapters/` — shared adapter keeping docking, filtering, scoring, caching, and Pareto logic common across the new variants.
- `iterative_pareto/` — legacy iterative two-objective GenMol workflow.
- `iterative_pareto_docking/` — legacy iterative three-objective GenMol plus docking workflow.
- `live_pareto/` — legacy live Pareto script and notebook.

Shared inputs, registry, aggregate helpers, and results remain at the project root.
