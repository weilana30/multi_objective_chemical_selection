# Shared generator adapter

The CReM, MolMIM, and REINVENT4 Mol2Mol launchers replace only the candidate-generation call in the existing iterative dock-first Pareto workflow. They retain its caffeine reference, docking cutoff, score cache, property scoring, cumulative round fronts, and global front calculation. Each generator has its own global archive/cache directory under `outputs/`.
