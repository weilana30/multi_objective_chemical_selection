# Iterative dock-first Pareto runs

- `caffeine_seeded_20261007_2055/` — primary caffeine-seeded exploration. User-stopped during docking in round 2; see `RUN_STATUS.md` for completed rounds, partial work, and score review.
- `caffeine_seeded_20261007_2049/` — small pilot with one sample at mask size 5. It produced two candidates and neither exceeded caffeine's docking score.

The primary run uses all candidates with docking scores strictly greater than the caffeine reference; there is no top-N cap.
