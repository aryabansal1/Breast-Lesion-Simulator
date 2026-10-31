# Benchmarks

`python scripts/benchmark.py` → `docs/benchmarks.csv`. One CEM system (8 electrodes, dense VICTRE phantom,
62.5 kHz), METIS nested-dissection ordering + SuperLU, 8 right-hand sides. Measured on the development VM
(Linux x86_64, 8 vCPU); an Apple-Silicon Mac should be similar or faster. **Re-run on your Mac and commit the CSV.**

| h (mm) | nodes | tets | factor (s) | solve 8 RHS (s) | LU nnz | peak RSS (MB) |
|---|---|---|---|---|---|---|
| 4.0 | 4,649 | 22,368 | 0.05 | 0.004 | 0.54 M | 135 |
| 3.0 | 10,636 | 54,030 | 0.26 | 0.02 | 2.9 M | 234 |
| 2.5 | 17,460 | 90,948 | 0.58 | 0.04 | 5.8 M | 336 |
| 2.0 | 33,121 | 177,504 | 1.76 | 0.09 | 13.0 M | 624 |
| 1.5 | 77,796 | 428,892 | 5.8 | 0.30 | 40.4 M | 1,726 |

Max relative residual in every case < 1e-12.

Full paired run `configs/victre_dense_paired.yaml` (2 mm + 1 mm refinement around the tumour → 70,423 nodes /
388,170 tets, 7 frequencies × 2 cases = 14 factorisations, 44 measurements): **~100 s, peak RSS 5.9 GB**
(field export included). Direct (LU) memory grows much faster than the mesh (17.9 GB at h = 1.25 mm), so for
global meshes finer than ~1.5 mm use the iterative solver (`solver: {method: iterative}`), whose memory grows
roughly linearly (1 mm: 4.1 GB, 0.75 mm: 8.8 GB for the full paired run).

Solver note: SciPy's default COLAMD ordering took 215 s for the 10k-node system (fill-in explosion);
the METIS ordering is required (`pymetis` dependency, with an MMD fallback).

## Mesh convergence on the dense VICTRE breast

`python scripts/convergence.py configs/victre_dense_paired.yaml -o outputs/convergence` runs the full paired
case (bilateral, 7 frequencies, 44 measurements per side → 308 complex \(Z\) values per case) at each mesh
level, one subprocess at a time. Results: `examples/victre_convergence/` (`convergence_summary.csv`,
per-measurement `convergence.csv`, `convergence.png`, `levels.csv`). Differences are against the finest global
level (h = 0.75 mm, tumour 0.375 mm, 1.1 M nodes). That is a reference, not the exact solution; nodes/tets are
both breasts. Levels ≥ 1.25 mm used the direct solver and finer ones the iterative solver; on the identical
2 mm case the two agree to 9e-9 relative in \(Z\) and 3e-7 pp in tumour effect.

| sweep | h (mm) | tumour h (mm) | solver | nodes | tets | runtime (s) | peak RSS (MB) | \|Z\| absent: median / max diff (%) | tumour effect on \|Z\|: median / max diff (pp) | phase effect max diff (°) |
|---|---|---|---|---|---|---|---|---|---|---|
| global | 3 | 1.5 | direct | 24,327 | 129,276 | 32 | 829 | 5.5 / 16.2 | 0.11 / 0.78 | 0.055 |
| global | 2.5 | 1.25 | direct | 38,584 | 208,560 | 81 | 1,297 | 5.9 / 18.6 | 0.05 / 0.41 | 0.046 |
| global (**default**) | 2 | 1 | direct | 70,423 | 388,170 | 170 | 2,629 | 3.7 / 11.7 | 0.06 / 0.44 | 0.056 |
| global | 1.5 | 0.75 | direct | 156,636 | 880,860 | 598 | 7,151 | 4.8 / 8.2 | 0.02 / 0.23 | 0.033 |
| global | 1.25 | 0.625 | direct | 256,314 | 1,455,600 | 1,848 | 17,860 | 2.4 / 7.9 | 0.03 / 0.20 | 0.054 |
| global | 1 | 0.5 | iterative | 484,786 | 2,782,140 | 1,177 | 4,101 | 1.7 / 4.8 | 0.06 / 0.50 | 0.037 |
| global (reference) | 0.75 | 0.375 | iterative | 1,097,944 | 6,361,782 | 2,699 | 8,845 | – | – | – |
| tumour | 2 | 2 | direct | 33,121 | 177,504 | 50 | 1,002 | 8.1 / 20.9 | 0.15 / 0.98 | 0.095 |
| tumour | 2 | 0.5 | direct | 212,492 | 1,205,166 | 1,428 | 12,430 | 4.4 / 11.7 | 0.06 / 0.46 | 0.050 |

Largest tumour effect at the reference level: 2.56 % on \|Z\|. Example, `across_vertical`, left, 62.5 kHz:
\|Z\| = 34.4 / 33.3 / 33.6 / 35.6 / 35.3 Ω and tumour effect = 0.77 / 0.65 / 0.58 / 0.71 / 0.58 % for
h = 2 / 1.5 / 1.25 / 1 / 0.75 mm.

Interpretation:
- **Tumour effect (with − without on the same mesh) reaches a noise floor by 2 mm.** Every level from 2.5 mm
  to 1 mm is within 0.02–0.06 pp (median) and 0.2–0.5 pp (worst of 308) of the 0.75 mm reference, and
  successive refinements keep moving it by that much without a trend. The default 2 mm / 1 mm mesh (0.06 /
  0.44 pp, 3 min, 2.6 GB) is as good as anything finer on this tensor-product mesh. Pairs across the breast
  vary by < 0.1 pp. Adjacent pairs whose sensing electrodes sit near the tumour jitter by up to ~0.5 pp,
  ~15–25 % of their 1.5–2.5 % effect. The meshed tumour volume matches the nominal sphere to < 0.1 %, so the
  jitter comes from re-sampling the stair-stepped skin and electrode patches at each mesh size, not from
  the tumour. Without tumour refinement (2 mm / 2 mm) the error grows to ~1 pp, so keep
  `mesh.tumour_h_mm` ≤ 1.
- **Absolute \|Z\| starts converging only below ~1.5 mm.** Differences from the reference shrink from
  ~4–6 % median at h ≥ 1.5 mm to 2.3 % at 1.25 mm and 1.7 % (max 4.8 %) at 1 mm. Successive 1 → 0.75 mm
  changes are 1.6 % median / 4.5 % max. The largest remaining changes are in adjacent, `upper_outer` and
  `bipolar_E1E5` pairs, and pairs across the breast are within ~1 %.
- **Practical granularity.** Tumour-effect studies: 2 mm / 1 mm (default). Absolute \|Z\|: 1 mm / 0.5 mm
  with the iterative solver (~20 min, ~4 GB per paired bilateral run) gives ~±2 % median and ~±5 % worst pair
  discretisation uncertainty. 0.75 mm (45 min, 8.8 GB) fits a 24 GB Mac but gains little. Going finer than
  0.75 mm is below the 0.5 mm anatomy grid, so it would need re-sampling the phantom. A conforming
  smooth-surface mesh (skin and electrodes) is the fix for the remaining near-electrode jitter.
- Memory: direct h = 1.25 mm peaked at 17.9 GB and 2 mm / 0.5 mm at 12.4 GB. Iterative h = 0.75 mm peaked at
  8.8 GB. All fit a 24 GB Mac if you run them one at a time.
