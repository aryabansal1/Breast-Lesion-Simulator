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
(field export included). The 24 GB Mac budget allows ~1 mm global meshes (~0.5 M nodes) only with an
iterative solver; stay ≤ 1.5 mm global with local refinement for direct solves.

Solver note: SciPy's default COLAMD ordering took 215 s for the 10k-node system (fill-in explosion);
the METIS ordering is required (`pymetis` dependency, with an MMD fallback).

## Mesh convergence on the dense VICTRE breast

`python scripts/convergence.py configs/victre_dense_paired.yaml -o outputs/convergence` runs the full paired
case (bilateral, 7 frequencies, 44 measurements per side → 308 complex \(Z\) values per case) at each mesh
level, one subprocess at a time. Results: `examples/victre_convergence/` (`convergence_summary.csv`,
per-measurement `convergence.csv`, `convergence.png`). Differences are against the finest global level
(h = 1.5 mm, tumour 0.75 mm). That is a reference, not the exact solution; nodes/tets are both breasts.

| sweep | h (mm) | tumour h (mm) | nodes | tets | runtime (s) | peak RSS (MB) | \|Z\| absent: median / max diff (%) | tumour effect on \|Z\|: median / max diff (pp) | phase effect max diff (°) |
|---|---|---|---|---|---|---|---|---|---|
| global | 3.0 | 1.5 | 24,327 | 129,276 | 32 | 829 | 3.4 / 11.6 | 0.12 / 0.64 | 0.030 |
| global | 2.5 | 1.25 | 38,584 | 208,560 | 81 | 1,297 | 3.3 / 11.7 | 0.05 / 0.27 | 0.020 |
| global (**default**) | 2.0 | 1.0 | 70,423 | 388,170 | 170 | 2,629 | 3.6 / 10.6 | 0.04 / 0.30 | 0.044 |
| global (reference) | 1.5 | 0.75 | 156,636 | 880,860 | 598 | 7,151 | – | – | – |
| tumour | 2.0 | 2.0 | 33,121 | 177,504 | 50 | 1,002 | 5.7 / 23.0 | 0.15 / 0.92 | 0.083 |
| tumour | 2.0 | 0.5 | 212,492 | 1,205,166 | 1,428 | 12,425 | 2.3 / 11.2 | 0.04 / 0.32 | 0.038 |

Largest tumour effect at the reference level: 2.79 % on \|Z\|. Example, `across_vertical`, left, 62.5 kHz:
\|Z\| = 32.8 / 33.1 / 34.4 / 33.3 Ω and tumour effect = 0.78 / 0.67 / 0.77 / 0.65 % for h = 3 / 2.5 / 2 / 1.5 mm.

Interpretation:
- **Tumour effect (with − without on the same mesh) is mesh-stable.** At the default 2 mm / 1 mm mesh it is
  within 0.04 pp (median) and 0.30 pp (worst of 308) of the 1.5 mm reference. Phase effects agree to < 0.05°.
  Refining only the tumour region from 1 mm to 0.5 mm moves the effect by ≤ 0.20 pp (median 0.015 pp), so the
  1 mm tumour refinement is converged. Without tumour refinement (2 mm / 2 mm) the error grows to 0.9 pp, so
  keep `mesh.tumour_h_mm` ≤ 1.
- **Absolute \|Z\| is not mesh-converged to better than ~±10 %.** The change is not monotone in h (see the
  left panel of `convergence.png`). Changing only the tumour refinement (which moves grid planes across the
  whole tensor-product mesh) also changes absolute \|Z\| by up to 14 %. The largest changes are in adjacent
  and `upper_outer` electrode pairs (6–11 %), and the smallest are in pairs across the breast (< 1 %). That
  pattern points to the stair-stepped skin and electrode-patch geometry being re-sampled at each mesh size,
  not to the breast interior.
- **Practical rule.** Use the simulator for paired tumour-induced changes. Read the absolute transfer impedance
  as having roughly ±10 % discretisation uncertainty (adjacent pairs), on top of the material and contact
  impedance uncertainty. A conforming smooth-surface mesh (skin and electrodes) would be the fix if absolute
  \|Z\| matters, for example when matching device readings against known loads.
- Memory: the 2 mm / 0.5 mm level peaked at 12.4 GB, and h = 1.5 mm at 7.2 GB. Both fit a 24 GB Mac, but run
  them one at a time.
