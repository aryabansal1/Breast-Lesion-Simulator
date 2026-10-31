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

Mesh sensitivity of the realistic case (tumour effect, `across_vertical`, 62.5 kHz) should be checked
with `mesh.h_mm` ∈ {2.5, 2, 1.5} before quoting numbers; see `docs/validation_procedure.md`.
