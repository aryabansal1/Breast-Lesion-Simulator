# Verification vs. validation

**Numerical verification** (does the code solve the equations correctly?) — automated in `tests/`:

| Check | Test | Result (this repo) |
|---|---|---|
| Homogeneous bar vs. analytic \(Z=L/(\gamma A)+2z_c/A\) | `test_homogeneous_bar_matches_analytic` | rel. error ~1e-14 |
| Current conservation, reference \(\sum U=0\) | `test_current_conservation_and_reference` | ~1e-14 |
| Linearity in injected current | `test_linear_scaling_with_current` | exact to 1e-10 |
| Reciprocity \(Z_{AB,CD}=Z_{CD,AB}\) (random heterogeneous γ) | `test_reciprocity` | < 1e-10 |
| Mesh convergence | `test_mesh_convergence_point_electrodes`, `scripts/benchmark.py` | monotone |
| Mesh convergence, dense VICTRE paired case | `scripts/convergence.py` → `examples/victre_convergence/` | tumour effect ≤ 0.3 pp vs 1.5 mm; absolute \|Z\| ≤ ~11 % |
| Zero-contrast tumour → no change | `test_zero_contrast_tumour_gives_no_change` | < 1e-12 |
| Determinism (repeat runs bitwise identical) | `test_tumour_changes_..._deterministic` | identical |
| Bilateral mirror symmetry without tumour | `test_bilateral_mirror_is_symmetric_without_tumour` | < 1e-6 |
| Per-solve residual and current balance | every run → `solver_diagnostics.csv` | ~1e-13 |

**Physical validation** (does the model predict real measurements?) has **not** been done. Proposed procedure:

## Stage 1 — device electronics against known loads
Purpose: verify the wearable's current source, demodulation and phase, independent of anatomy.
1. Resistors (e.g. 10, 100, 1 k Ω, 0.1 % tolerance) and RC networks (e.g. R ∥ C with corner frequencies inside
   the sweep), plus a 2R-1C Cole-like network.
2. Measure at the same frequencies, current amplitude and electrode switching as the simulation.
3. Compare magnitude/phase to the analytic network impedance; report error vs. frequency.
4. Also test with series "contact" resistors/capacitors on each lead to quantify contact-impedance sensitivity.

Known loads validate electronics only — they say nothing about current paths in a breast.

## Stage 2 — tissue-mimicking phantom (simulator + device together)
1. Build a phantom with known geometry (e.g. hemisphere of agar/gelatine + NaCl for gland-like conductivity,
   with an oil-in-gelatine or lower-salt layer for fat, and an inclusion of higher salt for the tumour).
2. Characterise each material separately with a four-electrode cell of known cell constant at the device
   frequencies and the **test temperature** (conductivity drifts ~2 %/°C).
3. Reproduce the phantom geometry in the simulator (`anatomy.type: synthetic_hemisphere` or a voxel phantom),
   same electrode sizes/positions (measure them), same contact medium (measure \(z_c\) with a two-size electrode
   test), same current and frequencies.
4. Measure inclusion-absent and inclusion-present, ideally with the inclusion moved between positions.
5. Compare absolute \(Z_{transfer}\) and especially the *differences*; report residuals with uncertainty bars
   from material characterisation and electrode-position error (run `breastsim sensitivity`).
6. Repeat on a second day for repeatability.

## Stage 3 — human data (only with appropriate ethics approval)
Compare distributions, not individual values; the anatomy and properties of any one subject are unknown.

## Uncertainty / sensitivity
`configs/victre_dense_sensitivity.yaml` varies tumour material source, fat/gland/skin conductivity (×0.5, ×2),
contact impedance, tumour depth/size and electrode position. Report simulated quantities as ranges under these
inputs, not as single exact values.
