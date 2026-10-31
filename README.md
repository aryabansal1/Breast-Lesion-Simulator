# Breast Lesion Simulator (`breastsim`)

Physics-based 3D simulator of the **complex transfer impedance** a wearable breast-bioimpedance device would
record through surface electrodes, with and without an inserted tumour.

- 3D frequency-domain electroquasistatic FEM, \(\nabla\cdot[(\sigma+j\omega\varepsilon)\nabla V]=0\), P1 tetrahedra,
  **complete electrode model** (finite electrodes, contact impedance, current injection, voltage sensing).
- Uncompressed, tissue-labelled anatomy from DIDSR breastPhantom / VICTRE (`_p_` phantoms), downsampled with labels,
  spacing and offsets preserved; explicit (assumed) chest-wall block.
- Frequency-dependent, source-traceable tissue properties (IT'IS V5.0, UCI/Jossinet-derived Cole fits, Surowiec 1988,
  Halter 2009) with extrapolation flags. See [`docs/materials.md`](docs/materials.md).
- Tumours: sphere, ellipsoid, voxel mask (e.g. VICTRE lesion models) or STL; placed by anatomical clock position and
  depth; rejected if they touch air, the boundary or an electrode.
- Paired tumour-absent / tumour-present runs on identical mesh, electrodes, contact and excitation; optional
  bilateral (mirrored, flagged as an assumption) runs.
- CSV / JSON / YAML outputs, VTU fields (potential, |E|, |J|, labels, tumour, electrodes), spectra plots, renders.
- CPU only (SciPy SuperLU + METIS ordering); runs on an Apple-Silicon laptop.

**Not included (by design):** machine learning, classification, diagnostic thresholds, temperature, mammography/DBT.
The simulator does not establish clinical detection performance; see
[`docs/assumptions_and_limitations.md`](docs/assumptions_and_limitations.md).

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[viz,stl,data,dev]"      # viz=pyvista renders, stl=trimesh tumours, data=UCI xls
pytest -q                                  # numerical verification suite (~15 s)
```
macOS (Apple Silicon): the same commands work with the system/Homebrew Python ≥ 3.10; all deps have arm64 wheels.

## Run

```bash
breastsim run configs/synthetic_quick.yaml --plot          # ~10 s smoke test
breastsim run configs/victre_dense_paired.yaml --plot      # realistic anatomy, 8 electrodes, 7 freqs (~2 min)
breastsim run configs/victre_dense_bilateral.yaml          # mirrored left/right
breastsim sensitivity configs/victre_dense_sensitivity.yaml
breastsim plot outputs/victre_dense_paired --render
```
Run one heavy job at a time: concurrent runs oversubscribe BLAS threads and slow down sharply.

Outputs (in `outputs/<name>/`):

| File | Content |
|---|---|
| `measurements.csv` | per side / case / frequency / measurement: injected current, electrode ids, V, Z (re, im, \|Z\|, phase) |
| `tumour_difference.csv` | Z_present − Z_absent (complex, magnitude, relative, phase) |
| `left_right_difference.csv` | bilateral runs |
| `solver_diagnostics.csv` | residuals, current conservation, factor/solve time, LU fill, DOFs |
| `metadata.json` | resolved config, mesh stats, electrode areas, tumour report, material table per frequency with extrapolation flags, library SHA-256, git commit, timing, peak RSS |
| `fields_<side>_<case>.vtu`, `electrodes_<side>.vtu` | open in ParaView / PyVista |
| `spectra_*.png`, `tumour_effect_*.png`, `render_*.png` | plots |

Example outputs from the realistic run are in [`examples/victre_dense_paired/`](examples/victre_dense_paired).

## Configuration (YAML)

Everything is configurable: anatomy file and side, mesh size / local refinement / chest-wall thickness, material
assignment per label and overrides / scaling, extrapolation policy, electrodes (count, clock position, axial
fraction `s` from nipple (0) to base (1), radius, contact model `constant` or `parallel_rc`), measurement pairs or
`adjacent` pattern, current, frequencies, tumour shape/size/orientation/position/material. Electrode positions are
anatomical (clock hour + `s`), so the same layout transfers across breast sizes and mesh resolutions; no final
wearable layout is hard-coded.

## Repository layout

```
src/breastsim/   anatomy.py mesh.py electrodes.py tumour.py materials.py fem.py simulate.py io.py plotting.py cli.py
data/anatomy/    downsampled uncompressed VICTRE phantoms (CC0)
data/materials/  library.yaml (+ source tables / fits)
scripts/         preprocess_victre.py fit_uci_cole.py extract_itis.py build_library.py benchmark.py
configs/  tests/  docs/  examples/  third_party/ (breastPhantom VTK 9 patch)
```

## Rebuilding inputs

```bash
python scripts/preprocess_victre.py <VICTRE>/Sample-phantom-data/dense_p_*.mhd <raw> data/anatomy/x.npz --factor 10
python scripts/fit_uci_cole.py   # needs BreastTissue.xls
python scripts/extract_itis.py   # needs IT'IS V5.0 download
python scripts/build_library.py
python scripts/benchmark.py
```

## Docs
[data inventory](docs/data_inventory.md) · [materials](docs/materials.md) ·
[assumptions & missing data](docs/assumptions_and_limitations.md) ·
[verification & validation procedure](docs/validation_procedure.md) · [benchmarks](docs/benchmarks.md)

## Licences of inputs
VICTRE / breastPhantom: CC0 (US FDA). IT'IS database values: © IT'IS Foundation, cite DOI 10.13099/VIP21000-05-0.
UCI Breast Tissue: CC BY 4.0. Literature values are cited per entry.
