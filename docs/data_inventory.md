# Input data inventory

| Input | Location / source | What it is | Status in the simulator |
|---|---|---|---|
| VICTRE sample phantoms (dense, fatty, hetero, scattered), `*_p_*.mhd/.raw.gz` | https://github.com/DIDSR/VICTRE (`Sample-phantom-data`), CC0 | **Uncompressed** breastPhantom label volumes, 0.05 mm voxels, uint8 tissue labels | Used. Block-majority downsampled to 0.5 mm by `scripts/preprocess_victre.py` → `data/anatomy/victre_*_p_0p5mm.npz` (header, spacing, offset, label list kept) |
| VICTRE `_pc_` / `_pcl_` files | same | Compressed (mammography paddle) and cropped phantoms | **Excluded**; the preprocessor refuses these file names |
| VICTRE mammograms / DBT projections / reconstructions | same | X-ray imaging chain | Out of scope, not used |
| VICTRE lesion models (`Lesion Insertion/VICTRE_LesionModels/*.raw`) | same, CC0 | Voxel lesion masks (e.g. 166³ uint8) | Supported as `tumour.shape: voxel` input (`tumour.py`) |
| DIDSR breastPhantom generator | https://github.com/DIDSR/breastPhantom, CC0 | C++ generator for new uncompressed phantoms (custom size/density/laterality) | Builds with VTK 9 after `third_party/breastPhantom-vtk9.patch`; new phantoms can be fed to the same preprocessor. Not required for the bundled examples |
| IT'IS Tissue Properties V5.0 | https://itis.swiss/virtual-population/tissue-properties/downloads, DOI 10.13099/VIP21000-05-0 | Gabriel 4-Cole-Cole parameters + low-frequency conductivity, per tissue | Used for skin, muscle, blood, connective tissue, fat (extract in `data/materials/itis_v5_extract.yaml`) |
| UCI Breast Tissue (`BreastTissue.xls`) | https://archive.ics.uci.edu/dataset/192/breast+tissue (also in the supplied papers ZIP) | 106 ex-vivo spectra summarised as 9 spectral features (from Jossinet's 12/7-point impedivity spectra) | Used: per-sample Cole impedivity fit (`scripts/fit_uci_cole.py`), class medians become adipose / glandular / connective / carcinoma / fibroadenoma / mastopathy materials |
| Surowiec et al. 1988 Table I | supplied PDF | Per-sample Cole–Cole fits, carcinoma and surrounding tissue, 37 °C | Transcribed to `data/materials/surowiec1988_table1.csv`; materials `carcinoma_surowiec`, `peritumoral_low_surowiec` |
| Halter et al. 2009 | supplied PDF / PMC2792899 | Probe-based in-vivo / ex-vivo conductivity & permittivity | Single-frequency carcinoma conductivity (`carcinoma_halter`, sensitivity only) |
| Other papers | supplied PDFs | see `docs/materials.md` | Reviewed; not adopted as solver inputs (reasons recorded) |

Not provided / still useful (see `docs/assumptions_and_limitations.md` for impact):
- Jossinet 1998 full spectra (the paper itself was not in the ZIP; only its derived UCI features are available).
- Gregory et al. 2012 full numeric data (only Cole relaxation frequencies are tabulated).
- Measured electrode–skin contact impedance for the actual wearable electrodes and gel.
- A real left/right pair from the same subject (VICTRE gives one breast; bilateral runs mirror it).
