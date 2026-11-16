# bioimpedance_v1 — simulated breast transfer impedance with and without lesions

Intended use: training/evaluating a model that **flags whether a lesion is present** from bioimpedance
differences (lesion breast vs. lesion-free breast). Malignant/benign labels are included only as descriptive
metadata (`lesion_class`) and are not the intended target.

Regenerate: `python scripts/generate_dataset.py configs/dataset_v1.yaml --workers 5`
(the sweep definition is copied to `sweep_config.yaml`; versions/commit in `provenance.json`).

## What was simulated

* **Breast compositions** — the four bundled VICTRE breastPhantom label volumes (0.5 mm, left breast):
  `fatty`, `scattered`, `hetero` (heterogeneously dense), `dense`. Tissue properties: UCI breast-tissue
  dielectric data (`uci_breast` assignment in `data/materials/library.yaml`, see `materials.csv`).
* **Electrodes** — 4 circular electrodes (5 mm radius) at 12, 3, 6, 9 o'clock, 55 % of the way from the nipple
  to the chest wall; complete electrode model, contact impedance 5e-3 Ω·m²; 100 µA drive current.
* **Frequencies** — 15.625, 31.25, 62.5, 125, 250, 500, 1000 kHz.
* **Lesion geometries** (`lesion_shape_id`) — `sphere_r4` (8 mm diameter), `sphere_r7` (14 mm), `ellipsoid_9x5x5`
  (18 × 10 × 10 mm), `victre_spiculated_14` (VICTRE spiculated mass `data/lesions/mass_-308854003_cropped_166.raw.gz`,
  scaled to 14 mm extent).
* **Lesion locations** — clock hour 12, 1:30, 3, 4:30, 6, 7:30, 9, 10:30 (i.e. under an electrode and halfway
  between electrodes) × depth 10 mm and 18 mm below the skin, at 55 % nipple-to-chest-wall.
* **Lesion materials** (`lesion_material`) — `carcinoma_uci`, `carcinoma_surowiec`, `fibroadenoma_uci`,
  `mastopathy_uci`.

Each **geometry** (composition × shape × location, `geometry_id`) is meshed once (2.5 mm global, 1.25 mm inside
the lesion). On that mesh, the lesion region is solved with its host tissue (the **lesion-absent baseline**,
`lesion_material` empty) and then with each lesion material. Every lesion record therefore has a baseline
on an identical mesh (`baseline_record_id`), so differences contain no meshing noise. Geometries the simulator
rejects (lesion touching skin/air/chest wall, or within 1 mm of an electrode) are listed in
`rejected_geometries.csv` and have no records.

## Files

| file | one row per | content |
|---|---|---|
| `records.csv` | simulated breast state | `record_id`, `geometry_id`, `composition`, `lesion_present`, `lesion_material`, `lesion_class`, `baseline_record_id`, lesion shape/clock/depth, lesion centre (mm, phantom coords), nominal and meshed volume, equivalent diameter, min distance to an electrode, fraction of glandular/fat tissue it replaced, mesh size, solver residual, materials evaluated outside their fitted frequency range |
| `measurements.csv` | record × frequency × measurement | complex transfer impedance `z_re_ohm`, `z_im_ohm`, `z_mag_ohm`, `z_phase_deg` |
| `measurement_definitions.csv` | measurement | `measurement_type` (tetrapolar/bipolar), current injected `inject_pos`→`inject_neg`, voltage sensed `sense_pos`−`sense_neg` |
| `lesion_effect.csv` | lesion record × frequency × measurement | lesion minus baseline: `dz_re_ohm`, `dz_im_ohm`, `d_mag_ohm`, `rel_d_mag` (=\|Z\|/\|Z₀\|−1), `d_phase_deg` |
| `electrodes.csv` | geometry × electrode | electrode centre (mm), clock hour, projected and meshed contact area |
| `materials.csv` | material × frequency | conductivity `sigma_s_per_m` and relative permittivity `eps_r` used, and whether it was extrapolated |
| `tensors.npz` | — | `Z` complex [record, frequency, measurement]; `record_id`, `baseline_index`, `lesion_present`, `composition`, `frequency_hz`, `measurement_id` |

Measurements (`measurement_id`): `adj_E1E2_E3E4` etc. — adjacent-drive tetrapolar (inject on one neighbouring
pair, sense on the opposite pair); `across_vertical` — inject 12↔6, sense 3−9 (near-balanced, so its
baseline is small and relative changes look large); `bipolar_E1E3`, `bipolar_E2E4` — inject and sense on the
same pair (dominated by contact impedance).

## Using it for left/right comparison

The simulator mirrors one breast to build the contralateral side, so a "between-breast difference" for a
lesion record is simply `record − baseline` (`lesion_effect.csv`), and a lesion-free pair has difference 0.
**Not simulated:** natural left/right anatomical asymmetry, electrode placement error, contact-impedance drift,
and instrument noise. Add these (e.g. pair a record with a baseline of a different geometry, or inject noise
in proportion to |Z|) before drawing conclusions about detectability. Split train/test by `geometry_id` (or by
composition) so a lesion record and its own baseline never fall on different sides of the split.
