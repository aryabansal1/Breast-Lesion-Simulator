# Assumptions, limitations and missing data

This is a **numerical model of what the device would record**, not a validated predictor of patient
measurements and not a diagnostic tool. No machine learning, classification, thresholds or thermal
modelling are included.

## Physics
- Electroquasistatic: \(\nabla\cdot[(\sigma+j\omega\varepsilon)\nabla V]=0\); magnetic induction, wave
  effects and displacement current in air are neglected (valid ≪ 10 MHz for a ~10 cm domain).
- Phasor convention \(v(t)=\mathrm{Re}\{V e^{+j\omega t}\}\).
- Tissues are linear, isotropic and piecewise homogeneous per label (muscle anisotropy ignored).
- Complete electrode model: each electrode is an equipotential with specific contact impedance
  \(z_c\) (Ω·m²); \(\sum_l I_l=0\); reference \(\sum_l U_l=0\). Outside electrodes the skin–air boundary is
  insulating (\(\gamma\,\partial_n V=0\)).
- The phantom's posterior face is closed by an **assumed 20 mm muscle block** (label 41, configurable
  `mesh.chest_wall_mm`) whose far faces are insulating. Real current can spread into the thorax; a thicker
  block or different material changes absolute impedances (mostly for widely spaced electrodes). Treat as a
  sensitivity parameter.

## Anatomy
- VICTRE/breastPhantom are synthetic, uncompressed (verified by file type `_p_`, not inferred). One breast per
  phantom; laterality of +y is **assumed** (`y_positive_is_patient_left`).
- Bilateral runs mirror one breast. Left/right differences then only reflect what you change (tumour, electrodes),
  not natural asymmetry.
- 0.5 mm downsampling + 1.5–4 mm FE mesh: thin skin (~1.5 mm) is resolved by only one or two elements at 2 mm;
  vessels/ducts are only partly resolved. The tetrahedral mesh is voxel-based (stair-stepped surface). Electrode
  contact area is corrected to the smooth (projected) area by scaling \(z_c\) (`electrodes.stair_step_area_correction`).
- The breast is in a supine-like free shape from the phantom; gravity deformation, strap pressure and posture of
  a worn device are not modelled.

## Materials
- See `docs/materials.md`. Key inferences: UCI `I0` treated as Ω·cm; Cole fits to *derived* UCI features;
  Surowiec grouping by σDC; IT'IS gland LF conductivity is a thyroid substitute; TDLU/duct/calcification → gland.
- Ex-vivo values (most sources) differ from in-vivo (Halter 2009 reports systematic differences); temperature
  ranges from room temperature to 37 °C between sources.
- Values outside a source's frequency range are extrapolated and flagged in `metadata.json`.

## Tumour
- Tumour elements **replace** the tissue they overlap (volume reported per replaced label). No peritumoral
  oedema / desmoplastic shell unless you add one as a second inclusion.
- Placement is rejected if the tumour touches air, the outer boundary, or comes within
  `electrode_margin_mm` of an electrode.

## Electrodes and electronics
- Contact impedance default 5×10⁻³ Ω·m² is an assumption (wet-gel order of magnitude). Dry or textile electrodes
  can be orders of magnitude higher and frequency dependent (use `model: parallel_rc`).
- Ideal current source and ideal voltmeter: no cable capacitance, common-mode effects, amplifier input impedance,
  or stray capacitance to ground.

## Missing data that would improve fidelity
1. Measured contact impedance spectrum of the actual electrodes on skin.
2. Raw Jossinet spectra (not just UCI-derived features) or another breast dataset with bulk σ/ε vs frequency.
3. In-vivo breast property data at the device's frequencies.
4. Target electrode layout / sizes and device frequencies.
5. Real paired left/right anatomy if bilateral comparison is a goal.
