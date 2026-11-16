# Lesion-flag network: design decisions

Code: `src/lesionnet/` · CLI: `scripts/train_lesion_net.py` · config: `configs/nn_v1.yaml` · tests: `tests/test_lesionnet.py`

**Task.** Given complex impedance spectra measured on the left and right breast with the same 4-electrode
pattern, output the probability that *at least one breast contains a lesion* (binary flag). No localisation,
no malignant-vs-benign (`lesion_class` / `lesion_material` are used only for stratified reporting).

**Status.** Research code for the simulated dataset. The results it produces are statements about the
simulator plus an **assumed** nuisance model (§4); they are not clinical performance.

---

## 1. Summary of the chosen design

| Decision | Choice | Main reason (section) |
|---|---|---|
| Input | complex log-ratio `log(Z_a / Z_b)` → 2 channels (log-magnitude ratio, phase difference) × 7 freq × 7 meas | bilateral comparison removes the shared anatomy; log makes multiplicative nuisance additive; keeps phase (§3.1) |
| Pairing | positive = lesion record vs its *own* lesion-free baseline (`same_mesh`); negative = baseline vs itself; healthy side always verified lesion-free | in the pilot, mesh-to-mesh baseline differences (3–14 % rms) are 10–100× the lesion effect (0.08–1.2 % rms) (§3.2) |
| Realism | stochastic nuisance added independently to each side every epoch: measurement noise, electrode re-placement, contact-impedance drift, left/right anatomical asymmetry; magnitudes domain-randomised ×0.5–2 | the simulator's two sides are bit-identical, so without nuisance the task is trivially solvable (§4) |
| Model | `PairDiffNet`: noise-whitening → shared 1-D conv along frequency per measurement → residual MLP (2 blocks, width 64) + explicit log-energy features → logit; symmetrised `(g(x)+g(−x))/2` | small tabular-spectral input, ~86 k params; exact left/right swap invariance (§5) |
| Training | BCE, AdamW, cosine LR, grad-clip, **held-out validation set with train and validation loss every epoch**, early stopping on validation loss, temperature scaling on validation | user requirement; small data needs early stopping; calibrated probabilities (§6) |
| Uncertainty | 5-member deep ensemble, each with a different held-out validation composition | variance reduction and better calibration on tiny data (§6.4) |
| Evaluation | outer leave-one-composition-out (LOCO) split, else geometry-grouped K-fold; a lesion record and its baseline never cross splits | avoids the leakage that pairing creates (§7) |
| Baselines | Ledoit–Wolf whitened energy detector; logistic regression on squared whitened features | the NN must beat a matched-filter-style detector to earn its place (§7.3) |
| Reports | AUROC, AUPRC, sensitivity @ 90/95 % specificity, Brier, ECE, accuracy; per lesion-effect bin / material / depth / composition; noise and asymmetry sweeps | effect size and nuisance level are the real drivers of detectability (§7.4) |

---

## 2. Problem characteristics that drive the design

1. **Small, structured tabular data.** ≈256 geometries × (1 baseline + 4 lesion materials) ≈ 1,280 records;
   each record is 7 frequencies × 7 measurements complex = 98 real numbers. The number of *independent*
   anatomies is far smaller (4 VICTRE compositions), so the effective sample size for generalisation across
   anatomy is ~4, not 1,280.
2. **Tiny effects.** Pilot (6 dense geometries, 4 mm sphere, 4 materials): rms over all readings of
   `|log(Z_lesion/Z_baseline)|` = 0.08 % / 0.30 % / 1.2 % (min / median / max); largest single reading 0.26–4 %.
   The across-breast tetrapolar reading carries most of it (median 0.76 % rms over frequency), adjacent
   tetrapolar 0.08–0.18 %, bipolar only 0.006–0.012 % (bipolar readings are dominated by contact impedance).
   This matches the bioimpedance literature: lesion contrast is large in conductivity/permittivity
   (Surowiec 1988; Jossinet 1998; Estrela da Silva 2000; Halter 2009) but heavily diluted in surface
   measurements of a whole breast (Zou & Guo 2003; Assenheimer 2001).
3. **Anatomy dominates.** Lesion-free baselines of *different* geometries of the same phantom differ by
   2.9 % / 7.3 % / 14.5 % rms (min / median / max), i.e. one to two orders of magnitude more than the lesion.
   Detection is only feasible relative to a matched reference — in practice the contralateral breast, which is
   exactly the clinical rationale of bilateral comparison (T-Scan: Assenheimer 2001; Cherepenin 2001; and
   bilateral-asymmetry features in mammography DL, Wu 2020).
4. **No real nuisance in the simulator.** Bilateral mode mirrors one breast, so the healthy contralateral side is
   bit-identical to the lesion-free baseline. Real systems are limited by electrode placement, contact
   impedance and natural asymmetry (Kolehmainen 1997; McAdams 1996; Boyle & Adler 2011; Wi 2014), not by
   instrument noise. Any detector trained on noise-free pairs would learn "is the difference ≠ 0", which is
   meaningless.

---

## 3. Data handling

### 3.1 Input representation — complex log-ratio
`pairs.pair_features(Za, Zb)`: `r = log(Za / Zb)`; channels `[Re r, Im r] = [ln|Za|−ln|Zb|, ∠Za−∠Zb]`
(radians), shape `[2, F, M]`.

* **Ratio, not two separate spectra.** A Siamese encoder over each breast followed by a difference
  (Bromley 1994; Daudt 2018 change detection) would first have to learn to cancel the ~7 % anatomy term that
  the ratio removes analytically. With ~4 anatomies the encoder could not learn that invariance. The
  log-ratio is exactly what a Siamese network with a log front-end and subtraction fusion computes, so we keep
  the inductive bias and drop the learning burden. Clinical bilateral systems also use L/R ratios or
  differences.
* **Log domain.** Placement, contact and anatomy perturbations act multiplicatively on |Z| (gain) and additively
  on phase, so in the log domain they are additive and roughly Gaussian, which suits both the whitening step and
  the linear baselines. Lesion effects (< 5 %) are linear in log space.
* **Keep phase.** The permittivity contrast of carcinoma is frequency dependent (β-dispersion, Surowiec 1988;
  Jossinet 1998), which shows up mostly in phase; discarding it (magnitude-only) throws away an independent
  channel. Using real/imag of the log-ratio rather than raw Re/Im avoids scale differences of ~10³ between
  bipolar and tetrapolar readings.
* **No per-record standardisation** (it would destroy the absolute effect size). Instead the model has a fixed
  per-feature scale `s` = std of lesion-free pairs under the training nuisance model
  (`train.feature_scale`) — a noise-whitening step, so a value of 1 means "one nuisance sigma".

### 3.2 Pair construction (`pairs.make_pairs`)
* Positive: (lesion record, its `baseline_index`). Negative: (baseline, baseline). Both sides then receive
  *independent* nuisance (§4), so negatives are not trivially zero.
* Class balance: by default each baseline is used as a negative as many times as it has lesion variants (4),
  giving 1:1 pairs. Balanced classes make BCE and AUROC well behaved; prevalence-specific thresholds should be
  set later on real data (we report AUPRC and sensitivity at fixed specificity for that reason; Saito &
  Rehmsmeier 2015).
* Random side swap: each pair is presented as (a, b) or (b, a). Combined with the symmetrised model the output
  is exactly invariant (tested).
* `same_mesh` (default) vs `cross_mesh` (healthy side = another geometry of the same composition). Cross-mesh
  pairing would mimic anatomical asymmetry with *real* simulated anatomy, but in the pilot those differences are
  3–14 % rms — far larger than any plausible left/right asymmetry of the same woman measured with the same
  electrode belt — and they would make every lesion undetectable. Real L/R asymmetry is therefore modelled
  parametrically (§4.4) and `cross_mesh` is kept as an option / stress test.

### 3.3 Why not a single-breast classifier?
A model on one breast alone would have to separate a 0.3 % lesion effect from 7 % inter-anatomy variation with 4
anatomies to learn from. It also contradicts the user's goal (differences between breasts).

---

## 4. Nuisance model (`noise.NoiseModel`) — sim-to-real gap

Because the simulator has no nuisance, *the nuisance model defines the task*. We use domain randomisation
(Tobin 2017; cf. Shrivastava 2017 for the sim-to-real problem): every training batch draws fresh nuisance for
each side, and every sigma is multiplied by a log-uniform factor in `scale_jitter` = [0.5, 2] so the network does
not overfit one assumed level. All defaults are **assumptions to be calibrated** against hardware repeatability
and volunteer L/R data; the noise sweeps (§7.4) show how conclusions depend on them.

| Term | Model | Default | Rationale |
|---|---|---|---|
| Measurement noise | complex Gaussian, relative | 1e-4 (≈80 dB SNR) | typical multi-frequency bioimpedance front-ends (Wi 2014) |
| Electrode re-placement | per electrode log-gain N(0, σ) (+10 % of it as phase), each reading gets the sum over its electrodes | 2e-3 | placement errors of a few mm change surface readings by ~0.1–1 % (Kolehmainen 1997; Boyle & Adler 2011) |
| Contact-impedance drift | constant-phase-element `ΔZ ∝ (jω)^−α`, α∈[0.5, 0.9], on bipolar readings; 1 % leak into tetrapolar | 2e-2 | skin–electrode impedance varies by tens of percent and is CPE-like (McAdams 1996); tetrapolar readings largely reject it |
| L/R anatomical asymmetry | smooth complex multiplicative field: low-order polynomials in log-frequency per measurement, plus random combinations of composition-difference directions (`anatomy_directions`) | 3e-3 rms | natural asymmetry is spectrally smooth and partly "looks like a different fat/fibroglandular mix"; magnitude unknown |

`NoiseModel.perturb` applies the first three to each side; `NoiseModel.asymmetry` multiplies one side by the
asymmetry field. Lesion-free pairs therefore differ by a realistic-looking random spectrum, and the network has to
learn the *shape* of lesion effects (which measurements and which frequency/phase signature) relative to it.

Test-time nuisance uses anatomy directions computed from all compositions (`dirs_all` in `train.evaluate_split`),
including the held-out one. This is not label leakage — no test record is seen in training and the directions
are only used to *generate* nuisance — and it makes the test harder: the model faces asymmetry directions it
never saw. Training uses only training-composition directions.

---

## 5. Architecture (`model.PairDiffNet`, ≈86 k parameters)

```
x [B,2,F=7,M=7] -> x / s                                   # noise whitening (buffer, not learned)
per measurement m: Conv1d(2→16,k3) GELU Conv1d(16→16,k3) GELU along frequency   # shared weights
concat tokens [B, 7·16·7] -> Linear -> h [B,64]
h += Linear(log mean_f x² per measurement & channel)       # explicit energy features [B, 14]
2 × ResBlock(pre-LayerNorm, Linear 64→128, GELU, Dropout 0.1, Linear 128→64) -> LayerNorm -> Linear -> g(x)
logit(x) = (g(x) + g(−x)) / 2                              # exact left/right swap invariance
```

Why each part:
* **Small residual MLP backbone.** On small tabular data, well-regularised MLPs / ResNet-style MLPs are
  competitive with transformers and close to gradient-boosted trees (Grinsztajn 2022; Shwartz-Ziv & Armon 2022;
  Gorishniy 2021; Kadra 2021). Pre-norm residual blocks (He 2016) train stably at this depth without tuning.
* **Shared 1-D convolution along frequency.** Impedance spectra are smooth in log-frequency (Cole-type
  dispersions); a shared local filter is the inductive bias that makes 1-D CNNs strong on small spectroscopic
  datasets (Acquarelli 2017), and weight sharing across the 7 measurements cuts parameters. The flattening keeps
  measurement identity (electrode geometry matters), unlike a permutation-invariant pooling.
* **Explicit energy features.** The optimal detector for a known signal in Gaussian noise is a whitened
  correlation; for an unknown signal, a whitened energy. Feeding log-energies per measurement gives the network
  this classical statistic for free so it only has to learn corrections (shape, phase signature).
* **Symmetrisation.** In log-ratio space, swapping breasts maps x → −x. Averaging g(x) and g(−x) makes the
  output exactly even in x, so "lesion in left" and "lesion in right" are the same flag, and halves the
  effective hypothesis space. (Swap augmentation alone would only make this approximately true.)
* **Size.** ~86 k parameters vs ~1–5 k training pairs per epoch (fresh nuisance each epoch makes the effective
  sample much larger). Dropout 0.1, weight decay 1e-4 and early stopping control overfitting.

### Alternatives considered
| Alternative | Why not (now) |
|---|---|
| Gradient-boosted trees (XGBoost/LightGBM) on log-ratio features | strong on tabular data (Grinsztajn 2022), but no exact swap symmetry, no natural use of fresh per-epoch nuisance, poorer probability calibration; worth adding as a baseline once the full dataset exists |
| FT-Transformer / attention over 49 readings | tokens are few and data tiny; attention adds parameters without a matching inductive bias (Gorishniy 2021 reports ResNet-MLP on par on small sets) |
| Siamese encoder per breast + |e_a − e_b| | see §3.1: must learn to cancel anatomy from ~4 anatomies; equivalent to our input up to the log front-end |
| 2-D CNN on the 7×7 grid | the measurement axis is not spatially ordered, so 2-D locality is not meaningful |
| Graph network over electrodes | 4 electrodes / 7 readings is too small for a GNN to pay off; electrode identity is already in the flattened token |
| EIT reconstruction then image CNN (e.g. Deep D-Bar, Hamilton & Hauptmann 2018; dominant-current DL, Wei 2019; review Khan & Yousaf 2019) | 4 electrodes give 7 independent readings — far too few to reconstruct an image; reconstruction would only add an ill-posed step for a binary decision |
| One-class / anomaly detection (train only on healthy pairs) | attractive for real data without labels; here labels are free, and supervised training exploits the lesion signature. Can be revisited with volunteer data |
| Gaussian-process / kernel classifier | good with small n, but the per-epoch fresh-nuisance training set is large; logistic regression on whitened energies (our baseline) covers the kernel-ish regime |

---

## 6. Training

### 6.1 Held-out validation set and per-epoch losses (user requirement)
Inside every outer training split, `pairs.inner_validation` holds out a validation set that is never used for
gradient updates:
* ≥ 2 compositions in the training split → one whole composition (rotated across ensemble members), so early
  stopping selects for *cross-anatomy* generalisation;
* otherwise → 20 % of geometries (`val_frac`), with each lesion record kept with its baseline.

Every epoch `train.train_member` passes both sets through the network in eval mode and logs, to
`training_history.csv` (columns `epoch, train_loss_running, train_loss, val_loss, val_auroc, lr, time_s, split,
member, val_composition, temperature`):
* `train_loss_running` — mean BCE over the epoch's minibatches (dropout on, fresh nuisance);
* `train_loss` — BCE on a fixed noisy draw of the training pairs, eval mode (directly comparable with val);
* `val_loss`, `val_auroc` — on a fixed set of validation pairs with `val_draws`=4 nuisance draws (fixed so the
  curve is not noise-dominated).
`loss_curves.png` plots train vs validation loss per member and split. The outer test split is untouched
until the end, so there are three disjoint sets: train / validation / test.

### 6.2 Optimisation
BCE with logits; AdamW (Loshchilov & Hutter 2019) lr 1e-3, weight decay 1e-4; cosine schedule over 200 epochs;
batch 256; gradient-norm clip 1.0; `pairs_per_epoch_mult`=2 passes over the pair list per epoch, each with
fresh nuisance. Early stopping: keep the weights with the lowest validation loss, stop after 40 epochs
without improvement (Prechelt 1998).

### 6.3 Calibration
Temperature scaling (Guo 2017) fitted on the validation logits of each member after early stopping; ECE and
Brier are reported on test.

### 6.4 Ensemble
5 members, each with its own seed and validation composition; the test probability is the mean of the
calibrated member probabilities (Lakshminarayanan 2017). With few anatomies, members that early-stopped on
different compositions give a cheap read on anatomy-induced variance.

---

## 7. Evaluation

### 7.1 Splits and leakage
Pairing creates the classic leakage trap (Kaufman 2011): a lesion record and its baseline share the anatomy, so if
the baseline is in training and the lesion in test, the network has "seen" the test breast. Splits are
therefore by group (Roberts 2017):
* **LOCO** (default when ≥ 2 compositions): train on 3 compositions, test on the 4th — tests generalisation to
  an unseen anatomy class, the closest analogue to a new patient.
* **Geometry-grouped K-fold** (group = `geometry_id`, or the record's `baseline_record_id` when that column is
  absent; fallback, or `split: group_kfold`): a geometry's baseline and all its lesion variants land in the same fold.
`BreastDataset.subset` refuses subsets in which a lesion record's baseline is missing, and
`test_splits_never_share_geometry` checks disjointness.

After CV, `run(final=True)` retrains on all data and saves `final_model.pt` (load with `train.load_ensemble`);
the reported performance is always the CV estimate, not the final model.

### 7.2 Metrics (`metrics.binary_metrics`)
AUROC; AUPRC (Saito & Rehmsmeier 2015); sensitivity at 90 % and 95 % specificity (screening-relevant operating
points); Brier score and 10-bin ECE (calibration); accuracy / sensitivity / specificity at p = 0.5. Reported per
outer split (`metrics_by_split.csv`) and as mean ± std (`metrics_summary.csv`); per-pair predictions in
`test_predictions.csv`.

### 7.3 Baselines
* `energy`: Ledoit–Wolf (2004) covariance of lesion-free pair features → whitened squared norm (a GLRT for an
  unknown signal in Gaussian nuisance).
* `logreg`: logistic regression on squared whitened features (learns which readings matter).
Both trained on the same nuisance-augmented training pairs.

### 7.4 Robustness and strata
* `noise_sweep.csv`: AUROC of all methods when *all* test-time sigmas are multiplied by 0.25–4, and when only
  the anatomical-asymmetry sigma is set to 0–3 %.
* `strata.csv`: AUROC/sensitivity per bin of the clean lesion effect (rms log-ratio, `pairs.clean_effect`), per
  lesion material, depth, shape and composition.
These answer "how big must a lesion effect be relative to the nuisance to be flagged", which is the transferable
result.

### 7.5 Pilot results (6 dense geometries — illustrative only)
`scripts/generate_dataset.py configs/dataset_v1.yaml --only dense --limit 6` (30 records), one composition so
geometry 5-fold CV, default nuisance:

| method | AUROC | AUPRC | sens @ 90 % spec |
|---|---|---|---|
| PairDiffNet ensemble | 0.85 ± 0.15 | 0.87 ± 0.12 | 0.61 ± 0.27 |
| logistic regression | 0.71 ± 0.26 | 0.79 ± 0.21 | 0.53 ± 0.38 |
| energy detector | 0.74 ± 0.14 | 0.71 ± 0.13 | 0.19 ± 0.12 |

Sensitivity by clean-effect bin: 0.13 (0.03–0.1 %), 0.56 (0.1–0.3 %), 0.97 (0.3–1 %), 1.0 (> 1 %).
Noise sweep (NN AUROC): 0.97 / 0.95 / 0.90 / 0.83 / 0.74 at ×0.25 / 0.5 / 1 / 2 / 4 nuisance; anatomical asymmetry
0 / 0.1 / 0.3 / 1 / 3 % → 0.95 / 0.95 / 0.91 / 0.76 / 0.63. With n = 6 geometries the confidence intervals are
wide; rerun on the full dataset before drawing conclusions.

---

## 8. Limitations and what to change with real data

* **Nuisance magnitudes are assumptions.** Detectability is set by effect / nuisance, and the nuisance here is
  invented. First calibration targets: test–retest repeatability with electrode re-placement, and L/R
  differences in healthy volunteers. Replace `NoiseModel` sigmas (or fit them) and rerun the sweeps.
* **Perfect bilateral symmetry in the simulator** — mitigated only by the parametric asymmetry model; a richer
  option is to simulate separate left/right phantoms of the same woman if such pairs become available.
* **Few anatomies** (4 compositions, all VICTRE): LOCO with 4 folds has high variance; expect a drop on real
  anatomy.
* **Lesion-present-in-one-breast only**; bilateral lesions would cancel partially in the ratio (rare clinically,
  but the model is not designed for it).
* **Not a diagnostic.** No thresholds are recommended; prevalence-dependent operating points must be chosen on
  real data.
* Fine-tuning path for real measurements: keep the input and symmetrisation, re-estimate `feature_scale` from
  real healthy pairs, fine-tune with the same grouped splits (by subject).

---

## 9. Running

```bash
pip install -e ".[dev,nn]"
python scripts/train_lesion_net.py configs/nn_v1.yaml                         # full dataset
python scripts/train_lesion_net.py configs/nn_v1.yaml --data outputs/pilot -o outputs/nn_pilot --no-final
python scripts/train_lesion_net.py configs/nn_v1.yaml --synthetic --quick -o outputs/nn_synth
pytest -q tests/test_lesionnet.py
```
Outputs: `metrics_by_split.csv`, `metrics_summary.csv`, `test_predictions.csv`, `training_history.csv`,
`loss_curves.png`, `noise_sweep.csv`, `strata.csv`, `results.json`, `final_model.pt` (+ final-model history).

---

## References

Deep learning / statistics
- Acquarelli J. et al. Convolutional neural networks for vibrational spectroscopic data analysis. *Anal. Chim. Acta* 954:22–31, 2017. doi:10.1016/j.aca.2016.12.010
- Bromley J. et al. Signature verification using a "Siamese" time delay neural network. 1994. doi:10.1142/9789812797926_0003
- Daudt R.C., Le Saux B., Boulch A. Fully convolutional Siamese networks for change detection. *ICIP* 2018. doi:10.1109/ICIP.2018.8451652
- Gorishniy Y. et al. Revisiting deep learning models for tabular data. *NeurIPS* 2021. arXiv:2106.11959
- Grinsztajn L., Oyallon E., Varoquaux G. Why do tree-based models still outperform deep learning on typical tabular data? *NeurIPS* 2022. doi:10.52202/068431-0037
- Guo C. et al. On calibration of modern neural networks. *ICML* 2017. arXiv:1706.04599
- He K. et al. Deep residual learning for image recognition. *CVPR* 2016. doi:10.1109/CVPR.2016.90
- Kadra A. et al. Well-tuned simple nets excel on tabular datasets. *NeurIPS* 2021. arXiv:2106.11189
- Kaufman S. et al. Leakage in data mining. *KDD* 2011. doi:10.1145/2020408.2020496
- Lakshminarayanan B., Pritzel A., Blundell C. Simple and scalable predictive uncertainty estimation using deep ensembles. *NeurIPS* 2017. arXiv:1612.01474
- Ledoit O., Wolf M. A well-conditioned estimator for large-dimensional covariance matrices. *J. Multivar. Anal.* 88:365–411, 2004. doi:10.1016/S0047-259X(03)00096-4
- Loshchilov I., Hutter F. Decoupled weight decay regularization. *ICLR* 2019. arXiv:1711.05101
- Prechelt L. Early stopping — but when? In *Neural Networks: Tricks of the Trade*, 1998.
- Roberts D.R. et al. Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography* 40:913–929, 2017. doi:10.1111/ecog.02881
- Saito T., Rehmsmeier M. The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE* 10:e0118432, 2015. doi:10.1371/journal.pone.0118432
- Shrivastava A. et al. Learning from simulated and unsupervised images through adversarial training. *CVPR* 2017. doi:10.1109/CVPR.2017.241
- Shwartz-Ziv R., Armon A. Tabular data: deep learning is not all you need. *Inf. Fusion* 81:84–90, 2022. doi:10.1016/j.inffus.2021.11.011
- Tobin J. et al. Domain randomization for transferring deep neural networks from simulation to the real world. *IROS* 2017. arXiv:1703.06907
- Wu N. et al. Deep neural networks improve radiologists' performance in breast cancer screening. *IEEE TMI* 39:1184–1194, 2020. doi:10.1109/TMI.2019.2945514

Bioimpedance / EIT
- Assenheimer M. et al. The T-SCAN technology: electrical impedance as a diagnostic tool for breast cancer detection. *Physiol. Meas.* 22:1–8, 2001. doi:10.1088/0967-3334/22/1/301
- Boyle A., Adler A. The impact of electrode area, contact impedance and boundary shape on EIT images. *Physiol. Meas.* 32:745–754, 2011. doi:10.1088/0967-3334/32/7/S02
- Cherepenin V. et al. A 3D electrical impedance tomography system for breast cancer detection. *Physiol. Meas.* 22:9–18, 2001. doi:10.1088/0967-3334/22/1/302
- Estrela da Silva J., Marques de Sá J.P., Jossinet J. Classification of breast tissue by electrical impedance spectroscopy. *Med. Biol. Eng. Comput.* 38:26–30, 2000. doi:10.1007/BF02344684
- Halter R.J. et al. The correlation of in vivo and ex vivo tissue dielectric properties to validate electromagnetic breast imaging. *Physiol. Meas.* 30:S121–S136, 2009. doi:10.1088/0967-3334/30/6/S08
- Hamilton S.J., Hauptmann A. Deep D-Bar: real-time electrical impedance tomography imaging with deep neural networks. *IEEE TMI* 37:2367–2377, 2018. doi:10.1109/TMI.2018.2828303
- Jossinet J. The impedivity of freshly excised human breast tissue. *Physiol. Meas.* 19:61–75, 1998. doi:10.1088/0967-3334/19/1/006
- Khan T.A., Yousaf M.H. Review on electrical impedance tomography: artificial intelligence methods and its applications. *Algorithms* 12:88, 2019. doi:10.3390/a12050088
- Kolehmainen V. et al. Assessment of errors in static electrical impedance tomography with adjacent and trigonometric current patterns. *Physiol. Meas.* 18:289–303, 1997. doi:10.1088/0967-3334/18/4/003
- McAdams E.T. et al. Factors affecting electrode-gel-skin interface impedance in electrical impedance tomography. *Med. Biol. Eng. Comput.* 34:397–408, 1996. doi:10.1007/BF02523842
- Surowiec A.J. et al. Dielectric properties of breast carcinoma and the surrounding tissues. *IEEE TBME* 35:257–263, 1988. doi:10.1109/10.1374
- Wei Z., Liu D., Chen X. Dominant-current deep learning scheme for electrical impedance tomography. *IEEE TBME* 66:2546–2555, 2019. doi:10.1109/TBME.2019.2891676
- Wi H. et al. Multi-frequency electrical impedance tomography system with automatic self-calibration for long-term monitoring. *IEEE TBioCAS* 8:119–128, 2014. doi:10.1109/TBCAS.2013.2256785
- Zou Y., Guo Z. A review of electrical impedance techniques for breast cancer detection. *Med. Eng. Phys.* 25:79–90, 2003. doi:10.1016/S1350-4533(02)00194-7
