"""Lesion-flag network: data loading, pairing, leakage-safe splits, nuisance model, model symmetry, training."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

from lesionnet.data import load_dataset, write_synthetic_dataset  # noqa: E402
from lesionnet.model import PairDiffNet  # noqa: E402
from lesionnet.noise import NoiseModel, anatomy_directions  # noqa: E402
from lesionnet.pairs import inner_validation, make_pairs, make_splits, pair_features, sample_pair_features  # noqa: E402
from lesionnet.train import EnergyDetector, load_ensemble, run  # noqa: E402

QUIET = dict(meas_sigma=0.0, placement_sigma=0.0, contact_sigma=0.0, anat_sigma=0.0)


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    p = write_synthetic_dataset(tmp_path_factory.mktemp("synth"), n_geometries=6, effect_range=(5e-3, 2e-2))
    return p, load_dataset(p)


def test_load_schema(synth):
    p, ds = synth
    assert ds.Z.shape == (4 * 6 * 3, 7, 7) and np.iscomplexobj(ds.Z)
    assert (ds.baseline_index[ds.baselines] == ds.baselines).all()
    assert not ds.lesion_present[ds.baseline_index[ds.lesions]].any()
    assert (ds.geometry_id[ds.baseline_index] == ds.geometry_id).all()
    assert list(ds.measurement_type).count("bipolar") == 2
    (p / "tensors.npz").rename(p / "tensors.bak")  # CSV fallback gives the same tensor
    try:
        ds2 = load_dataset(p)
    finally:
        (p / "tensors.bak").rename(p / "tensors.npz")
    j = [list(ds2.measurement_id).index(m) for m in ds.measurement_id]
    np.testing.assert_allclose(ds2.Z[:, :, j], ds.Z, rtol=1e-9)


@pytest.mark.parametrize("mode", ["same_mesh", "cross_mesh"])
def test_pairs_balanced_and_contralateral_healthy(synth, mode):
    _, ds = synth
    a, b, y = make_pairs(ds, np.random.default_rng(0), mode)
    assert y.sum() == len(ds.lesions) and (y == 0).sum() == len(ds.lesions)
    assert not ds.lesion_present[b].any()
    assert (ds.lesion_present[a] == (y == 1)).all()
    assert (ds.composition[a] == ds.composition[b]).all()
    if mode == "same_mesh":
        assert (b == ds.baseline_index[a]).all()
    else:
        assert (b != ds.baseline_index[a]).all()


@pytest.mark.parametrize("strategy", ["loco", "group_kfold"])
def test_splits_never_share_geometry(synth, strategy):
    _, ds = synth
    splits = make_splits(ds, strategy, k=3)
    covered = np.zeros(len(ds), int)
    for _, tr, te in splits:
        assert not set(ds.geometry_id[tr]) & set(ds.geometry_id[te])
        if strategy == "loco":
            assert len(set(ds.composition[te])) == 1 and not set(ds.composition[te]) & set(ds.composition[tr])
        covered[te] += 1
        sub = ds.subset(tr)
        itr, iva = inner_validation(sub, member=1)
        assert not set(sub.geometry_id[itr]) & set(sub.geometry_id[iva])
    assert (covered == 1).all()


def test_small_single_composition_splits_keep_training_data(synth):
    _, ds = synth
    c = ds.composition[0]
    g3 = sorted(set(ds.geometry_id[ds.composition == c]))[:3]
    small = ds.subset(np.flatnonzero(np.isin(ds.geometry_id, g3)))
    for _, tr, _ in make_splits(small, "group_kfold", k=5):
        itr, iva = inner_validation(small.subset(tr), member=0)
        assert len(itr) and len(iva)
    with pytest.raises(ValueError):
        make_splits(small.subset(np.flatnonzero(np.isin(small.geometry_id, g3[:2]))), "group_kfold")


def test_fixed_scale_jitter_is_applied():
    s = NoiseModel(scale=2.0, scale_jitter=(0.5, 0.5))._pair_scale(3, np.random.default_rng(0))
    np.testing.assert_allclose(s, 1.0)


def test_noise_free_pair_is_exact_and_swap_negates(synth):
    _, ds = synth
    nm = NoiseModel(**QUIET)
    dirs = anatomy_directions(ds.Z, ds.composition, ds.baselines)
    a, b, _ = make_pairs(ds, np.random.default_rng(0), "same_mesh")
    X = sample_pair_features(ds, a, b, nm, dirs, np.random.default_rng(1), swap=False)
    np.testing.assert_allclose(X, pair_features(ds.Z[a], ds.Z[b]), atol=1e-7)
    assert np.abs(X[ds.lesion_present[a] == 0]).max() < 1e-7
    np.testing.assert_allclose(pair_features(ds.Z[b], ds.Z[a]), -X, atol=1e-7)


def test_noise_magnitudes(synth):
    _, ds = synth
    rng = np.random.default_rng(0)
    Z = np.repeat(ds.Z[ds.baselines[:1]], 4000, 0)
    kw = dict(frequency_hz=ds.frequency_hz, measurement_type=ds.measurement_type,
              measurement_electrodes=ds.measurement_electrodes)
    r = np.log(NoiseModel(**{**QUIET, "meas_sigma": 1e-3}).perturb(Z, rng, **kw) / Z)
    assert abs(np.sqrt(np.mean(np.abs(r) ** 2)) / 1e-3 - 1) < 0.05
    r = np.log(NoiseModel(**{**QUIET, "contact_sigma": 1e-2}).perturb(Z, rng, **kw) / Z)
    bip = ds.measurement_type == "bipolar"
    assert np.abs(r[..., bip]).mean() > 20 * np.abs(r[..., ~bip]).mean()
    dirs = anatomy_directions(ds.Z, ds.composition, ds.baselines)
    asym = NoiseModel(**{**QUIET, "anat_sigma": 1e-2}).asymmetry(4000, rng, ds.frequency_hz, dirs)
    assert abs(np.sqrt(np.mean(np.abs(np.log(asym)) ** 2)) / 1e-2 - 1) < 0.1


def test_model_is_side_symmetric():
    torch.manual_seed(0)
    net = PairDiffNet(7, 7).eval()
    x = torch.randn(16, 2, 7, 7)
    with torch.no_grad():
        torch.testing.assert_close(net(x), net(-x))
        assert net(x).shape == (16,)
    asym = PairDiffNet(7, 7, symmetrize=False).eval()
    with torch.no_grad():
        assert not torch.allclose(asym(x), asym(-x))


def test_train_eval_end_to_end(tmp_path):
    torch.set_num_threads(1)
    ds = load_dataset(write_synthetic_dataset(tmp_path / "data", compositions=("fatty", "dense", "hetero"),
                                              n_geometries=4, effect_range=(5e-3, 2e-2)))
    cfg = dict(noise=dict(meas_sigma=1e-4, placement_sigma=5e-4, contact_sigma=5e-3, anat_sigma=5e-4),
               train=dict(ensemble=2, epochs=20, patience=20, scale_fit_rounds=3, val_draws=2),
               eval=dict(test_draws=2, noise_sweep=[1.0, 4.0], anat_sweep=[0.0]),
               model=dict(hidden=32, conv_channels=8))
    res = run(ds, cfg, tmp_path, final=True, log=lambda s: None)
    for f in ("metrics_by_split.csv", "metrics_summary.csv", "test_predictions.csv", "noise_sweep.csv",
              "strata.csv", "training_history.csv", "loss_curves.png", "model.pt", "results.json"):
        assert (tmp_path / f).exists(), f
    import pandas as pd
    h = pd.read_csv(tmp_path / "training_history.csv")
    assert {"train_loss", "train_loss_running", "val_loss", "val_auroc"} <= set(h.columns)
    assert h.groupby(["split", "member"]).train_loss.apply(lambda s: s.iloc[-1] < s.iloc[0]).mean() >= 0.75
    assert res["splits"] == [f"loco_{c}" for c in sorted(set(ds.composition))]
    st = pd.read_csv(tmp_path / "strata.csv")
    assert {"auroc_nn", "sensitivity_nn", "specificity_nn", "specificity_logreg", "n_neg"} <= set(st.columns)
    assert st.specificity_nn.between(0, 1).all()
    assert res["summary"]["nn"]["auroc_mean"] > 0.75
    pred = pd.read_csv(tmp_path / "test_predictions.csv")
    assert set(pred.label) == {0, 1} and pred.score_nn.between(0, 1).all()
    ens, ck = load_ensemble(tmp_path / "model.pt")
    x = torch.randn(5, 2, 7, 7).numpy() * 1e-3
    p = ens.predict_proba(x)
    assert p.shape == (5,) and ((p > 0) & (p < 1)).all()
    assert ck["measurement_id"] == list(ds.measurement_id)


def test_energy_detector_handles_noise_free_negatives():
    rng = np.random.default_rng(0)
    neg = np.zeros((20, 2, 7, 7), np.float32)
    pos = rng.normal(0, 1e-3, (20, 2, 7, 7)).astype(np.float32)
    e = EnergyDetector().fit(neg)
    s0, s1 = e.score(neg), e.score(pos)
    assert np.isfinite(s1).all() and s0.max() < s1.min()
