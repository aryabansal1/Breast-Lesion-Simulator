"""Left/right pair construction and leakage-safe splits.

A *pair* is (ipsilateral breast a, contralateral breast b). Label 1 = a carries a lesion, 0 = both lesion-free.
The contralateral side is always a lesion-free record: the simulator's bilateral mode mirrors one breast, so the
true healthy contralateral breast *is* the baseline record (plus the nuisance added in noise.py).
"""
from __future__ import annotations

import numpy as np

from .data import BreastDataset
from .noise import NoiseModel

PAIR_MODES = ("same_mesh", "cross_mesh")


def other_baseline(ds: BreastDataset, rows, rng):
    """For each row, a random lesion-free record of the same composition on a *different* mesh (if one exists)."""
    base = ds.baselines
    out = np.empty(len(rows), int)
    for c in np.unique(ds.composition[rows]):
        pool = base[ds.composition[base] == c]
        sel = np.flatnonzero(ds.composition[rows] == c)
        own = ds.baseline_index[rows[sel]]
        pick = pool[rng.integers(len(pool), size=len(sel))]
        if len(pool) > 1:
            clash = pick == own
            while clash.any():
                pick[clash] = pool[rng.integers(len(pool), size=int(clash.sum()))]
                clash = pick == own
        out[sel] = pick
    return out


def make_pairs(ds: BreastDataset, rng, mode="same_mesh", neg_per_baseline=None):
    """All lesion pairs plus a balanced set of lesion-free pairs. Returns (a_idx, b_idx, label)."""
    if mode not in PAIR_MODES:
        raise ValueError(f"pair mode must be one of {PAIR_MODES}")
    les, base = ds.lesions, ds.baselines
    k = neg_per_baseline or max(1, int(round(len(les) / max(len(base), 1))))
    neg = np.repeat(base, k)
    a = np.concatenate([les, neg])
    if mode == "same_mesh":
        b = ds.baseline_index[a]
    else:  # contralateral from another mesh for *both* classes, so mesh differences cannot act as a label cue
        b = other_baseline(ds, a, rng)
    y = np.concatenate([np.ones(len(les)), np.zeros(len(neg))]).astype(np.float32)
    return a, b, y


def pair_features(Za, Zb) -> np.ndarray:
    """log(Za/Zb) as [n, 2, F, M] float32: channel 0 = log-magnitude ratio, 1 = phase difference (rad)."""
    r = np.log(Za / Zb)
    return np.stack([r.real, r.imag], axis=1).astype(np.float32)


def sample_pair_features(ds: BreastDataset, a, b, noise: NoiseModel, anat_dirs, rng, swap=True, scale=None):
    """Apply independent device nuisance to both breasts and anatomical asymmetry to b; optionally swap sides."""
    n = len(a)
    s = noise._pair_scale(n, rng) if scale is None else np.broadcast_to(np.asarray(scale, float), (n,))
    kw = dict(frequency_hz=ds.frequency_hz, measurement_type=ds.measurement_type,
              measurement_electrodes=ds.measurement_electrodes, scale=s)
    Za = noise.perturb(ds.Z[a], rng, **kw)
    Zb = noise.perturb(ds.Z[b], rng, **kw) * noise.asymmetry(n, rng, ds.frequency_hz, anat_dirs, scale=s)
    if swap:
        flip = rng.random(n) < 0.5
        Za[flip], Zb[flip] = Zb[flip].copy(), Za[flip].copy()
    return pair_features(Za, Zb)


def clean_effect(ds: BreastDataset, a) -> np.ndarray:
    """Noise-free rms |log(Z_lesion / Z_baseline)| of each ipsilateral record (0 for baselines)."""
    r = np.log(ds.Z[a] / ds.Z[ds.baseline_index[a]])
    return np.sqrt(np.mean(np.abs(r) ** 2, axis=(1, 2)))


# ----------------------------------------------------------------------------------------------- splits

def loco_splits(ds: BreastDataset):
    """Leave-one-composition-out: [(name, train_rows, test_rows)]. Needs >= 2 compositions."""
    comps = sorted(set(ds.composition))
    if len(comps) < 2:
        raise ValueError("leave-one-composition-out needs at least two compositions")
    return [(f"loco_{c}", np.flatnonzero(ds.composition != c), np.flatnonzero(ds.composition == c)) for c in comps]


def group_kfold_splits(ds: BreastDataset, k=5, seed=0):
    """K folds over geometry_id (a lesion record and its baseline are always on the same side of the split)."""
    geoms = np.array(sorted(set(ds.geometry_id)))
    k = min(k, len(geoms))
    if k < 2:
        raise ValueError("group k-fold needs at least two geometries")
    perm = np.random.default_rng(seed).permutation(geoms)
    fold_of = {g: i % k for i, g in enumerate(perm)}
    f = np.array([fold_of[g] for g in ds.geometry_id])
    return [(f"geom_fold{i}", np.flatnonzero(f != i), np.flatnonzero(f == i)) for i in range(k)]


def make_splits(ds: BreastDataset, strategy="loco", k=5, seed=0):
    if strategy == "auto":
        strategy = "loco" if len(set(ds.composition)) >= 2 else "group_kfold"
    if strategy == "loco":
        return loco_splits(ds)
    if strategy == "group_kfold":
        return group_kfold_splits(ds, k, seed)
    raise ValueError(f"unknown split strategy {strategy!r}")


def inner_validation(ds: BreastDataset, member: int, frac=0.2, seed=0):
    """(train_rows, val_rows) inside a training set: rotate one held-out composition per ensemble member when
    >= 2 compositions are available (anatomy-level early stopping), else hold out ``frac`` of the geometries."""
    comps = sorted(set(ds.composition))
    if len(comps) >= 2:
        c = comps[member % len(comps)]
        return np.flatnonzero(ds.composition != c), np.flatnonzero(ds.composition == c)
    geoms = np.array(sorted(set(ds.geometry_id)))
    rng = np.random.default_rng(seed + 1000 * member)
    n_val = max(1, int(round(frac * len(geoms))))
    val = set(rng.permutation(geoms)[:n_val])
    m = np.array([g in val for g in ds.geometry_id])
    return np.flatnonzero(~m), np.flatnonzero(m)
