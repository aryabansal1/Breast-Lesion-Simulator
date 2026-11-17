"""Training, baselines, cross-validated evaluation and checkpointing for the lesion-flag detector."""
from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from .data import BreastDataset
from .metrics import binary_metrics
from .model import PairDiffNet
from .noise import NoiseModel, anatomy_directions
from .pairs import clean_effect, inner_validation, make_pairs, make_splits, other_baseline, sample_pair_features

DEFAULTS = dict(
    pair_mode="same_mesh",
    split="auto",            # loco | group_kfold | auto (loco when >= 2 compositions)
    k_folds=5,
    model=dict(conv_channels=16, hidden=64, n_blocks=2, dropout=0.1, symmetrize=True, energy_features=True),
    train=dict(ensemble=5, epochs=200, batch_size=256, lr=1e-3, weight_decay=1e-4, patience=40,
               pairs_per_epoch_mult=2, grad_clip=1.0, scale_fit_rounds=20, val_draws=4, val_frac=0.2),
    eval=dict(test_draws=4, noise_sweep=[0.25, 0.5, 1.0, 2.0, 4.0], anat_sweep=[0.0, 1e-3, 3e-3, 1e-2, 3e-2],
              effect_bins=[0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 1.0], baselines=True),
    seed=0,
)


def merge_config(cfg: dict | None) -> dict:
    out = copy.deepcopy(DEFAULTS)
    for k, v in (cfg or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    return out


@dataclass
class Member:
    model: PairDiffNet
    temperature: float
    history: pd.DataFrame
    val_composition: str = ""


@dataclass
class Ensemble:
    members: list = field(default_factory=list)

    @torch.no_grad()
    def logits(self, X):
        x = torch.as_tensor(X)
        return np.stack([(m.model.eval()(x) / m.temperature).numpy() for m in self.members])

    def predict_proba(self, X):
        return (1 / (1 + np.exp(-self.logits(X)))).mean(0)


# ----------------------------------------------------------------------------------------------- data helpers
def fixed_pair_set(ds: BreastDataset, noise, dirs, mode, draws, seed, scale=None):
    """Deterministic evaluation set: every pair of make_pairs() drawn ``draws`` times with independent nuisance."""
    rng = np.random.default_rng(seed)
    A, B, Y = [], [], []
    for _ in range(draws):
        a, b, y = make_pairs(ds, rng, mode)
        A.append(a), B.append(b), Y.append(y)
    a, b, y = np.concatenate(A), np.concatenate(B), np.concatenate(Y)
    X = sample_pair_features(ds, a, b, noise, dirs, rng, swap=True, scale=scale)
    return X, y, a, b


def feature_scale(ds: BreastDataset, noise, dirs, mode, rounds, seed):
    """Per-feature rms of lesion-free pairs (zero-mean by construction of the random side swap)."""
    rng = np.random.default_rng(seed)
    a = np.repeat(ds.baselines, rounds)
    b = a if mode == "same_mesh" else other_baseline(ds, a, rng)
    X = sample_pair_features(ds, a, b, noise, dirs, rng, swap=True)
    s = np.sqrt(np.mean(X.astype(np.float64) ** 2, axis=0))
    return np.maximum(s, 1e-6).astype(np.float32)   # floor: noise-free runs leave lesion-free pairs at exactly 0


# ----------------------------------------------------------------------------------------------- training
def _bce(model, X, y, bs=4096):
    lossf = nn.BCEWithLogitsLoss()
    model.eval()
    with torch.no_grad():
        z = torch.cat([model(torch.as_tensor(X[i:i + bs])) for i in range(0, len(X), bs)])
    return float(lossf(z, torch.as_tensor(y))), z.numpy()


def fit_temperature(logits, y):
    """Temperature scaling (Guo et al. 2017) on held-out validation logits."""
    z, t = torch.as_tensor(logits, dtype=torch.float64), torch.as_tensor(y, dtype=torch.float64)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(z / log_t.exp(), t)
        loss.backward()
        return loss

    opt.step(closure)
    return float(np.clip(log_t.exp().item(), 0.05, 20.0))


def train_member(ds_tr: BreastDataset, ds_va: BreastDataset, cfg, noise: NoiseModel, dirs, seed, log=print,
                 dirs_val=None):
    """Train one network (``dirs``: training anatomy directions; ``dirs_val``: for validation nuisance, default
    ``dirs``). Every epoch logs training loss (running, train mode), training loss and validation loss
    on fixed held-out pair sets (eval mode) and validation AUROC; early-stops on validation loss."""
    tc, mc, mode = cfg["train"], cfg["model"], cfg["pair_mode"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = PairDiffNet(len(ds_tr.frequency_hz), len(ds_tr.measurement_id), **mc)
    model.set_feature_scale(feature_scale(ds_tr, noise, dirs, mode, tc["scale_fit_rounds"], seed + 1))
    Xva, yva, _, _ = fixed_pair_set(ds_va, noise, dirs if dirs_val is None else dirs_val, mode, tc["val_draws"],
                                    seed + 2)
    Xtr_eval, ytr_eval, _, _ = fixed_pair_set(ds_tr, noise, dirs, mode, 1, seed + 3)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tc["epochs"])
    lossf = nn.BCEWithLogitsLoss()
    best, best_state, wait, hist = np.inf, None, 0, []
    for ep in range(tc["epochs"]):
        t0 = time.perf_counter()
        A, B, Y = zip(*[make_pairs(ds_tr, rng, mode) for _ in range(tc["pairs_per_epoch_mult"])])
        a, b, y = np.concatenate(A), np.concatenate(B), np.concatenate(Y)
        X = torch.as_tensor(sample_pair_features(ds_tr, a, b, noise, dirs, rng, swap=True))
        y = torch.as_tensor(y)
        perm = torch.as_tensor(rng.permutation(len(y)))
        model.train()
        run, nb = 0.0, 0
        for i in range(0, len(perm), tc["batch_size"]):
            j = perm[i:i + tc["batch_size"]]
            opt.zero_grad()
            loss = lossf(model(X[j]), y[j])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            opt.step()
            run, nb = run + loss.item() * len(j), nb + len(j)
        sched.step()
        tr_loss, _ = _bce(model, Xtr_eval, ytr_eval)
        va_loss, zva = _bce(model, Xva, yva)
        va_auc = float(roc_auc_score(yva, zva)) if 0 < yva.sum() < len(yva) else float("nan")
        hist.append(dict(epoch=ep + 1, train_loss_running=run / nb, train_loss=tr_loss, val_loss=va_loss,
                         val_auroc=va_auc, lr=sched.get_last_lr()[0], time_s=time.perf_counter() - t0))
        if va_loss < best - 1e-5:
            best, best_state, wait = va_loss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
        if (ep + 1) % 25 == 0 or ep == 0:
            log(f"    epoch {ep + 1:4d}  train {tr_loss:.4f}  val {va_loss:.4f}  val AUROC {va_auc:.3f}")
        if wait >= tc["patience"]:
            log(f"    early stop at epoch {ep + 1} (best val loss {best:.4f})")
            break
    model.load_state_dict(best_state)
    _, zva = _bce(model, Xva, yva)
    T = fit_temperature(zva, yva)
    return Member(model, T, pd.DataFrame(hist))


def train_ensemble(ds: BreastDataset, cfg, noise, dirs, seed, log=print):
    ens = Ensemble()
    for k in range(cfg["train"]["ensemble"]):
        tr, va = inner_validation(ds, k, cfg["train"]["val_frac"], seed)
        vc = ",".join(sorted(set(ds.composition[va])))
        log(f"  member {k}: train {len(tr)} records, validation {len(va)} records ({vc})")
        ds_tr = ds.subset(tr)
        dirs_tr = anatomy_directions(ds_tr.Z, ds_tr.composition, ds_tr.baselines)  # no validation anatomy in training
        m = train_member(ds_tr, ds.subset(va), cfg, noise, dirs_tr, seed + 7919 * (k + 1), log, dirs_val=dirs)
        m.val_composition = vc
        ens.members.append(m)
    return ens


# ----------------------------------------------------------------------------------------------- baselines
class EnergyDetector:
    """Classical GLRT for an unknown additive change in Gaussian noise: Mahalanobis energy d^T S^-1 d of the
    left/right log-ratio, S = Ledoit-Wolf covariance of lesion-free pairs (Hotelling-type statistic)."""

    def fit(self, Xneg):
        C = LedoitWolf().fit(Xneg.reshape(len(Xneg), -1).astype(np.float64)).covariance_
        C = C + np.eye(len(C)) * max(1e-12, 1e-9 * np.trace(C) / len(C))   # noise-free runs: C == 0
        self.L = np.linalg.cholesky(np.linalg.inv(C))
        return self

    def whiten(self, X):
        return X.reshape(len(X), -1) @ self.L

    def score(self, X):
        return np.sum(self.whiten(X) ** 2, axis=1)


class QuadraticLogReg:
    """Logistic regression on squared whitened features (a learned, weighted energy detector)."""

    def fit(self, X, y, energy: EnergyDetector):
        self.e = energy
        self.clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
        self.clf.fit(self.e.whiten(X) ** 2, y)
        return self

    def proba(self, X):
        return self.clf.predict_proba(self.e.whiten(X) ** 2)[:, 1]


def fit_baselines(ds, cfg, noise, dirs, seed):
    X, y, _, _ = fixed_pair_set(ds, noise, dirs, cfg["pair_mode"], 8, seed)
    e = EnergyDetector().fit(X[y == 0])
    return dict(energy=e, logreg=QuadraticLogReg().fit(X, y, e))


def score_all(ens, base, X):
    out = dict(nn=ens.predict_proba(X))
    if base:
        out["energy"] = base["energy"].score(X)
        out["logreg"] = base["logreg"].proba(X)
    return out


CALIBRATED = {"nn", "logreg"}


# ----------------------------------------------------------------------------------------------- evaluation
def evaluate_split(name, ds, tr, te, cfg, noise, log=print):
    seed = cfg["seed"]
    ds_tr, ds_te = ds.subset(tr), ds.subset(te)
    dirs_tr = anatomy_directions(ds_tr.Z, ds_tr.composition, ds_tr.baselines)
    dirs_all = anatomy_directions(ds.Z, ds.composition, ds.baselines)  # test nuisance may use unseen anatomy
    log(f"[{name}] train {len(tr)} records ({','.join(sorted(set(ds_tr.composition)))}), "
        f"test {len(te)} records ({','.join(sorted(set(ds_te.composition)))})")
    ens = train_ensemble(ds_tr, cfg, noise, dirs_tr, seed, log)
    base = fit_baselines(ds_tr, cfg, noise, dirs_tr, seed + 11) if cfg["eval"]["baselines"] else None
    ec = cfg["eval"]
    X, y, a, b = fixed_pair_set(ds_te, noise, dirs_all, cfg["pair_mode"], ec["test_draws"], seed + 13)
    scores = score_all(ens, base, X)
    rows = [dict(split=name, method=k, **binary_metrics(y, s, s if k in CALIBRATED else None))
            for k, s in scores.items()]
    rec = ds_te.records
    pred = pd.DataFrame(dict(split=name, record_id=rec.record_id.to_numpy()[a],
                             contralateral_record_id=rec.record_id.to_numpy()[b], label=y.astype(int),
                             clean_effect_rms=clean_effect(ds_te, a), **{f"score_{k}": v for k, v in scores.items()}))
    sweep = []
    for s in ec["noise_sweep"]:
        Xs, ys, _, _ = fixed_pair_set(ds_te, noise.scaled(s), dirs_all, cfg["pair_mode"], ec["test_draws"],
                                      seed + 17)
        for k, sc in score_all(ens, base, Xs).items():
            sweep.append(dict(split=name, sweep="all_sigmas_x", value=s, method=k,
                              auroc=binary_metrics(ys, sc)["auroc"]))
    for s in ec.get("anat_sweep") or []:
        Xs, ys, _, _ = fixed_pair_set(ds_te, noise.replace(anat_sigma=s), dirs_all, cfg["pair_mode"],
                                      ec["test_draws"], seed + 19)
        for k, sc in score_all(ens, base, Xs).items():
            sweep.append(dict(split=name, sweep="anat_sigma", value=s, method=k,
                              auroc=binary_metrics(ys, sc)["auroc"]))
    hist = pd.concat([m.history.assign(split=name, member=i, val_composition=m.val_composition,
                                       temperature=m.temperature) for i, m in enumerate(ens.members)])
    return pd.DataFrame(rows), pred, pd.DataFrame(sweep), hist, ens


def strata_table(pred: pd.DataFrame, records: pd.DataFrame, effect_bins):
    """Per subgroup: AUROC, sensitivity and specificity (calibrated 0.5 threshold). Negatives are the lesion-free
    pairs of the same split whose geometry shares the subgroup value (shape, depth, composition); for lesion-only
    attributes (material, effect bin) all lesion-free pairs of the split."""
    meta = records.set_index("record_id")
    cols = [c for c in ("lesion_material", "lesion_shape_id", "lesion_depth_mm", "composition") if c in meta]
    p = pred.join(meta[cols], on="record_id")
    p["clean_effect_bin"] = pd.cut(p.clean_effect_rms, effect_bins).astype(str)
    methods = [c[6:] for c in p.columns if c.startswith("score_") and c[6:] in CALIBRATED]
    out = []
    for col in cols + ["clean_effect_bin"]:
        for (split, val), g in p[p.label == 1].groupby(["split", col]):
            neg = p[(p.split == split) & (p.label == 0)]
            same = neg[neg[col].astype(str) == str(val)]
            neg = same if len(same) else neg
            yy = np.r_[np.ones(len(g)), np.zeros(len(neg))]
            row = dict(split=split, stratum=col, value=str(val), n_pos=len(g), n_neg=len(neg),
                       median_clean_effect=float(g.clean_effect_rms.median()))
            for k in methods:
                ss = np.r_[g[f"score_{k}"], neg[f"score_{k}"]]
                row[f"auroc_{k}"] = binary_metrics(yy, ss).get("auroc", np.nan)
                row[f"sensitivity_{k}"] = float(np.mean(g[f"score_{k}"] >= 0.5))
                row[f"specificity_{k}"] = float(np.mean(neg[f"score_{k}"] < 0.5))
            out.append(row)
    return pd.DataFrame(out)


def plot_history(hist: pd.DataFrame, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    splits = list(dict.fromkeys(hist.split))
    fig, axes = plt.subplots(1, len(splits), figsize=(4.5 * len(splits), 3.6), squeeze=False)
    for ax, s in zip(axes[0], splits):
        for i, g in hist[hist.split == s].groupby("member"):
            c = f"C{i}"
            ax.plot(g.epoch, g.train_loss, c, lw=1, label=f"m{i} train")
            ax.plot(g.epoch, g.val_loss, c, lw=1, ls="--", label=f"m{i} val ({g.val_composition.iloc[0]})")
        ax.set(title=s, xlabel="epoch", ylabel="BCE loss", yscale="log")
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_ensemble(ens: Ensemble, path, cfg, noise, ds: BreastDataset):
    torch.save(dict(config=cfg, noise=noise.to_dict(), frequency_hz=ds.frequency_hz.tolist(),
                    measurement_id=list(map(str, ds.measurement_id)),
                    members=[dict(state=m.model.state_dict(), temperature=m.temperature) for m in ens.members]),
               path)


def load_ensemble(path) -> tuple[Ensemble, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    mc = merge_config(ck["config"])["model"]
    F, M = len(ck["frequency_hz"]), len(ck["measurement_id"])
    ens = Ensemble()
    for m in ck["members"]:
        net = PairDiffNet(F, M, **mc)
        net.load_state_dict(m["state"])
        ens.members.append(Member(net.eval(), m["temperature"], pd.DataFrame()))
    return ens, ck


def run(ds: BreastDataset, cfg: dict, out_dir, final=True, log=print):
    """Cross-validated evaluation (+ optional final ensemble on all data). Writes CSV/JSON/PNG into out_dir."""
    cfg = merge_config(cfg)
    noise = NoiseModel.from_dict(cfg.get("noise"))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    splits = make_splits(ds, cfg["split"], cfg["k_folds"], cfg["seed"])
    M, P, S, H = [], [], [], []
    for name, tr, te in splits:
        m, p, s, h, _ = evaluate_split(name, ds, tr, te, cfg, noise, log)
        M.append(m), P.append(p), S.append(s), H.append(h)
        log(m.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    metrics, pred, sweep, hist = (pd.concat(x, ignore_index=True) for x in (M, P, S, H))
    strata = strata_table(pred, ds.records, cfg["eval"]["effect_bins"])
    num = metrics.select_dtypes("number").columns
    summary = metrics.groupby("method")[list(num)].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    metrics.to_csv(out / "metrics_by_split.csv", index=False)
    summary.to_csv(out / "metrics_summary.csv")
    pred.to_csv(out / "test_predictions.csv", index=False)
    sweep.to_csv(out / "noise_sweep.csv", index=False)
    strata.to_csv(out / "strata.csv", index=False)
    hist.to_csv(out / "training_history.csv", index=False)
    plot_history(hist, out / "loss_curves.png")
    res = dict(config=cfg, noise=noise.to_dict(), n_records=len(ds), n_lesion=int(ds.lesion_present.sum()),
               splits=[n for n, _, _ in splits],
               summary={k: {c: float(v) for c, v in r.items()} for k, r in summary.iterrows()})
    if final:
        log("[final] training ensemble on all records")
        dirs = anatomy_directions(ds.Z, ds.composition, ds.baselines)
        ens = train_ensemble(ds, cfg, noise, dirs, cfg["seed"] + 99, log)
        save_ensemble(ens, out / "model.pt", cfg, noise, ds)
        fh = pd.concat([m.history.assign(split="final", member=i, val_composition=m.val_composition,
                                         temperature=m.temperature) for i, m in enumerate(ens.members)])
        fh.to_csv(out / "final_training_history.csv", index=False)
        plot_history(fh, out / "final_loss_curves.png")
    (out / "results.json").write_text(json.dumps(res, indent=1, default=float))
    return res
