"""Binary detection metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def sens_at_spec(y, p, spec):
    fpr, tpr, _ = roc_curve(y, p)
    ok = fpr <= 1 - spec + 1e-12
    return float(tpr[ok].max()) if ok.any() else 0.0


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def binary_metrics(y, score, prob=None):
    """AUROC/AUPRC/sens@spec from any monotone score; Brier/ECE/accuracy only when calibrated ``prob`` given."""
    y = np.asarray(y, float)
    out = dict(n=int(len(y)), n_pos=int(y.sum()))
    if 0 < y.sum() < len(y):
        out.update(auroc=float(roc_auc_score(y, score)), auprc=float(average_precision_score(y, score)),
                   sens_at_spec90=sens_at_spec(y, score, 0.90), sens_at_spec95=sens_at_spec(y, score, 0.95))
    if prob is not None:
        prob = np.asarray(prob, float)
        out.update(brier=float(np.mean((prob - y) ** 2)), ece=ece(y, prob),
                   accuracy=float(np.mean((prob >= 0.5) == y)),
                   sensitivity=float(np.mean(prob[y == 1] >= 0.5)) if y.sum() else float("nan"),
                   specificity=float(np.mean(prob[y == 0] < 0.5)) if (1 - y).sum() else float("nan"))
    return out
