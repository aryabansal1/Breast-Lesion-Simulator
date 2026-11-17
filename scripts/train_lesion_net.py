"""Train and evaluate the lesion-flag network (lesion present vs absent from left/right bioimpedance pairs).

Usage: python scripts/train_lesion_net.py configs/nn_v1.yaml [--data DIR] [-o OUT] [--quick] [--no-final]
       python scripts/train_lesion_net.py configs/nn_v1.yaml --synthetic -o outputs/nn_synthetic --quick
See docs/nn_design.md for every design decision.
"""
import argparse
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lesionnet.data import load_dataset, write_synthetic_dataset  # noqa: E402
from lesionnet.train import run  # noqa: E402

QUICK = dict(train=dict(ensemble=2, epochs=30, patience=10, scale_fit_rounds=5), eval=dict(test_draws=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("config")
    ap.add_argument("--data", help="dataset directory (overrides config 'data')")
    ap.add_argument("-o", "--out", help="output directory (overrides config 'out_dir')")
    ap.add_argument("--synthetic", action="store_true", help="train on a toy dataset in the same schema")
    ap.add_argument("--quick", action="store_true", help="tiny ensemble / few epochs (smoke test)")
    ap.add_argument("--no-final", action="store_true", help="skip the final all-data ensemble (model.pt)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--split", choices=["auto", "loco", "group_kfold"], help="override config 'split'")
    ap.add_argument("--iid-noise", type=float, metavar="SIGMA",
                    help="replace the nuisance model by i.i.d. complex measurement noise of rms SIGMA*|Z| per reading "
                         "and breast (no placement / contact / asymmetry terms, no jitter); disables the sweeps")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cfg = yaml.safe_load(Path(a.config).read_text()) or {}
    out = ROOT / (a.out or cfg.pop("out_dir", "outputs/nn"))
    cfg.pop("out_dir", None)
    data = cfg.pop("data", None)
    if a.synthetic:
        data = write_synthetic_dataset(out / "synthetic_data")
    data = Path(a.data or data)
    if not data.is_absolute():
        data = ROOT / data
    if a.split:
        cfg["split"] = a.split
    if a.iid_noise is not None:
        cfg["noise"] = dict(meas_sigma=a.iid_noise, placement_sigma=0.0, contact_sigma=0.0, anat_sigma=0.0,
                            scale_jitter=[1.0, 1.0])
        cfg.setdefault("eval", {}).update(noise_sweep=[1.0], anat_sweep=[])
    if a.quick:
        for k, v in QUICK.items():
            cfg.setdefault(k, {}).update(v)
    ds = load_dataset(data)
    print(f"loaded {data}: {len(ds)} records ({int(ds.lesion_present.sum())} lesion, "
          f"{len(ds.baselines)} lesion-free), compositions {sorted(set(ds.composition))}, "
          f"{len(ds.frequency_hz)} freqs x {len(ds.measurement_id)} measurements", flush=True)
    res = run(ds, cfg, out, final=not a.no_final, log=lambda s: print(s, flush=True))
    print("\nmean over splits:")
    for m, r in res["summary"].items():
        print(f"  {m:7s} AUROC {r.get('auroc_mean', float('nan')):.3f} +- {r.get('auroc_std', float('nan')):.3f}  "
              f"sens@95%spec {r.get('sens_at_spec95_mean', float('nan')):.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
