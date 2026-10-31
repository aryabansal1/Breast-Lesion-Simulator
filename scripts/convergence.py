"""Mesh-convergence study for a paired (tumour absent/present) configuration.

Usage: python scripts/convergence.py configs/victre_dense_paired.yaml -o outputs/convergence \
           --levels 3:1.5 2.5:1.25 2:1 1.5:0.75 --tumour-levels 2:2 2:0.5
Each level "h:tumour_h" runs in a fresh subprocess (one at a time: BLAS threads). Results are compared
with the finest level of the main sweep (smallest h). Writes convergence.csv, convergence_summary.csv
and convergence.png into the output directory.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

CASE = r"""
import json, resource, sys
sys.path.insert(0, "{root}/src")
from breastsim.simulate import load_config, run
cfg = load_config("{config}")
cfg["mesh"]["h_mm"] = {h}
cfg["mesh"]["tumour_h_mm"] = {th}
cfg.pop("fields", None)
res = run(cfg, "{out}", log=lambda *a: None, write_fields=False)
m = res["metadata"]["sides"]
print(json.dumps(dict(nodes=sum(s["mesh"]["nodes"] for s in m.values()),
                      tets=sum(s["mesh"]["tets"] for s in m.values()),
                      peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)))
"""


def run_level(config, out, h, th):
    d = Path(out) / f"h{h:g}_t{th:g}"
    t = time.perf_counter()
    r = subprocess.run([sys.executable, "-c", CASE.format(root=ROOT, config=Path(config).resolve(), h=h, th=th,
                                                          out=d)], capture_output=True, text=True, cwd=ROOT)
    if r.returncode:
        raise RuntimeError(r.stderr[-3000:])
    info = json.loads(r.stdout.strip().splitlines()[-1])
    info.update(h_mm=h, tumour_h_mm=th, runtime_s=time.perf_counter() - t, dir=str(d))
    print(f"h={h:g} tumour_h={th:g}: {info['nodes']} nodes, {info['tets']} tets, "
          f"{info['runtime_s']:.0f} s, {info['peak_rss_mb']:.0f} MB", flush=True)
    return info


def load_level(info):
    d = Path(info["dir"])
    td = pd.read_csv(d / "tumour_difference.csv")
    td["z_absent_mag"] = np.hypot(td.z_absent_re, td.z_absent_im)
    td["z_present_mag"] = np.hypot(td.z_present_re, td.z_present_im)
    td["rel_d_mag_pct"] = 100 * td.rel_d_mag
    for k in ("h_mm", "tumour_h_mm", "nodes", "tets", "runtime_s", "peak_rss_mb"):
        td[k] = info[k]
    return td


def parse(levels):
    return [tuple(float(x) for x in s.split(":")) for s in levels]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("-o", "--out", default=str(ROOT / "outputs/convergence"))
    ap.add_argument("--levels", nargs="+", default=["3:1.5", "2.5:1.25", "2:1", "1.5:0.75"])
    ap.add_argument("--tumour-levels", nargs="*", default=["2:2", "2:0.5"])
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    main_lv, tum_lv = parse(a.levels), parse(a.tumour_levels)
    infos, seen = [], set()
    for sweep, lv in (("global", main_lv), ("tumour", tum_lv)):
        for h, th in lv:
            if (h, th) in seen:
                continue
            seen.add((h, th))
            info = run_level(a.config, out, h, th)
            info["sweep"] = sweep
            infos.append(info)
            pd.DataFrame(infos).to_csv(out / "levels.csv", index=False)
    df = pd.concat([load_level(i).assign(sweep=i["sweep"]) for i in infos], ignore_index=True)
    k = ["side", "frequency_hz", "measurement_id"]
    ref_h, ref_th = min(main_lv)
    ref = df[(df.h_mm == ref_h) & (df.tumour_h_mm == ref_th)].set_index(k)
    df = df.join(ref[["z_absent_mag", "rel_d_mag_pct", "d_phase_deg"]], on=k, rsuffix="_ref")
    df["z_absent_err_pct"] = 100 * (df.z_absent_mag / df.z_absent_mag_ref - 1)
    df["rel_d_mag_err_pp"] = df.rel_d_mag_pct - df.rel_d_mag_pct_ref
    df["d_phase_err_deg"] = df.d_phase_deg - df.d_phase_deg_ref
    df.to_csv(out / "convergence.csv", index=False, float_format="%.9g")
    # tetrapolar |Z| near zero blows up relative errors; summarise over measurements with |Z| >= 1 ohm
    ok = df.z_absent_mag_ref >= 1.0
    g = df[ok].groupby(["sweep", "h_mm", "tumour_h_mm", "nodes", "tets", "runtime_s", "peak_rss_mb"])
    summ = g.agg(z_abs_err_pct_median=("z_absent_err_pct", lambda s: s.abs().median()),
                 z_abs_err_pct_max=("z_absent_err_pct", lambda s: s.abs().max()),
                 tumour_effect_err_pp_median=("rel_d_mag_err_pp", lambda s: s.abs().median()),
                 tumour_effect_err_pp_max=("rel_d_mag_err_pp", lambda s: s.abs().max()),
                 tumour_effect_ref_pct_max=("rel_d_mag_pct_ref", lambda s: s.abs().max()),
                 phase_effect_err_deg_max=("d_phase_err_deg", lambda s: s.abs().max())).reset_index()
    summ = summ.sort_values(["sweep", "h_mm", "tumour_h_mm"], ascending=[True, False, False])
    summ.to_csv(out / "convergence_summary.csv", index=False, float_format="%.4g")
    print(summ.to_string(index=False))
    plot(df, main_lv, out)


def plot(df, main_lv, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f0 = sorted(df.frequency_hz.unique())[len(df.frequency_hz.unique()) // 2]
    g = df[(df.sweep == "global") | df.set_index(["h_mm", "tumour_h_mm"]).index.isin(main_lv)]
    g = g[g.frequency_hz == f0]
    top = (g[g.h_mm == min(main_lv)[0]].assign(a=lambda d: d.rel_d_mag_pct.abs())
           .sort_values("a", ascending=False).measurement_id.head(6).tolist())
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for mid in top:
        s = g[g.measurement_id == mid].sort_values("h_mm")
        ax[0].plot(s.h_mm, s.z_absent_mag, "o-", label=mid)
        ax[1].plot(s.h_mm, s.rel_d_mag_pct, "o-", label=mid)
    ax[0].set(xlabel="global element size h (mm); tumour refinement h/2", ylabel="|Z| tumour absent (ohm)",
              title=f"Transfer impedance vs mesh size ({f0 / 1e3:g} kHz)")
    ax[1].set(xlabel="global element size h (mm); tumour refinement h/2", ylabel="tumour effect on |Z| (%)",
              title="Tumour effect vs mesh size")
    for x in ax:
        x.invert_xaxis()
        x.grid(alpha=0.3)
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "convergence.png", dpi=130)


if __name__ == "__main__":
    main()
