"""Command line interface: breastsim run|sensitivity|plot CONFIG/OUTDIR."""
import argparse
import json
from pathlib import Path

from .simulate import load_config, run, sensitivity


def main(argv=None):
    ap = argparse.ArgumentParser(prog="breastsim", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="paired tumour-absent / tumour-present simulation")
    r.add_argument("config")
    r.add_argument("-o", "--out", help="output directory (default outputs/<name>)")
    r.add_argument("--no-fields", action="store_true", help="skip VTU field export")
    r.add_argument("--plot", action="store_true", help="write spectra and 3D renders")
    s = sub.add_parser("sensitivity", help="one-at-a-time sensitivity study (config 'sensitivity' block)")
    s.add_argument("config")
    s.add_argument("-o", "--out")
    p = sub.add_parser("plot", help="plot an existing output directory")
    p.add_argument("out")
    p.add_argument("--render", action="store_true", help="also render VTU fields (needs pyvista)")
    a = ap.parse_args(argv)
    if a.cmd in ("run", "sensitivity"):
        cfg = load_config(a.config)
        out = Path(a.out or Path("outputs") / (cfg.get("name") or Path(a.config).stem))
        if a.cmd == "sensitivity":
            sensitivity(cfg, out)
            print(f"wrote {out / 'sensitivity.csv'}")
            return
        res = run(cfg, out, write_fields=not a.no_fields)
        m = res["metadata"]
        print(json.dumps(dict(total_time_s=round(m["total_time_s"], 2), peak_rss_mb=round(m["peak_rss_mb"]),
                              mesh={k: v["mesh"]["nodes"] for k, v in m["sides"].items()}), indent=1))
        if a.plot:
            _plot(out, render=not a.no_fields)
    elif a.cmd == "plot":
        _plot(Path(a.out), render=a.render)


def _plot(out, render):
    from .plotting import plot_spectra, render_fields

    for f in plot_spectra(out):
        print("wrote", f)
    if render:
        for vtu in sorted(out.glob("fields_*_*.vtu")):
            _, side, case = vtu.stem.split("_")
            print("wrote", render_fields(out, side, case))


if __name__ == "__main__":
    main()
