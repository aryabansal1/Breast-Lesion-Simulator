"""Runtime / memory benchmark: mesh size vs assembly, factorisation and solve time.

Usage: python scripts/benchmark.py [--anatomy data/anatomy/victre_dense_p_0p5mm.npz] [--h 4 3 2.5 2 1.5]
Each mesh size runs in a fresh subprocess so peak RSS is per-case. Writes docs/benchmarks.csv.
"""
import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

CASE = r"""
import json, resource, sys, time, numpy as np
sys.path.insert(0, "{root}/src")
from breastsim.anatomy import Anatomy, build_mesh
from breastsim.electrodes import place_electrodes
from breastsim.fem import CEMSystem, drive_pattern
from breastsim.materials import MaterialLibrary
an = Anatomy.from_npz("{anatomy}")
t = time.perf_counter(); am = build_mesh(an, h_mm={h}); t_mesh = time.perf_counter() - t
clocks = [12, 1.5, 3, 4.5, 6, 7.5, 9, 10.5]
specs = [dict(id=f"E{{i}}", clock_hour=c, s=0.55, radius_mm=5) for i, c in enumerate(clocks)]
els = place_electrodes(am.mesh.nodes, am.skin_facets, am.landmarks, specs)
lib = MaterialLibrary.load("{root}/data/materials/library.yaml")
gam, _ = lib.label_admittivity(lib.resolve_assignment("uci_breast", {{}}), 62500.0, policy="allow")
lut = np.zeros(256, complex)
for k, v in gam.items(): lut[k] = v
t = time.perf_counter()
s = CEMSystem(am.mesh.nodes, am.mesh.tets, lut[am.mesh.labels], [e.facets for e in els], 5e-3)
t_asm = time.perf_counter() - t
s.factorize()
cur = np.stack([np.ravel(drive_pattern(8, i, (i + 1) % 8, 1e-4)) for i in range(8)], 1)
sol = s.solve(cur)
print(json.dumps(dict(h_mm={h}, nodes=am.stats["nodes"], tets=am.stats["tets"], unknowns=s.A.shape[0],
    mesh_s=t_mesh, assemble_s=t_asm, factor_s=s.factor_time, solve_8rhs_s=sol.diagnostics["solve_time_s"],
    lu_nnz=sol.diagnostics["lu_nnz"], max_rel_residual=float(np.max(sol.diagnostics["relative_residual"])),
    peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 if sys.platform != "darwin" else 1024**2))))
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anatomy", default=str(ROOT / "data/anatomy/victre_dense_p_0p5mm.npz"))
    ap.add_argument("--h", type=float, nargs="+", default=[4.0, 3.0, 2.5, 2.0, 1.5])
    ap.add_argument("--out", default=str(ROOT / "docs/benchmarks.csv"))
    a = ap.parse_args()
    rows = []
    for h in a.h:
        code = CASE.format(root=ROOT, anatomy=a.anatomy, h=h)
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        if r.returncode:
            print(r.stderr[-2000:])
            continue
        row = json.loads(r.stdout.strip().splitlines()[-1])
        row.update(machine=platform.machine(), platform=platform.platform())
        print(row)
        rows.append(row)
    pd.DataFrame(rows).to_csv(a.out, index=False, float_format="%.4g")


if __name__ == "__main__":
    main()
