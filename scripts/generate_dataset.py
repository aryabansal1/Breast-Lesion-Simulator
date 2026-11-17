"""Generate a lesion bioimpedance dataset from a sweep spec (e.g. configs/dataset_v1.yaml).

One job = one (composition, lesion shape, location). It meshes the breast once (local refinement at the
lesion), then solves the lesion-absent case and one lesion-present case per lesion material on that same
mesh at every frequency. Jobs run in parallel, one process per job (one BLAS thread each) and are cached in
<out_dir>/jobs/, so an interrupted sweep resumes where it stopped.

Usage: python scripts/generate_dataset.py configs/dataset_v1.yaml --workers 6 [--limit N] [--only dense]
       python scripts/generate_dataset.py configs/dataset_v1.yaml --assemble-only
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import copy  # noqa: E402
import datetime as dt  # noqa: E402
import functools  # noqa: E402
import hashlib  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from breastsim import __version__  # noqa: E402
from breastsim.anatomy import TUMOUR, Anatomy  # noqa: E402
from breastsim.electrodes import contact_impedance  # noqa: E402
from breastsim.fem import CEMSystem  # noqa: E402
from breastsim.materials import EPS0, MaterialLibrary  # noqa: E402
from breastsim.simulate import git_commit, measurement_list, prepare_side  # noqa: E402

COST = {"fatty": 4, "scattered": 3, "hetero": 2, "dense": 1}  # rough relative cost, largest first


def _fmt(x):
    return f"{float(x):g}".replace(".", "p")


def make_jobs(spec):
    loc = spec["locations"]
    out = []
    for comp, shape, ch, s, d in itertools.product(spec["compositions"], spec["lesion_shapes"],
                                                   loc["clock_hour"], loc["s"], loc["depth_mm"]):
        out.append(dict(geometry_id=f"{comp}__{shape}__c{_fmt(ch)}_s{_fmt(s)}_d{_fmt(d)}", composition=comp,
                        shape_id=shape, clock_hour=float(ch), s=float(s), depth_mm=float(d)))
    return out


def job_config(spec, job):
    cfg = copy.deepcopy(spec["base"])
    cfg["anatomy"] = dict(type="victre_npz", path=str(ROOT / spec["compositions"][job["composition"]]),
                          side="left", bilateral=False)
    t = copy.deepcopy(spec["lesion_shapes"][job["shape_id"]])
    if "path" in t:
        t["path"] = str(ROOT / t["path"])
    t["placement"] = dict(clock_hour=job["clock_hour"], s=job["s"], depth_mm=job["depth_mm"])
    t["electrode_margin_mm"] = float(spec.get("electrode_margin_mm", 1.0))
    cfg["tumour"] = t
    cfg["materials"]["library"] = str(ROOT / cfg["materials"]["library"])
    return cfg


@functools.lru_cache
def _library_sha(path):
    return MaterialLibrary.load(path).sha256


@functools.lru_cache
def _file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def job_fingerprint(spec, job):
    """Hash of every input that determines a job's result, so a cached job from a changed sweep is re-run."""
    shape = spec["lesion_shapes"][job["shape_id"]]
    key = dict(job=job, base=spec["base"], anatomy=spec["compositions"][job["composition"]], shape=shape,
               lesion_materials=list(spec["lesion_materials"]),
               electrode_margin_mm=spec.get("electrode_margin_mm", 1.0),
               library_sha256=_library_sha(str(ROOT / spec["base"]["materials"]["library"])),
               anatomy_sha256=_file_sha(str(ROOT / spec["compositions"][job["composition"]])),
               lesion_file_sha256=_file_sha(str(ROOT / shape["path"])) if "path" in shape else None)
    return hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()


def record_id(gid, material):
    return f"{gid}__{material or 'none'}"


def write_cache(path, out):
    tmp = path.with_name(path.name + ".tmp")  # write-then-rename: an interrupted write never leaves a partial cache
    tmp.write_text(json.dumps(out, default=lambda o: o.item() if isinstance(o, np.generic) else str(o)))
    os.replace(tmp, path)


def read_cache(path, fingerprint):
    try:
        r = json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return r if r.get("fingerprint") == fingerprint else None


def run_job(spec, job, cache_dir):
    path = Path(cache_dir) / f"{job['geometry_id']}.json"
    fp = job_fingerprint(spec, job)
    cached = read_cache(path, fp)
    if cached is not None:
        return cached
    t0 = time.perf_counter()
    cfg = job_config(spec, job)
    lib = MaterialLibrary.load(cfg["materials"]["library"])
    anat = Anatomy.from_npz(cfg["anatomy"]["path"], side="left")
    out = dict(job=job, status="ok", fingerprint=fp)
    try:
        am, els, cases, trep, mesh_time, _ = prepare_side(cfg, anat, lib, lambda *a: None)
    except ValueError as e:  # lesion rejected (touches skin/air/boundary or too close to an electrode)
        out.update(status="rejected", reason=str(e), time_s=time.perf_counter() - t0)
        write_cache(path, out)
        return out
    ids = [e.id for e in els]
    meas = measurement_list(cfg, ids)
    current = float(cfg["measurements"]["current_a"])
    pairs = sorted({tuple(m["inject"]) for m in meas})
    inj = np.zeros((len(els), len(pairs)), complex)
    for p, (a, b) in enumerate(pairs):
        inj[ids.index(a), p] += current
        inj[ids.index(b), p] -= current
    mcfg = cfg["materials"]
    states = [(None, cases["absent"])] + [(m, cases["present"]) for m in spec["lesion_materials"]]
    z_rows, max_res, extrap_all, perm = [], 0.0, set(), None
    for f in [float(x) for x in cfg["frequencies_hz"]]:
        zc = []
        for it, e in zip(cfg["electrodes"]["items"], els):
            zs = contact_impedance(it.get("contact", cfg["electrodes"]["contact"]), f)
            zc.append(zs * e.area_m2 / e.projected_area_m2 if e.projected_area_m2 > 0 else zs)
        for mat, labels in states:
            assignment = lib.resolve_assignment(mcfg["assignment"], {TUMOUR: mat} if mat else {})
            gam, extrap = lib.label_admittivity(assignment, f, policy=mcfg.get("extrapolation", "warn"))
            extrap_all |= set(extrap)
            lut = np.zeros(int(labels.max()) + 1, complex)
            for lab, g in gam.items():
                if lab < len(lut):
                    lut[lab] = g
            sysm = CEMSystem(am.mesh.nodes, am.mesh.tets, lut[labels], [e.facets for e in els], zc)
            sol = sysm.factorize(perm).solve(inj)
            perm = sysm.perm  # same mesh -> same sparsity pattern for every state and frequency
            max_res = max(max_res, float(np.max(sol.diagnostics["relative_residual"])))
            for m in meas:
                U = sol.U[:, pairs.index(tuple(m["inject"]))]
                z = (U[ids.index(m["sense"][0])] - U[ids.index(m["sense"][1])]) / current
                z_rows.append([record_id(job["geometry_id"], mat), f, m["id"], z.real, z.imag])
    out.update(
        measurements=[dict(id=m["id"], type=m["type"], inject=m["inject"], sense=m["sense"]) for m in meas],
        z=z_rows, current_a=current, lesion=trep, mesh=am.stats, mesh_time_s=mesh_time,
        electrodes=[dict(id=e.id, clock_hour=e.clock_hour, s=e.s, center_mm=e.center_mm.tolist(),
                         projected_area_mm2=e.projected_area_m2 * 1e6, meshed_area_mm2=e.area_m2 * 1e6)
                    for e in els],
        max_relative_residual=max_res, extrapolated_materials=sorted(extrap_all), time_s=time.perf_counter() - t0)
    write_cache(path, out)
    return out


def detectability_summary(eff, rec, md, thresholds=(0.001, 0.005, 0.01)):
    """Per composition x shape x depth: largest relative |Z| change over all tetrapolar measurements
    except the near-balanced across_vertical, and the fraction of lesion records exceeding each threshold."""
    tetra = md[(md.measurement_type == "tetrapolar") & (md.measurement_id != "across_vertical")].measurement_id
    e = eff[eff.measurement_id.isin(tetra)].assign(a=lambda d: d.rel_d_mag.abs())
    best = e.groupby("record_id").a.max().rename("max_rel_d_mag").reset_index().merge(rec, on="record_id")
    keys = ["composition", "lesion_shape_id", "lesion_depth_mm"]
    out = best.groupby(keys).max_rel_d_mag.agg(n="size", median="median", min="min", max="max")
    for t in thresholds:
        out[f"frac_above_{t * 100:g}pct"] = best.assign(x=best.max_rel_d_mag > t).groupby(keys).x.mean()
    return out.reset_index()


def assemble(spec, out_dir, results, n_planned=None):
    lib = MaterialLibrary.load(ROOT / spec["base"]["materials"]["library"])
    shapes, classes = spec["lesion_shapes"], spec["lesion_materials"]
    records, zrows, failures, electrodes, meas_defs = [], [], [], [], {}
    for r in sorted(results, key=lambda r: r["job"]["geometry_id"]):
        j = r["job"]
        geom = dict(geometry_id=j["geometry_id"], composition=j["composition"], lesion_shape_id=j["shape_id"],
                    lesion_shape_type=shapes[j["shape_id"]]["shape"], lesion_clock_hour=j["clock_hour"],
                    lesion_s=j["s"], lesion_depth_mm=j["depth_mm"])
        if r["status"] != "ok":
            failures.append(dict(**geom, reason=r["reason"]))
            continue
        L = r["lesion"]
        vol = L["nominal_volume_mm3"]
        rep = L["replaced_tissue_mm3_by_label"]
        tot = sum(rep.values())
        lesion_geom = dict(
            lesion_center_x_mm=L["center_mm"][0], lesion_center_y_mm=L["center_mm"][1],
            lesion_center_z_mm=L["center_mm"][2], lesion_nominal_volume_mm3=vol,
            lesion_equiv_diameter_mm=(6 * vol / np.pi) ** (1 / 3), lesion_meshed_volume_mm3=L["meshed_volume_mm3"],
            lesion_min_distance_to_electrode_mm=L["min_distance_to_electrode_mm"],
            lesion_replaced_glandular_fraction=sum(v for k, v in rep.items() if int(k) in (29, 95, 125)) / tot,
            lesion_replaced_fat_fraction=rep.get("1", rep.get(1, 0.0)) / tot,
            lesion_warnings=";".join(L["warnings"]))
        base = record_id(j["geometry_id"], None)
        common = dict(mesh_nodes=r["mesh"]["nodes"], mesh_tets=r["mesh"]["tets"],
                      max_relative_residual=r["max_relative_residual"],
                      extrapolated_materials=";".join(r["extrapolated_materials"]))
        records.append(dict(record_id=base, **geom, lesion_present=False, lesion_material="", lesion_class="none",
                            baseline_record_id=base, **{k: np.nan for k in lesion_geom if k != "lesion_warnings"},
                            lesion_warnings="", **common))
        for mat in spec["lesion_materials"]:
            records.append(dict(record_id=record_id(j["geometry_id"], mat), **geom, lesion_present=True,
                                lesion_material=mat, lesion_class=classes[mat], baseline_record_id=base,
                                **lesion_geom, **common))
        zrows += r["z"]
        for m in r["measurements"]:
            meas_defs[m["id"]] = m
        for e in r["electrodes"]:
            electrodes.append(dict(geometry_id=j["geometry_id"], composition=j["composition"], electrode_id=e["id"],
                                   clock_hour=e["clock_hour"], s=e["s"], center_x_mm=e["center_mm"][0],
                                   center_y_mm=e["center_mm"][1], center_z_mm=e["center_mm"][2],
                                   projected_area_mm2=e["projected_area_mm2"], meshed_area_mm2=e["meshed_area_mm2"]))
    if not records:
        raise SystemExit(f"no successful jobs to assemble ({len(failures)} rejected)")
    rec = pd.DataFrame(records)
    md = pd.DataFrame([dict(measurement_id=k, measurement_type=v["type"], inject_pos=v["inject"][0],
                            inject_neg=v["inject"][1], sense_pos=v["sense"][0], sense_neg=v["sense"][1])
                       for k, v in meas_defs.items()])
    z = pd.DataFrame(zrows, columns=["record_id", "frequency_hz", "measurement_id", "z_re_ohm", "z_im_ohm"])
    zc = z.z_re_ohm + 1j * z.z_im_ohm
    z["z_mag_ohm"], z["z_phase_deg"] = np.abs(zc), np.degrees(np.angle(zc))
    out_dir.mkdir(parents=True, exist_ok=True)
    rec.to_csv(out_dir / "records.csv", index=False, float_format="%.6g")
    md.to_csv(out_dir / "measurement_definitions.csv", index=False)
    z.to_csv(out_dir / "measurements.csv", index=False, float_format="%.9g")
    pd.DataFrame(electrodes).to_csv(out_dir / "electrodes.csv", index=False, float_format="%.6g")
    pd.DataFrame(failures, columns=["geometry_id", "composition", "lesion_shape_id", "lesion_shape_type",
                                    "lesion_clock_hour", "lesion_s", "lesion_depth_mm", "reason"]
                 ).to_csv(out_dir / "rejected_geometries.csv", index=False)
    # lesion effect relative to the lesion-free baseline on the same mesh
    zi = z.set_index(["record_id", "frequency_hz", "measurement_id"])
    les = rec[rec.lesion_present][["record_id", "baseline_record_id"]]
    eff = []
    for rid, bid in les.itertuples(index=False):
        p, a = zi.loc[rid], zi.loc[bid]
        zp = (p.z_re_ohm + 1j * p.z_im_ohm).to_numpy()
        za = (a.z_re_ohm + 1j * a.z_im_ohm).reindex(p.index).to_numpy()
        eff.append(pd.DataFrame(dict(
            record_id=rid, baseline_record_id=bid, frequency_hz=p.index.get_level_values(0),
            measurement_id=p.index.get_level_values(1), dz_re_ohm=(zp - za).real, dz_im_ohm=(zp - za).imag,
            d_mag_ohm=np.abs(zp) - np.abs(za), rel_d_mag=np.abs(zp) / np.abs(za) - 1,
            d_phase_deg=np.degrees(np.angle(zp) - np.angle(za)))))
    if eff:
        eff = pd.concat(eff, ignore_index=True)
        eff.to_csv(out_dir / "lesion_effect.csv", index=False, float_format="%.6g")
        detectability_summary(eff, rec, md).to_csv(out_dir / "detectability_summary.csv", index=False,
                                                   float_format="%.4g")
    freqs = sorted(z.frequency_hz.unique())
    mids = list(md.measurement_id)
    Z = np.full((len(rec), len(freqs), len(mids)), np.nan + 0j)
    ri = {k: i for i, k in enumerate(rec.record_id)}
    Z[z.record_id.map(ri).to_numpy(), np.searchsorted(freqs, z.frequency_hz), z.measurement_id.map(
        {k: i for i, k in enumerate(mids)}).to_numpy()] = zc.to_numpy()
    np.savez_compressed(out_dir / "tensors.npz", Z=Z, record_id=rec.record_id.to_numpy(str),
                        baseline_index=rec.baseline_record_id.map(ri).to_numpy(),
                        lesion_present=rec.lesion_present.to_numpy(bool), composition=rec.composition.to_numpy(str),
                        frequency_hz=np.array(freqs), measurement_id=np.array(mids))
    mats = sorted(set(a for a in lib.resolve_assignment(spec["base"]["materials"]["assignment"]).values())
                  | set(spec["lesion_materials"]))
    mrows = []
    for f in freqs:
        for m in mats:
            g, ex = lib.materials[m].admittivity(f, policy="allow")
            g = complex(np.ravel(g)[0])
            mrows.append(dict(material=m, frequency_hz=f, sigma_s_per_m=g.real, eps_r=g.imag / (2 * np.pi * f * EPS0),
                              extrapolated=bool(np.ravel(ex)[0])))
    pd.DataFrame(mrows).to_csv(out_dir / "materials.csv", index=False, float_format="%.6g")
    (out_dir / "sweep_config.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    (out_dir / "provenance.json").write_text(json.dumps(dict(
        breastsim_version=__version__, git_commit=git_commit(), created=dt.datetime.now().astimezone().isoformat(),
        material_library_sha256=lib.sha256, n_geometries=len(results), n_rejected=len(failures),
        n_not_run=(n_planned or len(results)) - len(results),
        n_records=len(rec), total_job_time_s=sum(r.get("time_s", 0.0) for r in results)), indent=1))
    return rec, failures


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("spec")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", nargs="+", help="restrict to these compositions")
    ap.add_argument("--limit", type=int, help="run at most N new jobs (pilot)")
    ap.add_argument("-o", "--out", help="override out_dir")
    ap.add_argument("--assemble-only", action="store_true")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--job-timeout", type=float, default=5400, help="seconds before a stalled job is killed")
    ap.add_argument("--run-one", help=argparse.SUPPRESS)
    a = ap.parse_args()
    spec = yaml.safe_load(Path(a.spec).read_text())
    out_dir = ROOT / (a.out or spec["out_dir"])
    cache = out_dir / "jobs"
    cache.mkdir(parents=True, exist_ok=True)
    jobs = make_jobs(spec)
    if a.only:
        jobs = [j for j in jobs if j["composition"] in a.only]
    todo = [j for j in jobs if read_cache(cache / f"{j['geometry_id']}.json", job_fingerprint(spec, j)) is None]
    todo.sort(key=lambda j: -COST.get(j["composition"], 0))
    if a.limit is not None:
        todo = todo[:a.limit]
    if a.run_one:
        run_job(spec, next(j for j in jobs if j["geometry_id"] == a.run_one), cache)
        return
    if not a.assemble_only and todo:
        print(f"{len(jobs)} geometries, {len(todo)} to run, {a.workers} workers", flush=True)
        t0, done_n = time.perf_counter(), [0]

        def launch(j):
            # one process per job: a native crash (seen intermittently inside METIS) only loses that job
            for attempt in range(1, a.retries + 1):
                try:
                    p = subprocess.run([sys.executable, __file__, a.spec, "--run-one", j["geometry_id"],
                                        *(["-o", a.out] if a.out else [])], capture_output=True, text=True,
                                       timeout=a.job_timeout)
                except subprocess.TimeoutExpired:
                    print(f"{j['geometry_id']}: attempt {attempt} timed out after {a.job_timeout:.0f}s", flush=True)
                    continue
                r = read_cache(cache / f"{j['geometry_id']}.json", job_fingerprint(spec, j))
                if r is not None:
                    done_n[0] += 1
                    msg = r["status"] + ("" if r["status"] == "ok" else f" ({r['reason']})")
                    print(f"[{done_n[0]}/{len(todo)} {time.perf_counter() - t0:7.0f}s] {j['geometry_id']}: {msg} "
                          f"{r.get('time_s', 0):.0f}s", flush=True)
                    return True
                print(f"{j['geometry_id']}: attempt {attempt} failed (exit {p.returncode}): "
                      f"{next((ln for ln in p.stderr.splitlines() if 'Error' in ln), '')}", flush=True)
            return False

        with ThreadPoolExecutor(a.workers) as ex:
            ok = list(ex.map(launch, todo))
        failed = [j["geometry_id"] for j, o in zip(todo, ok) if not o]
        if failed:
            raise SystemExit(f"{len(failed)} geometries failed all {a.retries} attempts; not assembling: "
                             + ", ".join(failed))
    done = [r for j in jobs
            if (r := read_cache(cache / f"{j['geometry_id']}.json", job_fingerprint(spec, j))) is not None]
    if len(done) < len(jobs):
        print(f"warning: {len(jobs) - len(done)} of {len(jobs)} geometries not run yet; assembling a partial dataset")
    rec, failures = assemble(spec, out_dir, done, n_planned=len(jobs))
    print(f"wrote {out_dir}: {len(rec)} records ({int(rec.lesion_present.sum()) if len(rec) else 0} with lesion), "
          f"{len(failures)} rejected geometries")


if __name__ == "__main__":
    main()
