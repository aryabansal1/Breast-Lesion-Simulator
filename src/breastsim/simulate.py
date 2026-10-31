"""Config-driven simulation runner: paired tumour-absent / tumour-present runs on an
identical mesh, optional bilateral (mirrored) models, CSV/JSON/VTU outputs."""

from __future__ import annotations

import copy
import datetime as _dt
import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import __version__
from .anatomy import Anatomy, build_mesh, synthetic_hemisphere
from .electrodes import contact_impedance, place_electrodes
from .fem import CEMSystem
from .materials import EPS0, MaterialLibrary
from .tumour import insert_tumour, make_tumour, resolve_center

REPO = Path(__file__).resolve().parents[2]


def load_config(path):
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    cfg["_config_path"] = str(path.resolve())
    return cfg


def resolve_path(p, cfg):
    p = Path(p)
    if p.is_absolute():
        return p
    for base in (Path(cfg.get("_config_path", ".")).parent, REPO, Path.cwd()):
        if (base / p).exists():
            return base / p
    return REPO / p


def make_anatomy(cfg) -> Anatomy:
    a = cfg["anatomy"]
    if a["type"] == "victre_npz":
        return Anatomy.from_npz(resolve_path(a["path"], cfg), side=a.get("side", "left"),
                                y_positive_is_patient_left=a.get("y_positive_is_patient_left", True))
    if a["type"] == "synthetic_hemisphere":
        kw = {k: v for k, v in a.items() if k not in ("type", "bilateral")}
        return synthetic_hemisphere(**kw)
    raise ValueError(f"unknown anatomy type {a['type']}")


def measurement_list(cfg, ids):
    m = cfg["measurements"]
    out = []
    for i, p in enumerate(m.get("pairs", [])):
        out.append(dict(id=p.get("id", f"M{i}"), inject=list(p["inject"]), sense=list(p["sense"])))
    pat = m.get("pattern")
    if pat == "adjacent":
        L = len(ids)
        for k in range(L):
            inj = [ids[k], ids[(k + 1) % L]]
            for j in range(L):
                sen = [ids[j], ids[(j + 1) % L]]
                if set(sen) & set(inj):
                    continue
                out.append(dict(id=f"adj_{inj[0]}{inj[1]}_{sen[0]}{sen[1]}", inject=inj, sense=sen))
    if not out:
        raise ValueError("no measurements configured")
    for mm in out:
        for e in mm["inject"] + mm["sense"]:
            if e not in ids:
                raise KeyError(f"measurement {mm['id']} references unknown electrode {e}")
        mm["type"] = "bipolar" if set(mm["inject"]) == set(mm["sense"]) else "tetrapolar"
    return out


def git_commit():
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def peak_rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e6 if sys.platform == "darwin" else r / 1e3


def prepare_side(cfg, anat, lib, log, center_override=None):
    """Mesh, electrodes and (optional) tumour labels for one breast."""
    mcfg = cfg.get("mesh", {})
    h = float(mcfg.get("h_mm", 2.0))
    cw = float(mcfg.get("chest_wall_mm", 20.0))
    ecfg = cfg["electrodes"]
    mirror = ecfg.get("mirror_clock_for_right", True)
    side = anat.side
    refine = [dict(r) for r in mcfg.get("refine", [])]
    tcfg = cfg.get("tumour")
    # A side not in tumour.sides gets only the "absent" case, but keeps the same local refinement at the
    # mirrored position so left/right meshes stay comparable.
    insert = bool(tcfg) and side in tcfg.get("sides", ["left", "right"])
    center = None
    if tcfg:
        am0 = build_mesh(anat, h_mm=h, chest_wall_mm=cw)
        if center_override is not None:
            center, place_info = np.asarray(center_override, float), {"placement": "mirrored from contralateral"}
        else:
            center, place_info = resolve_center(tcfg, am0, side, mirror)
        tum0 = make_tumour(tcfg, center)
        if mcfg.get("tumour_h_mm"):
            refine.append(dict(center_mm=center.tolist(), radius_mm=tum0.bounding_radius_mm + h,
                               h_mm=float(mcfg["tumour_h_mm"])))
    t0 = time.perf_counter()
    am = build_mesh(anat, h_mm=h, refine=refine, chest_wall_mm=cw, grading=mcfg.get("grading", 0.4))
    mesh_time = time.perf_counter() - t0
    els = place_electrodes(am.mesh.nodes, am.skin_facets, am.landmarks, ecfg["items"], side, mirror)
    cases = {"absent": am.mesh.labels.copy()}
    trep = None
    if insert:
        tum = make_tumour(tcfg, center)
        if "material" in tcfg:
            tum.label = int(tcfg.get("label", 200))
        lab, trep = insert_tumour(am, tum, els, margin_mm=tcfg.get("electrode_margin_mm", 1.0))
        trep.update(place_info)
        cases["present"] = lab
    log(f"[{side}] mesh {am.stats['nodes']} nodes / {am.stats['tets']} tets in {mesh_time:.2f}s")
    return am, els, cases, trep, mesh_time, center


def run(cfg, out_dir=None, log=print, write_fields=True):
    cfg = copy.deepcopy(cfg)
    t_start = time.perf_counter()
    lib = MaterialLibrary.load(resolve_path(cfg["materials"].get("library", "data/materials/library.yaml"), cfg))
    mcfg = cfg["materials"]
    tcfg = cfg.get("tumour") or {}
    overrides = dict(mcfg.get("overrides", {}))
    if tcfg.get("material"):
        overrides[int(tcfg.get("label", 200))] = tcfg["material"]
    assignment = lib.resolve_assignment(mcfg.get("assignment", "uci_breast"), overrides)
    policy = mcfg.get("extrapolation", "warn")
    scales = mcfg.get("scales", {})
    freqs = [float(f) for f in cfg["frequencies_hz"]]
    current = float(cfg["measurements"].get("current_a", 1e-4))
    base = make_anatomy(cfg)
    anats = [base]
    if cfg["anatomy"].get("bilateral"):
        anats.append(base.mirror())
    rows, meta_sides, diags = [], {}, []
    fields_out = {}
    base_center = None
    for anat in anats:
        side = anat.side
        # The mirrored breast reuses the exactly mirrored tumour centre so both meshes stay mirror images.
        override = anat.mirror_point(base_center) if (anat.mirrored and base_center is not None) else None
        am, els, cases, trep, mesh_time, center = prepare_side(cfg, anat, lib, log, override)
        if not anat.mirrored:
            base_center = center
        ids = [e.id for e in els]
        meas = measurement_list(cfg, ids)
        pairs = sorted({tuple(m["inject"]) for m in meas})
        inj = np.zeros((len(els), len(pairs)), complex)
        for p, (a, b) in enumerate(pairs):
            inj[ids.index(a), p] += current
            inj[ids.index(b), p] -= current
        missing = sorted(set(np.unique(np.concatenate(list(cases.values())))) - set(assignment))
        if missing:
            raise KeyError(f"labels {missing} present in the mesh have no material assignment")
        mat_table = []
        for f in freqs:
            gam, extrap = lib.label_admittivity(assignment, f, policy=policy, scales=scales)
            for lab_, g in gam.items():
                mat_table.append(dict(frequency_hz=f, label=lab_, label_name=lib.label_names.get(lab_, ""),
                                      material=assignment[lab_], sigma_s_per_m=g.real,
                                      eps_r=g.imag / (2 * np.pi * f * EPS0), extrapolated=assignment[lab_] in extrap))
            # Specific contact impedance z_s applies to the smooth contact area A_p; the FE surface
            # has area A_m >= A_p, so z_eff = z_s * A_m / A_p keeps the total contact admittance A_p / z_s.
            correct = cfg["electrodes"].get("stair_step_area_correction", True)
            z = []
            for it, e in zip(cfg["electrodes"]["items"], els):
                zs = contact_impedance(it.get("contact", cfg["electrodes"]["contact"]), f)
                if correct and e.projected_area_m2 > 0:
                    zs = zs * e.area_m2 / e.projected_area_m2
                z.append(zs)
            for case, labels in cases.items():
                lut = np.zeros(int(labels.max()) + 1, complex)
                for lab_, g in gam.items():
                    if lab_ < len(lut):
                        lut[lab_] = g
                gamma_e = lut[labels]
                scfg = cfg.get("solver", {})
                sysm = CEMSystem(am.mesh.nodes, am.mesh.tets, gamma_e, [e.facets for e in els], z,
                                 method=scfg.get("method", "direct"), tol=float(scfg.get("tol", 1e-10)),
                                 maxiter=int(scfg.get("maxiter", 2000))).factorize()
                sol = sysm.solve(inj)
                d = dict(side=side, case=case, frequency_hz=f, **sol.diagnostics)
                diags.append(d)
                for m in meas:
                    p = pairs.index(tuple(m["inject"]))
                    U = sol.U[:, p]
                    v = U[ids.index(m["sense"][0])] - U[ids.index(m["sense"][1])]
                    zt = v / current
                    rows.append(dict(side=side, case=case, tumour_present=case == "present", frequency_hz=f,
                                     measurement_id=m["id"], measurement_type=m["type"],
                                     inject_pos=m["inject"][0], inject_neg=m["inject"][1],
                                     sense_pos=m["sense"][0], sense_neg=m["sense"][1], current_a=current,
                                     v_re=v.real, v_im=v.imag, z_re=zt.real, z_im=zt.imag, z_mag_ohm=abs(zt),
                                     z_phase_deg=np.degrees(np.angle(zt))))
                fcfg = cfg.get("fields")
                if write_fields and fcfg and np.isclose(f, float(fcfg["frequency_hz"])):
                    mid = fcfg.get("measurement", meas[0]["id"])
                    m = next(mm for mm in meas if mm["id"] == mid)
                    p = pairs.index(tuple(m["inject"]))
                    fields_out[(side, case)] = (am, sysm, sol.u[:, p], labels, els, gamma_e, mid, f)
        meta_sides[side] = dict(
            mirrored=anat.mirrored, anatomy_notes=anat.notes, mesh=am.stats, mesh_time_s=mesh_time,
            landmarks={k: np.asarray(v).tolist() for k, v in am.landmarks.items()},
            electrodes=[dict(id=e.id, clock_hour=e.clock_hour, s=e.s, center_mm=e.center_mm.tolist(),
                             meshed_area_mm2=e.area_m2 * 1e6, nominal_area_mm2=e.nominal_area_m2 * 1e6,
                             projected_area_mm2=e.projected_area_m2 * 1e6,
                             n_facets=int(len(e.facets))) for e in els],
            tumour=trep, materials=mat_table, measurements=meas)
    df = pd.DataFrame(rows)
    res = dict(measurements=df, diff=tumour_diff(df), lr=left_right_diff(df), diagnostics=pd.DataFrame(diags))
    meta = dict(breastsim_version=__version__, git_commit=git_commit(),
                timestamp=_dt.datetime.now().astimezone().isoformat(), platform=platform.platform(),
                machine=platform.machine(), python=platform.python_version(),
                config={k: v for k, v in cfg.items() if not k.startswith("_")},
                config_path=cfg.get("_config_path"), material_library=dict(path=lib.path, sha256=lib.sha256,
                                                                          version=lib.version),
                assignment={int(k): v for k, v in assignment.items()}, sides=meta_sides,
                total_time_s=time.perf_counter() - t_start, peak_rss_mb=peak_rss_mb(),
                phasor_convention="v(t) = Re{V exp(+j omega t)}", z_definition="Z = (U_sense_pos - U_sense_neg) / I")
    res["metadata"] = meta
    if out_dir:
        write_outputs(res, fields_out, Path(out_dir), log)
    return res


def tumour_diff(df):
    if df.empty or not (df.case == "present").any():
        return pd.DataFrame()
    k = ["side", "frequency_hz", "measurement_id"]
    p0 = df[df.case == "present"].set_index(k)
    a = df[df.case == "absent"].set_index(k).loc[p0.index]
    p = p0.reindex(a.index)
    za = (a.z_re + 1j * a.z_im).to_numpy()
    zp = (p.z_re + 1j * p.z_im).to_numpy()
    dz = zp - za
    out = pd.DataFrame(index=a.index, data=dict(
        z_absent_re=za.real, z_absent_im=za.imag, z_present_re=zp.real, z_present_im=zp.imag,
                            dz_re=dz.real, dz_im=dz.imag, dz_abs=np.abs(dz), d_mag_ohm=np.abs(zp) - np.abs(za),
                            rel_d_mag=np.abs(zp) / np.abs(za) - 1,
                            d_phase_deg=np.degrees(np.angle(zp) - np.angle(za))))
    return out.reset_index()


def left_right_diff(df):
    if df.empty or df.side.nunique() < 2:
        return pd.DataFrame()
    # Compare like-for-like cases; a side without a tumour contributes its "absent" case to every
    # comparison (e.g. left tumour vs healthy right).
    k = ["frequency_hz", "measurement_id"]
    out = []
    for case in ("absent", "present"):
        lt = df[(df.side == "left") & (df.case == case)]
        rt = df[(df.side == "right") & (df.case == case)]
        if lt.empty and rt.empty:
            continue
        lcase, rcase = (case if not lt.empty else "absent"), (case if not rt.empty else "absent")
        lt = df[(df.side == "left") & (df.case == lcase)].set_index(k)
        rt = df[(df.side == "right") & (df.case == rcase)].set_index(k).reindex(lt.index)
        zl = (lt.z_re + 1j * lt.z_im).to_numpy()
        zr = (rt.z_re + 1j * rt.z_im).to_numpy()
        d = zr - zl
        out.append(pd.DataFrame(index=lt.index, data=dict(
            case=case, left_case=lcase, right_case=rcase,
            z_left_re=zl.real, z_left_im=zl.imag, z_right_re=zr.real, z_right_im=zr.imag,
            dz_re=d.real, dz_im=d.imag, rel_d_mag=np.abs(zr) / np.abs(zl) - 1)).reset_index())
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def write_outputs(res, fields_out, out, log):
    out.mkdir(parents=True, exist_ok=True)
    res["measurements"].to_csv(out / "measurements.csv", index=False, float_format="%.9g")
    if not res["diff"].empty:
        res["diff"].to_csv(out / "tumour_difference.csv", index=False, float_format="%.9g")
    if not res["lr"].empty:
        res["lr"].to_csv(out / "left_right_difference.csv", index=False, float_format="%.9g")
    res["diagnostics"].to_csv(out / "solver_diagnostics.csv", index=False, float_format="%.6g")
    (out / "metadata.json").write_text(json.dumps(res["metadata"], indent=1, default=_json_default))
    (out / "config_resolved.yaml").write_text(yaml.safe_dump(res["metadata"]["config"], sort_keys=False))
    from .io import write_fields_vtu

    for (side, case), args in fields_out.items():
        write_fields_vtu(out / f"fields_{side}_{case}.vtu", out / f"electrodes_{side}.vtu", *args)
    log(f"wrote outputs to {out}")


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, complex):
        return [o.real, o.imag]
    raise TypeError(type(o))


def set_dotted(cfg, path, value):
    keys = path.split(".")
    d = cfg
    for k in keys[:-1]:
        if isinstance(d, list):
            d = d[int(k)]
        else:
            d = d.setdefault(k, {})
    last = keys[-1]
    if isinstance(d, list):
        d[int(last)] = value
    else:
        d[last] = value


def sensitivity(cfg, out_dir, log=print):
    """One-at-a-time sensitivity study defined in cfg['sensitivity']['parameters']."""
    out_dir = Path(out_dir)
    base_res = run(cfg, None, log, write_fields=False)
    summ = []

    def collect(res, param, value):
        d = res["diff"] if not res["diff"].empty else res["measurements"]
        for _, r in d.iterrows():
            summ.append(dict(parameter=param, value=str(value), **r.to_dict()))

    collect(base_res, "baseline", "-")
    for p in cfg["sensitivity"]["parameters"]:
        for v in p["values"]:
            c = copy.deepcopy(cfg)
            set_dotted(c, p["path"], v)
            log(f"sensitivity: {p['path']} = {v}")
            collect(run(c, None, log, write_fields=False), p["path"], v)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(summ)
    df.to_csv(out_dir / "sensitivity.csv", index=False, float_format="%.9g")
    return df
