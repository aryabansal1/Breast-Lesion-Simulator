"""Dataset sweep generator and gzip voxel lesion loading."""
import copy
import gzip
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from breastsim.tumour import make_tumour

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("generate_dataset", ROOT / "scripts" / "generate_dataset.py")
gd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gd)
SPEC = yaml.safe_load((ROOT / "configs" / "dataset_v1.yaml").read_text())


def test_gzip_voxel_matches_raw(tmp_path):
    m = np.zeros((4, 5, 6), np.uint8)
    m[1:3, 1:4, 2:5] = 1
    raw = tmp_path / "m.raw"
    m.transpose(2, 1, 0).tofile(raw)
    gz = tmp_path / "m.raw.gz"
    gz.write_bytes(gzip.compress(raw.read_bytes()))
    a = make_tumour(dict(shape="voxel", path=str(raw), dims=[4, 5, 6], voxel_mm=1.0), np.zeros(3))
    b = make_tumour(dict(shape="voxel", path=str(gz), dims=[4, 5, 6], voxel_mm=1.0), np.zeros(3))
    assert np.array_equal(a.mask, b.mask) and a.mask.sum() == 18


def test_job_matrix_covers_every_combination():
    jobs = gd.make_jobs(SPEC)
    loc = SPEC["locations"]
    n = len(SPEC["compositions"]) * len(SPEC["lesion_shapes"]) * len(loc["clock_hour"]) * len(loc["s"]) * len(
        loc["depth_mm"])
    assert len(jobs) == n and len({j["geometry_id"] for j in jobs}) == n


def test_tiny_sweep_records_baseline_and_lesions(tmp_path):
    spec = copy.deepcopy(SPEC)
    spec["base"]["mesh"].update(h_mm=6.0, tumour_h_mm=3.0)
    spec["base"]["frequencies_hz"] = [62500]
    spec["compositions"] = {"dense": SPEC["compositions"]["dense"]}
    spec["lesion_shapes"] = {"sphere_r7": SPEC["lesion_shapes"]["sphere_r7"]}
    spec["lesion_materials"] = {"carcinoma_uci": "malignant", "fibroadenoma_uci": "benign"}
    spec["locations"] = {"clock_hour": [1.5], "s": [0.55], "depth_mm": [12.0]}
    (job,) = gd.make_jobs(spec)
    cache = tmp_path / "jobs"
    cache.mkdir()
    r = gd.run_job(spec, job, cache)
    assert r["status"] == "ok" and r["max_relative_residual"] < 1e-8
    f = cache / f"{job['geometry_id']}.json"
    assert gd.read_cache(f, gd.job_fingerprint(spec, job)) is not None
    changed = copy.deepcopy(spec)
    changed["base"]["frequencies_hz"] = [125000]
    assert gd.read_cache(f, gd.job_fingerprint(changed, job)) is None
    bad = cache / "partial.json"
    bad.write_text("{")
    assert gd.read_cache(bad, gd.job_fingerprint(spec, job)) is None
    rec, failures = gd.assemble(spec, tmp_path, [r])
    assert not failures
    assert list(rec.lesion_present) == [False, True, True]
    assert (rec.baseline_record_id == rec.record_id[0]).all()
    z = pd.read_csv(tmp_path / "measurements.csv")
    assert len(z) == 3 * len(pd.read_csv(tmp_path / "measurement_definitions.csv"))
    eff = pd.read_csv(tmp_path / "lesion_effect.csv")
    assert (eff.d_mag_ohm.abs() > 0).any()
    assert len(pd.read_csv(tmp_path / "detectability_summary.csv")) == 1
    t = np.load(tmp_path / "tensors.npz")
    assert t["Z"].shape == (3, 1, 7) and np.isfinite(t["Z"]).all()
