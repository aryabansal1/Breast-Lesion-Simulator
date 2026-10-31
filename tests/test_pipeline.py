"""Anatomy / electrode / tumour pipeline checks on a small synthetic breast."""
import copy
from pathlib import Path

import numpy as np
import pytest

from breastsim.anatomy import synthetic_hemisphere
from breastsim.simulate import load_config, run

CFG = Path(__file__).resolve().parents[1] / "configs" / "synthetic_quick.yaml"


@pytest.fixture(scope="module")
def cfg():
    c = load_config(CFG)
    c["mesh"]["h_mm"] = 5.0
    c["mesh"]["tumour_h_mm"] = 2.5
    c["frequencies_hz"] = [62500]
    return c


def test_zero_contrast_tumour_gives_no_change(cfg):
    """Tumour material equal to every tissue it replaces -> identical measurements."""
    c = copy.deepcopy(cfg)
    c["tumour"]["material"] = "glandular_uci"
    c["materials"]["overrides"] = {lab: "glandular_uci" for lab in (1, 2, 29, 33, 40, 41, 88, 95, 125, 150, 225, 250)}
    res = run(c, None, log=lambda *a: None, write_fields=False)
    assert res["metadata"]["sides"]["left"]["tumour"]["n_elements"] > 0
    assert np.max(np.abs(res["diff"].rel_d_mag)) < 1e-12


def test_tumour_changes_transfer_impedance_and_is_deterministic(cfg):
    a = run(copy.deepcopy(cfg), None, log=lambda *a: None, write_fields=False)
    b = run(copy.deepcopy(cfg), None, log=lambda *a: None, write_fields=False)
    assert np.max(np.abs(a["diff"].rel_d_mag)) > 1e-4
    assert np.array_equal(a["measurements"].z_re.values, b["measurements"].z_re.values)


def test_tumour_touching_boundary_is_rejected(cfg):
    c = copy.deepcopy(cfg)
    c["tumour"]["placement"]["depth_mm"] = 2.0
    with pytest.raises(ValueError):
        run(c, None, log=lambda *a: None, write_fields=False)


def test_bilateral_mirror_is_symmetric_without_tumour(cfg):
    c = copy.deepcopy(cfg)
    c.pop("tumour")
    c["anatomy"]["bilateral"] = True
    res = run(c, None, log=lambda *a: None, write_fields=False)
    lr = res["lr"]
    assert len(lr) > 0
    z = res["measurements"]
    zl = z[z.side == "left"].z_mag_ohm.values
    zr = z[z.side == "right"].z_mag_ohm.values
    np.testing.assert_allclose(zl, zr, rtol=1e-6)


def test_mirror_records_assumption():
    a = synthetic_hemisphere(radius_mm=30.0)
    m = a.mirror()
    assert m.mirrored and m.side != a.side
    assert any("mirror" in n.lower() for n in m.notes)


def test_point_triangle_distance():
    from breastsim.tumour import points_to_triangles_min_distance

    tri = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], float)
    assert points_to_triangles_min_distance(np.array([[0.2, 0.2, 3.0]]), tri) == pytest.approx(3.0)
    assert points_to_triangles_min_distance(np.array([[2.0, 0.0, 0.0]]), tri) == pytest.approx(1.0)
    assert points_to_triangles_min_distance(np.array([[1.0, 1.0, 0.0]]), tri) == pytest.approx(np.sqrt(0.5))


def test_electrode_clearance_uses_tumour_surface():
    """Clearance is measured from the tumour surface, not element centroids."""
    from breastsim.anatomy import build_mesh
    from breastsim.electrodes import place_electrodes, surface_point
    from breastsim.tumour import insert_tumour, make_tumour

    am = build_mesh(synthetic_hemisphere(radius_mm=50.0), h_mm=3.0, chest_wall_mm=15.0)
    els = place_electrodes(am.mesh.nodes, am.skin_facets, am.landmarks,
                           [dict(id="E1", clock_hour=12, s=0.5, radius_mm=6)])
    _, c, n = surface_point(am.mesh.nodes * 1e3, am.skin_facets, am.landmarks, 12, 0.5)
    radius, depth = 6.0, 10.0
    tum = make_tumour(dict(shape="sphere", radius_mm=radius), c - n * depth)
    _, rep = insert_tumour(am, tum, els, margin_mm=1.0)
    clearance = depth - radius
    assert rep["min_distance_to_electrode_mm"] == pytest.approx(clearance, abs=1.0)
    with pytest.raises(ValueError, match="electrode"):
        insert_tumour(am, tum, els, margin_mm=rep["min_distance_to_electrode_mm"] + 0.3)


def test_unilateral_tumour_leaves_contralateral_breast_healthy(cfg):
    c = copy.deepcopy(cfg)
    c["anatomy"]["bilateral"] = True
    c["tumour"]["sides"] = ["left"]
    c["mesh"]["h_mm"] = 4.0
    c["mesh"].pop("tumour_h_mm")  # uniform grid -> mirror-identical meshes (see assumptions doc)
    res = run(c, None, log=lambda *a: None, write_fields=False)
    m = res["measurements"]
    assert set(m[m.side == "right"].case) == {"absent"}
    assert set(m[m.side == "left"].case) == {"absent", "present"}
    assert set(res["diff"].side) == {"left"}
    lr = res["lr"]
    pres = lr[lr.case == "present"]
    assert (pres.right_case == "absent").all() and np.abs(pres.rel_d_mag).max() > 1e-4
    absent = lr[lr.case == "absent"]
    assert np.abs(absent.rel_d_mag).max() < 1e-6


def test_unknown_scale_key_is_rejected(cfg):
    c = copy.deepcopy(cfg)
    c["materials"]["scales"] = {"adipose_uci": {"sigma": 2.0}}
    with pytest.raises(KeyError, match="sigma_scale"):
        run(c, None, log=lambda *a: None, write_fields=False)


def test_sigma_scale_changes_result(cfg):
    base = run(copy.deepcopy(cfg), None, log=lambda *a: None, write_fields=False)["measurements"]
    c = copy.deepcopy(cfg)
    c["materials"]["scales"] = {"adipose_uci": {"sigma_scale": 2.0}}
    scaled = run(c, None, log=lambda *a: None, write_fields=False)["measurements"]
    assert not np.allclose(base.z_mag_ohm.values, scaled.z_mag_ohm.values, rtol=1e-6)
