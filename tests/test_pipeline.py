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
