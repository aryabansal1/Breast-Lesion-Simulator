"""Surface electrode placement from anatomical landmarks.

An electrode position is given by a clock hour (viewer facing the patient, 12 =
superior, 3 = viewer's right) and an axial fraction ``s`` from the nipple (s = 0)
to the breast base at the chest wall (s = 1). The same specification therefore
lands at the same anatomical location regardless of breast size or mesh
resolution. For a right breast the clock hour is mirrored (h -> 12 - h) by default
so that, e.g., a lateral electrode stays lateral.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mesh import facet_geometry


@dataclass
class Electrode:
    id: str
    facets: np.ndarray
    center_mm: np.ndarray
    area_m2: float
    nominal_area_m2: float
    clock_hour: float
    s: float
    projected_area_m2: float = 0.0  # facet area projected on the local smoothed normal


def effective_clock(hour, side, mirror_for_right=True):
    return (12.0 - hour) % 12.0 if (side == "right" and mirror_for_right) else hour % 12.0


def smoothed_normal(area, normal, cen, c, radius_mm=6.0):
    """Area-weighted mean facet normal near c (voxel facets are axis-aligned staircases)."""
    sel = np.linalg.norm(cen - c, axis=1) <= radius_mm
    n = (normal[sel] * area[sel, None]).sum(0)
    return n / np.linalg.norm(n)


def surface_point(nodes_mm, skin_facets, lm, clock_hour, s, side="left", mirror_for_right=True):
    """Return (index of nearest skin facet, its centroid, smoothed outward normal) for a landmark position."""
    area, normal, cen = facet_geometry(nodes_mm, skin_facets)
    h = effective_clock(clock_hour, side, mirror_for_right)
    th = np.deg2rad(30.0 * h)
    a, b, depth = lm["axis"], lm["base_center_mm"], lm["breast_depth_mm"]
    d = np.cos(th) * lm["superior"] + np.sin(th) * lm["viewer_right"]
    d = d - (d @ a) * a
    d /= np.linalg.norm(d)
    rel = cen - b
    t = rel @ a
    rad = rel - np.outer(t, a)
    rho = np.linalg.norm(rad, axis=1)
    cosang = np.clip((rad @ d) / np.maximum(rho, 1e-12), -1, 1)
    cost = (t - (1 - s) * depth) ** 2 + (rho * np.arccos(cosang)) ** 2
    i = int(np.argmin(cost))
    return i, cen[i], smoothed_normal(area, normal, cen, cen[i])


def place_electrodes(nodes_m, skin_facets, lm, specs, side="left", mirror_for_right=True):
    nodes_mm = nodes_m * 1e3
    area, normal, cen = facet_geometry(nodes_mm, skin_facets)
    taken = np.zeros(len(skin_facets), bool)
    out = []
    for sp in specs:
        i, c, n = surface_point(nodes_mm, skin_facets, lm, sp["clock_hour"], sp["s"], side, mirror_for_right)
        r = sp["radius_mm"]
        sel = (np.linalg.norm(cen - c, axis=1) <= r) & (normal @ n > 0.0)
        if sel.sum() == 0:
            sel[i] = True
        if (sel & taken).any():
            raise ValueError(f"electrode {sp['id']} overlaps another electrode; reduce radius or move it")
        taken |= sel
        # Voxel-derived skin is stair-stepped: its facet area overestimates the smooth surface
        # area. The projection onto the smoothed normal approximates the smooth contact area.
        proj = float(np.sum(area[sel] * np.clip(normal[sel] @ n, 0.0, None)) * 1e-6)
        out.append(Electrode(sp["id"], skin_facets[sel], c, float(area[sel].sum() * 1e-6),
                             float(np.pi * r ** 2 * 1e-6), sp["clock_hour"], sp["s"], proj))
    return out


def contact_impedance(spec, f):
    """Specific contact impedance z (ohm*m^2) at frequency f.

    spec: {model: constant, z_ohm_m2: value} or {model: parallel_rc, r_ohm_m2, c_f_per_m2}."""
    m = spec.get("model", "constant")
    if m == "constant":
        return complex(spec["z_ohm_m2"])
    if m == "parallel_rc":
        w = 2 * np.pi * f
        return spec["r_ohm_m2"] / (1 + 1j * w * spec["r_ohm_m2"] * spec["c_f_per_m2"])
    raise ValueError(f"unknown contact model {m}")
