"""Tumour geometry, placement and insertion.

A tumour supplies SHAPE only. Its electrical properties come from the material
library (label 200 by default). Inserting a tumour REPLACES the tissue whose tet
centroids fall inside the tumour (no tissue displacement / mass effect).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .anatomy import AIR, SKIN, TUMOUR, AnatomyMesh
from .electrodes import surface_point
from .mesh import facet_geometry


def rotation_matrix(deg_zyx):
    z, y, x = np.deg2rad(deg_zyx)
    rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    return rz @ ry @ rx


@dataclass
class Tumour:
    shape: str
    center_mm: np.ndarray
    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    semi_axes_mm: np.ndarray | None = None
    mask: np.ndarray | None = None  # voxel lesion
    voxel_mm: float = 0.0
    mask_center: np.ndarray | None = None
    trimesh_obj: object = None
    label: int = TUMOUR

    def local(self, pts_mm):
        return (pts_mm - self.center_mm) @ self.R

    def contains(self, pts_mm):
        p = self.local(np.atleast_2d(pts_mm))
        if self.shape in ("sphere", "ellipsoid"):
            return ((p / self.semi_axes_mm) ** 2).sum(1) <= 1.0
        if self.shape == "voxel":
            ijk = np.round(p / self.voxel_mm + self.mask_center).astype(int)
            ok = ((ijk >= 0) & (ijk < np.array(self.mask.shape))).all(1)
            out = np.zeros(len(p), bool)
            out[ok] = self.mask[ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]]
            return out
        if self.shape == "stl":
            return self.trimesh_obj.contains(p)
        raise ValueError(self.shape)

    @property
    def bounding_radius_mm(self):
        if self.semi_axes_mm is not None:
            return float(np.max(self.semi_axes_mm))
        if self.mask is not None:
            pts = (np.argwhere(self.mask) - self.mask_center) * self.voxel_mm
            return float(np.linalg.norm(pts, axis=1).max() + self.voxel_mm)
        return float(np.linalg.norm(self.trimesh_obj.vertices, axis=1).max())

    @property
    def nominal_volume_mm3(self):
        if self.semi_axes_mm is not None:
            return float(4 / 3 * np.pi * np.prod(self.semi_axes_mm))
        if self.mask is not None:
            return float(self.mask.sum() * self.voxel_mm ** 3)
        return float(self.trimesh_obj.volume)


def make_tumour(spec, center_mm) -> Tumour:
    shape = spec["shape"]
    R = rotation_matrix(spec.get("rotation_deg", [0, 0, 0]))
    c = np.asarray(center_mm, float)
    if shape == "sphere":
        return Tumour(shape, c, R, semi_axes_mm=np.full(3, float(spec["radius_mm"])))
    if shape == "ellipsoid":
        return Tumour(shape, c, R, semi_axes_mm=np.asarray(spec["semi_axes_mm"], float))
    if shape == "voxel":
        n = spec["dims"]
        m = np.fromfile(spec["path"], dtype=np.uint8).reshape(n[::-1]).transpose(2, 1, 0) > 0
        idx = np.argwhere(m)
        extent = (idx.max(0) - idx.min(0) + 1).max()
        voxel = float(spec["size_mm"]) / extent if "size_mm" in spec else float(spec["voxel_mm"])
        return Tumour(shape, c, R, mask=m, voxel_mm=voxel, mask_center=idx.mean(0))
    if shape == "stl":
        import trimesh

        tm = trimesh.load(spec["path"], force="mesh")
        tm.apply_scale(spec.get("scale", 1.0))
        tm.apply_translation(-tm.center_mass)
        return Tumour(shape, c, R, trimesh_obj=tm)
    raise ValueError(f"unknown tumour shape {shape}")


def resolve_center(spec, am: AnatomyMesh, side="left", mirror_for_right=True):
    """Absolute centre_mm, or anatomical {clock_hour, s, depth_mm} measured inward from the skin."""
    if "center_mm" in spec:
        return np.asarray(spec["center_mm"], float), {"placement": "absolute"}
    p = spec["placement"]
    _, c, n = surface_point(am.mesh.nodes * 1e3, am.skin_facets, am.landmarks, p["clock_hour"], p["s"],
                            side, mirror_for_right)
    centre = c - n * p["depth_mm"]
    return centre, {"placement": "anatomical", "skin_point_mm": c.tolist(), "inward_normal": (-n).tolist()}


def insert_tumour(am: AnatomyMesh, tumour: Tumour, electrodes, margin_mm=1.0):
    """Return (tet labels with tumour, report). Raises ValueError if checks fail."""
    mesh = am.mesh
    cen_mm = mesh.centroids() * 1e3
    inside = tumour.contains(cen_mm)
    if not inside.any():
        raise ValueError("tumour contains no mesh elements; refine the mesh near the tumour")
    rb = tumour.bounding_radius_mm
    g = np.linspace(-rb, rb, 25)
    G = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3) @ tumour.R.T + tumour.center_mm
    G = G[tumour.contains(G)]
    labs = am.labels_at(G)
    if (labs == AIR).any():
        raise ValueError("tumour intersects air or extends outside the computational domain")
    on_bnd = np.isin(mesh.tets[inside], am.boundary_nodes).any(1)
    if on_bnd.any():
        raise ValueError("tumour touches the external boundary of the domain")
    min_d = np.inf
    for e in electrodes:
        _, _, fc = facet_geometry(mesh.nodes * 1e3, e.facets)
        d = np.linalg.norm(cen_mm[inside][:, None, :] - fc[None], axis=2).min()
        min_d = min(min_d, float(d))
    if min_d < margin_mm:
        raise ValueError(f"tumour is {min_d:.2f} mm from an electrode (< margin {margin_mm} mm)")
    vol = mesh.volumes() * 1e9
    replaced = {int(k): float(vol[inside & (mesh.labels == k)].sum()) for k in np.unique(mesh.labels[inside])}
    warnings = []
    if SKIN in replaced:
        warnings.append("tumour replaces skin elements")
    new = mesh.labels.copy()
    new[inside] = tumour.label
    meshed = float(vol[inside].sum())
    rep = dict(shape=tumour.shape, center_mm=tumour.center_mm.tolist(), nominal_volume_mm3=tumour.nominal_volume_mm3,
               meshed_volume_mm3=meshed, volume_error=meshed / tumour.nominal_volume_mm3 - 1,
               n_elements=int(inside.sum()), replaced_tissue_mm3_by_label=replaced,
               min_distance_to_electrode_mm=min_d, warnings=warnings)
    return new, rep
