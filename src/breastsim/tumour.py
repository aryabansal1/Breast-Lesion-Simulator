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

    def surface_samples(self, n=2000) -> np.ndarray:
        """Points (mm, world frame) on or near the nominal tumour surface, for clearance checks."""
        if self.semi_axes_mm is not None:
            k = np.arange(n) + 0.5
            phi = np.arccos(1 - 2 * k / n)
            th = np.pi * (1 + 5 ** 0.5) * k
            u = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)
            local = u * self.semi_axes_mm
        elif self.mask is not None:
            idx = np.argwhere(self.mask)
            corners = np.array(np.meshgrid([-.5, .5], [-.5, .5], [-.5, .5], indexing="ij")).reshape(3, -1).T
            local = ((idx[:, None, :] + corners[None]).reshape(-1, 3) - self.mask_center) * self.voxel_mm
        else:
            local = np.asarray(self.trimesh_obj.vertices)
        return local @ self.R.T + self.center_mm

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


def points_to_triangles_min_distance(p, tri, chunk=4096):
    """Minimum Euclidean distance between points p (N,3) and triangles tri (M,3,3)."""
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    ab, ac = b - a, c - a
    n = np.cross(ab, ac)
    nn = np.einsum("ij,ij->i", n, n)
    best = np.inf
    for i in range(0, len(p), chunk):
        q = p[i:i + chunk, None, :]                       # (n,1,3)
        ap = q - a
        t = np.einsum("nmj,mj->nm", ap, n) / nn            # signed offset along normal / |n|^2
        proj = q - t[..., None] * n
        # barycentric test of projection
        v0, v1, v2 = ab, ac, proj - a
        d00 = np.einsum("ij,ij->i", v0, v0)
        d01 = np.einsum("ij,ij->i", v0, v1)
        d11 = np.einsum("ij,ij->i", v1, v1)
        d20 = np.einsum("nmj,mj->nm", v2, v0)
        d21 = np.einsum("nmj,mj->nm", v2, v1)
        den = d00 * d11 - d01 ** 2
        v = (d11 * d20 - d01 * d21) / den
        w = (d00 * d21 - d01 * d20) / den
        inside = (v >= 0) & (w >= 0) & (v + w <= 1)
        d_plane = np.where(inside, np.abs(t) * np.sqrt(nn), np.inf)
        d_edge = np.full(d_plane.shape, np.inf)
        for s0, s1 in ((a, b), (b, c), (c, a)):
            e = s1 - s0
            u = np.clip(np.einsum("nmj,mj->nm", q - s0, e) / np.einsum("ij,ij->i", e, e), 0.0, 1.0)
            d_edge = np.minimum(d_edge, np.linalg.norm(q - (s0 + u[..., None] * e), axis=2))
        best = min(best, float(np.minimum(d_plane, d_edge).min()))
    return best


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
    # Clearance = min distance from the tumour (meshed tumour nodes + nominal surface samples)
    # to the electrode contact triangles.
    tum_pts = np.vstack([mesh.nodes[np.unique(mesh.tets[inside])] * 1e3, tumour.surface_samples()])
    min_d = np.inf
    for e in electrodes:
        tri = mesh.nodes[e.facets] * 1e3
        min_d = min(min_d, float(points_to_triangles_min_distance(tum_pts, tri)))
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
