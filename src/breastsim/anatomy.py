"""Uncompressed tissue-label anatomy and its conversion to a tetrahedral mesh.

Coordinates follow breastPhantom: x = chest wall -> nipple, y = lateral, z =
inferior -> superior, millimetres. Meshes are returned in metres.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from .mesh import TetMesh, facet_geometry, rect_tet_mesh

AIR, FAT, SKIN, GLAND, NIPPLE, MUSCLE, CHEST_EXT, TUMOUR = 0, 1, 2, 29, 33, 40, 41, 200
KNOWN_LABELS = np.array([0, 1, 2, 29, 33, 40, 41, 88, 95, 125, 150, 200, 225, 250], dtype=np.uint8)


@dataclass
class Anatomy:
    labels: np.ndarray  # (nx, ny, nz) uint8
    spacing_mm: np.ndarray
    origin_mm: np.ndarray  # centre of voxel [0, 0, 0]
    side: str = "left"
    source: str = ""
    mirrored: bool = False
    y_positive_is_patient_left: bool = True
    notes: list = field(default_factory=list)

    @classmethod
    def from_npz(cls, path, side="left", y_positive_is_patient_left=True):
        d = np.load(path)
        a = cls(d["labels"], d["spacing_mm"].astype(float), d["offset_mm"].astype(float), side,
                str(d["source"]), False, y_positive_is_patient_left)
        a.notes.append(f"Loaded from {path} (source {d['source']}, downsampled x{int(d['factor'])} by block "
                       "majority from the uncompressed phantom).")
        a.notes.append("ASSUMPTION: +y of the phantom points to the patient's left; the phantom is treated as "
                       f"a {side} breast. breastPhantom does not document laterality.")
        return a

    def mirror(self) -> "Anatomy":
        """Mirror across the y mid-plane to create the contralateral breast (ASSUMPTION: symmetry)."""
        other = "right" if self.side == "left" else "left"
        a = Anatomy(self.labels[:, ::-1, :].copy(), self.spacing_mm.copy(), self.origin_mm.copy(), other,
                    self.source, not self.mirrored, self.y_positive_is_patient_left, list(self.notes))
        a.notes.append(f"{other} breast created by MIRRORING the {self.side} breast; this does not represent "
                       "natural left/right anatomical variability.")
        return a

    @property
    def mirror_plane_y_mm(self) -> float:
        return float(self.origin_mm[1] + (self.labels.shape[1] - 1) * self.spacing_mm[1] / 2)

    def mirror_point(self, p_mm):
        q = np.array(p_mm, float)
        q[..., 1] = 2 * self.mirror_plane_y_mm - q[..., 1]
        return q

    def edges(self, ax):
        n = self.labels.shape[ax]
        return self.origin_mm[ax] - self.spacing_mm[ax] / 2 + np.arange(n + 1) * self.spacing_mm[ax]

    def centers(self, ax):
        return self.origin_mm[ax] + np.arange(self.labels.shape[ax]) * self.spacing_mm[ax]

    def landmarks(self) -> dict:
        lab = self.labels
        skin_x = np.nonzero((lab == SKIN).any(axis=(1, 2)))[0]
        if len(skin_x) == 0:
            raise ValueError("anatomy has no skin label (2); cannot locate the breast surface")
        i_cw = int(skin_x.min())
        chest_x = self.edges(0)[i_cw]
        nip = np.argwhere(lab == NIPPLE)
        if len(nip):
            ijk = nip.mean(0)
            # nipple tip: most anterior nipple voxel column near the centroid
            ijk[0] = nip[:, 0].max()
        else:
            t = np.argwhere(lab > 0)
            ijk = t[t[:, 0] == t[:, 0].max()].mean(0)
        nipple = self.origin_mm + ijk * self.spacing_mm
        jk = np.argwhere(lab[i_cw] > 0).mean(0)
        base = np.array([chest_x, *(self.origin_mm[1:] + jk * self.spacing_mm[1:])])
        axis = nipple - base
        depth = float(np.linalg.norm(axis))
        viewer_right = np.array([0.0, 1.0 if self.y_positive_is_patient_left else -1.0, 0.0])
        return dict(nipple_mm=nipple, base_center_mm=base, axis=axis / depth, breast_depth_mm=depth,
                    chest_wall_x_mm=float(chest_x), viewer_right=viewer_right,
                    superior=np.array([0.0, 0.0, 1.0]))


def synthetic_hemisphere(radius_mm=50.0, skin_mm=1.5, gland_semi_axes_mm=(25.0, 30.0, 30.0),
                         muscle_mm=10.0, spacing_mm=0.5, side="left") -> Anatomy:
    """Simple heterogeneous verification breast: hemisphere of fat with an ellipsoidal glandular
    core, a skin shell and a muscle slab behind it."""
    r = radius_mm
    n = (int(np.ceil((r + muscle_mm) / spacing_mm)), int(np.ceil(2 * r / spacing_mm)) + 2,
         int(np.ceil(2 * r / spacing_mm)) + 2)
    origin = np.array([-muscle_mm + spacing_mm / 2, -(n[1] - 1) * spacing_mm / 2, -(n[2] - 1) * spacing_mm / 2])
    x = origin[0] + np.arange(n[0]) * spacing_mm
    y = origin[1] + np.arange(n[1]) * spacing_mm
    z = origin[2] + np.arange(n[2]) * spacing_mm
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    lab = np.zeros(n, np.uint8)
    inb = (R <= r) & (X >= 0)
    lab[inb] = FAT
    g = gland_semi_axes_mm
    lab[inb & ((X / g[0]) ** 2 + (Y / g[1]) ** 2 + (Z / g[2]) ** 2 <= 1)] = GLAND
    lab[inb & (R > r - skin_mm)] = SKIN
    lab[(X < 0) & (np.sqrt(Y ** 2 + Z ** 2) <= r)] = MUSCLE
    lab[(R <= 4) & (X > r - 4)] = NIPPLE
    a = Anatomy(lab, np.full(3, spacing_mm), origin, side, "synthetic_hemisphere")
    a.notes.append(f"Synthetic hemisphere r={r} mm, skin {skin_mm} mm, glandular ellipsoid {g} mm, "
                   f"muscle slab {muscle_mm} mm (verification geometry, not anatomy).")
    return a


def graded_edges(lo, hi, h, windows, grading=0.4):
    """1D node coordinates with spacing h, refined to hf inside windows (c, r, hf)."""

    def target(x):
        t = h
        for c, r, hf in windows:
            t = min(t, hf + grading * max(abs(x - c) - r, 0.0))
        return t

    e = [lo]
    while e[-1] < hi - 1e-9:
        x = e[-1]
        e.append(min(x + target(x + 0.5 * target(x)), hi))
    if len(e) > 2 and e[-1] - e[-2] < 0.3 * target(hi):
        e.pop(-2)
    return np.array(e)


@dataclass
class AnatomyMesh:
    mesh: TetMesh
    cell_labels: np.ndarray
    edges_mm: list
    landmarks: dict
    skin_facets: np.ndarray  # outer skin boundary facets (node indices)
    boundary_nodes: np.ndarray
    stats: dict
    anatomy: Anatomy

    def locate_cells(self, pts_mm):
        """Coarse-grid cell index (i, j, k) of points; -1 outside the grid."""
        idx = []
        for ax in range(3):
            e = self.edges_mm[ax]
            i = np.searchsorted(e, pts_mm[:, ax], side="right") - 1
            i[(pts_mm[:, ax] < e[0]) | (pts_mm[:, ax] > e[-1])] = -1
            i = np.minimum(i, len(e) - 2)
            idx.append(i)
        return np.stack(idx, 1)

    def labels_at(self, pts_mm):
        ijk = self.locate_cells(pts_mm)
        out = np.zeros(len(pts_mm), np.int32)
        ok = (ijk >= 0).all(1)
        out[ok] = self.cell_labels[ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]]
        return out


def build_mesh(anat: Anatomy, h_mm=2.0, refine=(), chest_wall_mm=20.0, grading=0.4) -> AnatomyMesh:
    """Resample the label volume onto a (graded) rectilinear grid and Kuhn-split into tets.

    refine: iterable of dicts {center_mm: [x,y,z], radius_mm, h_mm} (tensor-product refinement).
    chest_wall_mm: thickness of an assumed muscle block (label 41) added behind the posterior
    face of the phantom; 0 disables it.
    """
    lab = anat.labels
    lm = anat.landmarks()
    tissue_idx = np.argwhere(lab > 0)
    lo_i, hi_i = tissue_idx.min(0), tissue_idx.max(0) + 1
    ex = [anat.edges(a) for a in range(3)]
    lo = np.array([ex[a][lo_i[a]] for a in range(3)])
    hi = np.array([ex[a][hi_i[a]] for a in range(3)])
    lo_x_mesh = lo[0] - chest_wall_mm
    edges = []
    for ax in range(3):
        wins = [(r["center_mm"][ax], r["radius_mm"], r["h_mm"]) for r in refine]
        a0 = lo_x_mesh if ax == 0 else lo[ax]
        if ax == 0 and chest_wall_mm > 0:
            e_ext = graded_edges(a0, lo[0], h_mm, wins, grading)
            e_in = graded_edges(lo[0], hi[0], h_mm, wins, grading)
            edges.append(np.concatenate([e_ext, e_in[1:]]))
        elif ax == 1 and anat.mirrored:
            # grade in the unmirrored frame and reflect, so a mirrored breast gets the mirror-image grid
            m2 = 2 * anat.mirror_plane_y_mm
            e = graded_edges(m2 - hi[ax], m2 - a0, h_mm, [(m2 - c, r, hf) for c, r, hf in wins], grading)
            edges.append((m2 - e)[::-1])
        else:
            edges.append(graded_edges(a0, hi[ax], h_mm, wins, grading))
    shape = tuple(len(e) - 1 for e in edges)
    ncell = int(np.prod(shape))

    # map fine voxels to coarse cells (block counts by label)
    lut = np.full(256, len(KNOWN_LABELS), np.int64)
    lut[KNOWN_LABELS] = np.arange(len(KNOWN_LABELS))
    nl = len(KNOWN_LABELS) + 1
    ci = [np.searchsorted(edges[a], anat.centers(a), side="right") - 1 for a in range(3)]
    valid = [(c >= 0) & (c < shape[a]) for a, c in enumerate(ci)]
    counts = np.zeros((ncell, nl), np.int64)
    J, K = np.meshgrid(ci[1], ci[2], indexing="ij")
    vjk = np.outer(valid[1], valid[2])
    for i in np.nonzero(valid[0])[0]:
        cid = (ci[0][i] * shape[1] + J[vjk]) * shape[2] + K[vjk]
        counts += np.bincount(cid * nl + lut[lab[i][vjk]], minlength=ncell * nl).reshape(ncell, nl)
    total = counts.sum(1)
    # fallback for cells smaller than a fine voxel: sample at the cell centre
    cc = [0.5 * (e[1:] + e[:-1]) for e in edges]
    centre_lab = np.zeros(shape, np.uint8)
    idx = [np.clip(np.round((cc[a] - anat.origin_mm[a]) / anat.spacing_mm[a]).astype(int), 0, lab.shape[a] - 1)
           for a in range(3)]
    inside = [(cc[a] >= ex[a][0]) & (cc[a] <= ex[a][-1]) for a in range(3)]
    sub = lab[np.ix_(idx[0], idx[1], idx[2])]
    centre_lab[np.ix_(*inside)] = sub[np.ix_(*inside)]
    counts = counts.reshape(*shape, nl)
    total = total.reshape(shape)
    i_air, i_skin = 0, int(lut[SKIN])
    tissue_cnt = total - counts[..., i_air] - counts[..., nl - 1]
    is_tissue = np.where(total > 0, tissue_cnt * 2 >= np.maximum(total, 1), centre_lab > 0)
    nonskin = counts.copy()
    nonskin[..., [i_air, i_skin, nl - 1]] = -1
    maj = KNOWN_LABELS[np.minimum(nonskin.argmax(-1), len(KNOWN_LABELS) - 1)]
    has_nonskin = nonskin.max(-1) > 0
    has_skin = counts[..., i_skin] > 0
    labels = np.where(total > 0, np.where(has_nonskin, maj, SKIN), centre_lab).astype(np.uint8)
    labels[~is_tissue] = AIR
    # skin on the outer surface of the breast (anterior to the chest-wall plane)
    pad = np.pad(labels > 0, 1, constant_values=False)
    near_air = np.zeros(shape, bool)
    for ax in range(3):
        for sgn in (-1, 1):
            near_air |= ~np.roll(pad, sgn, axis=ax)[1:-1, 1:-1, 1:-1]
    breast_region = (cc[0] >= lm["chest_wall_x_mm"])[:, None, None]
    labels[(labels > 0) & has_skin & near_air & breast_region] = SKIN
    labels[(labels > 0) & (labels != SKIN) & near_air & breast_region & (total == 0)] = SKIN
    # chest-wall extension: copy the footprint of the first in-phantom layer
    n_ext = int(np.sum(cc[0] < lo[0]))
    if n_ext:
        foot = labels[n_ext] > 0
        labels[:n_ext][:, foot] = CHEST_EXT
    # keep the largest face-connected tissue component (isolated islands make the system singular)
    comp, ncomp = ndimage.label(labels > 0)
    dropped = 0
    if ncomp > 1:
        sizes = np.bincount(comp.ravel())
        sizes[0] = 0
        keep = sizes.argmax()
        dropped = int(sizes.sum() - sizes[keep])
        labels[(comp != keep) & (comp > 0)] = AIR
    edges_m = [e * 1e-3 for e in edges]
    nodes, tets, cell_ijk = rect_tet_mesh(labels > 0, edges_m)
    tet_labels = labels[cell_ijk[:, 0], cell_ijk[:, 1], cell_ijk[:, 2]].astype(np.int32)
    mesh = TetMesh(nodes, tets, tet_labels)
    bf, owner = mesh.boundary_facets()
    _, _, fc = facet_geometry(nodes * 1e3, bf)
    skin_facets = bf[(tet_labels[owner] == SKIN) & (fc[:, 0] > lm["chest_wall_x_mm"])]
    outer = bf[fc[:, 0] > lm["chest_wall_x_mm"]]
    vol = mesh.volumes()
    stats = dict(h_mm=h_mm, grid_shape=list(shape), nodes=int(mesh.nnodes), tets=int(mesh.ntets),
                 min_edge_mm=float(min(np.diff(e).min() for e in edges)),
                 max_edge_mm=float(max(np.diff(e).max() for e in edges)),
                 chest_wall_extension_mm=chest_wall_mm, dropped_island_cells=dropped,
                 skin_coverage_of_breast_surface=float(len(skin_facets) / max(len(outer), 1)),
                 volume_ml_by_label={int(k): float(vol[tet_labels == k].sum() * 1e6)
                                     for k in np.unique(tet_labels)})
    return AnatomyMesh(mesh, labels, edges, lm, skin_facets, np.unique(bf), stats, anat)
