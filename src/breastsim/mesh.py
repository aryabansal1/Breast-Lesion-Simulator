"""Tetrahedral meshes built from voxel grids (coordinates in metres)."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

_FACES = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])


@dataclass
class TetMesh:
    nodes: np.ndarray  # (N, 3) metres
    tets: np.ndarray  # (M, 4)
    labels: np.ndarray  # (M,) int

    @property
    def nnodes(self):
        return len(self.nodes)

    @property
    def ntets(self):
        return len(self.tets)

    def centroids(self):
        return self.nodes[self.tets].mean(axis=1)

    def volumes(self):
        x = self.nodes[self.tets]
        return np.abs(np.einsum("ij,ij->i", x[:, 1] - x[:, 0],
                                np.cross(x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]))) / 6.0

    def boundary_facets(self):
        """Return (facets (F,3), owner tet index (F,)) for faces used by exactly one tet.

        Facets are oriented with outward normals."""
        faces = self.tets[:, _FACES].reshape(-1, 3)
        owner = np.repeat(np.arange(self.ntets), 4)
        s = np.sort(faces, axis=1).astype(np.int64)
        n = np.int64(self.nnodes)
        if n < 2_000_000:
            key = (s[:, 0] * n + s[:, 1]) * n + s[:, 2]
            _, idx, cnt = np.unique(key, return_index=True, return_counts=True)
        else:
            _, idx, cnt = np.unique(s, axis=0, return_index=True, return_counts=True)
        b = idx[cnt == 1]
        return faces[b], owner[b]


def facet_geometry(nodes, facets):
    x = nodes[facets]
    cr = np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0])
    area = 0.5 * np.linalg.norm(cr, axis=1)
    normal = cr / (2 * area[:, None])
    return area, normal, x.mean(axis=1)


def voxel_tet_mesh(active: np.ndarray, origin, spacing):
    """Kuhn (6-tet) subdivision of every active voxel of a uniform grid."""
    active = np.asarray(active, bool)
    edges = [np.asarray(origin, float)[a] + np.arange(n + 1) * np.asarray(spacing, float)[a]
             for a, n in enumerate(active.shape)]
    return rect_tet_mesh(active, edges)


def rect_tet_mesh(active: np.ndarray, edges):
    """Kuhn (6-tet) subdivision of every active cell of a rectilinear grid.

    ``edges`` holds the node coordinates along each axis (len n_axis + 1), so the
    grid can be graded for local refinement. All cells share the same main
    diagonal, so the mesh is conforming. Returns (nodes, tets, cell_ijk)."""
    active = np.asarray(active, bool)
    nx, ny, nz = active.shape
    cells = np.argwhere(active)
    corner = lambda off: np.ravel_multi_index(  # noqa: E731
        (cells[:, 0] + off[0], cells[:, 1] + off[1], cells[:, 2] + off[2]), (nx + 1, ny + 1, nz + 1))
    tets = []
    for perm in itertools.permutations(range(3)):
        o = np.zeros(3, int)
        path = [corner(o)]
        for ax in perm:
            o = o.copy()
            o[ax] = 1
            path.append(corner(o))
        tets.append(np.stack(path, axis=1))
    tets = np.concatenate(tets)
    cell_ijk = np.tile(cells, (6, 1))
    used, inv = np.unique(tets, return_inverse=True)
    tets = inv.reshape(tets.shape)
    ii, jj, kk = np.unravel_index(used, (nx + 1, ny + 1, nz + 1))
    nodes = np.stack([np.asarray(edges[0], float)[ii], np.asarray(edges[1], float)[jj],
                      np.asarray(edges[2], float)[kk]], 1)
    x = nodes[tets]
    det = np.einsum("ij,ij->i", x[:, 1] - x[:, 0], np.cross(x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]))
    neg = det < 0
    tets[neg] = tets[neg][:, [0, 2, 1, 3]]
    return nodes, tets, cell_ijk


def box_mesh(size, n, label=1, origin=(0.0, 0.0, 0.0)):
    n = tuple(int(v) for v in n)
    spacing = np.asarray(size, float) / np.asarray(n)
    nodes, tets, _ = voxel_tet_mesh(np.ones(n, bool), origin, spacing)
    return TetMesh(nodes, tets, np.full(len(tets), label, dtype=np.int32))
