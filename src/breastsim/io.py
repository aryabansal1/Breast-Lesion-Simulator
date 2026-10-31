"""Field and geometry export (VTU via meshio)."""

from __future__ import annotations

import meshio
import numpy as np


def write_fields_vtu(path, electrode_path, am, sysm, u, labels, electrodes, gamma_e, measurement_id, f):
    mesh = am.mesh
    e, j = sysm.element_fields(u)
    cells = [("tetra", mesh.tets)]
    cd = {
        "label": [labels.astype(np.int32)],
        "tumour": [(labels == 200).astype(np.int8)],
        "sigma_S_per_m": [gamma_e.real],
        "E_abs_V_per_m": [np.linalg.norm(np.abs(e), axis=1)],
        "J_abs_A_per_m2": [np.linalg.norm(np.abs(j), axis=1)],
        "J_re": [j.real],
        "J_im": [j.imag],
    }
    pdict = {"V_re": u.real, "V_im": u.imag, "V_abs": np.abs(u)}
    m = meshio.Mesh(mesh.nodes * 1e3, cells, point_data=pdict, cell_data=cd)
    m.field_data = {}
    meshio.write(str(path), m)
    tri = np.concatenate([el.facets for el in electrodes])
    eid = np.concatenate([np.full(len(el.facets), i, np.int32) for i, el in enumerate(electrodes)])
    meshio.write(str(electrode_path), meshio.Mesh(mesh.nodes * 1e3, [("triangle", tri)],
                                                  cell_data={"electrode_index": [eid]}))
    return path
