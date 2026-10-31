"""Numerical verification of the CEM/FEM solver (not physical validation)."""
import numpy as np
import pytest

from breastsim.fem import CEMSystem, drive_pattern
from breastsim.mesh import box_mesh, facet_geometry

LX, LY, LZ = 0.04, 0.01, 0.01


def bar(shape=(8, 2, 2), gamma=0.3 + 0.05j, z=2e-4 + 1e-5j):
    m = box_mesh((LX, LY, LZ), shape)
    f, _ = m.boundary_facets()
    _, _, c = facet_geometry(m.nodes, f)
    elec = [f[np.isclose(c[:, 0], 0)], f[np.isclose(c[:, 0], LX)]]
    return m, CEMSystem(m.nodes, m.tets, np.full(m.ntets, gamma), elec, z), gamma, z


@pytest.mark.parametrize("shape", [(8, 2, 2), (16, 4, 4)])
def test_homogeneous_bar_matches_analytic(shape):
    _, s, g, z = bar(shape)
    sol = s.solve(drive_pattern(2, 0, 1, 1e-3))
    z_num = (sol.U[0, 0] - sol.U[1, 0]) / 1e-3
    z_ana = LX / (g * LY * LZ) + 2 * z / (LY * LZ)
    assert abs(z_num - z_ana) / abs(z_ana) < 1e-10
    assert sol.diagnostics["relative_residual"][0] < 1e-10


def four_electrode_bar(gamma_e=None, shape=(12, 4, 4)):
    m = box_mesh((LX, LY, LZ), shape)
    f, _ = m.boundary_facets()
    _, _, c = facet_geometry(m.nodes, f)
    top = np.isclose(c[:, 2], LZ)
    elec = [f[np.isclose(c[:, 0], 0)], f[np.isclose(c[:, 0], LX)],
            f[top & (np.abs(c[:, 0] - 0.0133) < 0.003)], f[top & (np.abs(c[:, 0] - 0.0267) < 0.003)]]
    if gamma_e is None:
        gamma_e = np.full(m.ntets, 0.3 + 0.02j)
    return m, CEMSystem(m.nodes, m.tets, gamma_e, elec, 1e-3)


def test_current_conservation_and_reference():
    _, s = four_electrode_bar()
    sol = s.solve(drive_pattern(4, 0, 1, 1e-3))
    assert sol.diagnostics["current_conservation_error"][0] < 1e-10
    assert abs(sol.U[:, 0].sum()) < 1e-12 * np.abs(sol.U).max()


def test_linear_scaling_with_current():
    _, s = four_electrode_bar()
    a = s.solve(drive_pattern(4, 0, 1, 1e-3)).U
    b = s.solve(drive_pattern(4, 0, 1, 7e-3)).U
    np.testing.assert_allclose(b, 7 * a, rtol=1e-10, atol=1e-14)


def test_reciprocity():
    rng = np.random.default_rng(0)
    m = box_mesh((LX, LY, LZ), (12, 4, 4))
    g = 0.2 + rng.random(m.ntets) * 0.5 + 1j * rng.random(m.ntets) * 0.05
    _, s = four_electrode_bar(g)
    i_ab = drive_pattern(4, 0, 1, 1.0)
    i_cd = drive_pattern(4, 2, 3, 1.0)
    u1 = s.solve(i_ab).U[:, 0]
    u2 = s.solve(i_cd).U[:, 0]
    z_ab_cd = u1[2] - u1[3]
    z_cd_ab = u2[0] - u2[1]
    assert abs(z_ab_cd - z_cd_ab) / abs(z_ab_cd) < 1e-10


def test_zero_contrast_inclusion_gives_identical_result():
    m, s0 = four_electrode_bar()
    g = np.full(m.ntets, 0.3 + 0.02j)
    _, s1 = four_electrode_bar(g.copy())
    a = s0.solve(drive_pattern(4, 0, 1, 1e-3)).U
    b = s1.solve(drive_pattern(4, 0, 1, 1e-3)).U
    np.testing.assert_allclose(a, b, rtol=1e-12)


def test_deterministic_repeat():
    _, s = four_electrode_bar()
    a = s.solve(drive_pattern(4, 0, 1, 1e-3)).U
    _, s2 = four_electrode_bar()
    b = s2.solve(drive_pattern(4, 0, 1, 1e-3)).U
    assert np.array_equal(a, b)


def test_mesh_convergence_point_electrodes():
    """Four-electrode transfer impedance converges monotonically as h -> 0."""
    zs = []
    for k in (1, 2, 4):
        m = box_mesh((LX, LY, LZ), (12 * k, 4 * k, 4 * k))
        f, _ = m.boundary_facets()
        _, _, c = facet_geometry(m.nodes, f)
        top = np.isclose(c[:, 2], LZ)
        elec = [f[np.isclose(c[:, 0], 0)], f[np.isclose(c[:, 0], LX)],
                f[top & (np.abs(c[:, 0] - 0.0133) < 0.0034) & (np.abs(c[:, 1] - 0.005) < 0.0026)],
                f[top & (np.abs(c[:, 0] - 0.0267) < 0.0034) & (np.abs(c[:, 1] - 0.005) < 0.0026)]]
        s = CEMSystem(m.nodes, m.tets, np.full(m.ntets, 0.3 + 0.0j), elec, 1e-3)
        u = s.solve(drive_pattern(4, 0, 1, 1.0)).U[:, 0]
        zs.append(u[2] - u[3])
    e1, e2 = abs(zs[0] - zs[1]), abs(zs[1] - zs[2])
    assert e2 < 0.7 * e1
    assert e2 / abs(zs[2]) < 0.02


def test_iterative_solver_matches_direct():
    rng = np.random.default_rng(1)
    m = box_mesh((LX, LY, LZ), (24, 6, 6))
    g = 0.02 + rng.random(m.ntets) * 0.5 + 1j * rng.random(m.ntets) * 0.05
    f, _ = m.boundary_facets()
    _, _, c = facet_geometry(m.nodes, f)
    top = np.isclose(c[:, 2], LZ)
    elec = [f[np.isclose(c[:, 0], 0)], f[np.isclose(c[:, 0], LX)],
            f[top & (np.abs(c[:, 0] - 0.0133) < 0.003)], f[top & (np.abs(c[:, 0] - 0.0267) < 0.003)]]
    z = 1e-3 + 2e-4j
    inj = np.stack([drive_pattern(4, 0, 1, 1e-3), drive_pattern(4, 2, 3, 1e-3)], axis=1)
    d = CEMSystem(m.nodes, m.tets, g, elec, z).solve(inj)
    it = CEMSystem(m.nodes, m.tets, g, elec, z, method="iterative", tol=1e-12).solve(inj)
    assert np.abs(it.U - d.U).max() / np.abs(d.U).max() < 1e-8
    assert max(it.diagnostics["relative_residual"]) < 1e-11
    assert max(it.diagnostics["current_conservation_error"]) < 1e-8
    assert it.diagnostics["iterations"] and it.diagnostics["lu_nnz"] is None
