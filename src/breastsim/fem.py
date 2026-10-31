"""P1 finite-element solver for the electroquasistatic problem with the complete
electrode model (CEM).

    div(gamma grad u) = 0            in Omega, gamma = sigma + j*omega*eps
    gamma du/dn = 0                  on boundary outside electrodes
    u + z_l gamma du/dn = U_l        on electrode e_l
    int_{e_l} gamma du/dn dS = I_l   (current into the body through electrode l)
    sum_l U_l = 0                    (reference-potential constraint, applied by grounding
                                     one electrode and shifting the solution)

Phasor convention: v(t) = Re{V exp(j omega t)}. SI units throughout.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .mesh import facet_geometry


def p1_gradients(nodes, tets):
    x = nodes[tets]
    a = np.transpose(x[:, 1:] - x[:, [0]], (0, 2, 1))  # columns are edge vectors
    det = np.linalg.det(a)
    inv = np.linalg.inv(a)  # rows = grad(lambda_1..3)
    g = np.empty((len(tets), 4, 3))
    g[:, 1:] = inv
    g[:, 0] = -inv.sum(axis=1)
    return np.abs(det) / 6.0, g


def assemble_stiffness(nodes, tets, gamma_e):
    vol, g = p1_gradients(nodes, tets)
    ke = np.einsum("eik,ejk->eij", g, g) * (vol * gamma_e)[:, None, None]
    rows = np.repeat(tets, 4, axis=1).ravel()
    cols = np.tile(tets, (1, 4)).ravel()
    n = len(nodes)
    return sp.csr_matrix((ke.ravel(), (rows, cols)), shape=(n, n))


def electrode_terms(nodes, facets):
    """Boundary mass matrix M, load vector b = int phi_i dS, and area for one electrode."""
    area, _, _ = facet_geometry(nodes, facets)
    m_loc = (np.ones((3, 3)) + np.eye(3)) / 12.0
    me = area[:, None, None] * m_loc
    rows = np.repeat(facets, 3, axis=1).ravel()
    cols = np.tile(facets, (1, 3)).ravel()
    n = len(nodes)
    m = sp.csr_matrix((me.ravel(), (rows, cols)), shape=(n, n))
    b = np.bincount(facets.ravel(), weights=np.repeat(area / 3.0, 3), minlength=n)
    return m, b, float(area.sum())


@dataclass
class CEMSolution:
    u: np.ndarray  # (N, P) nodal potentials
    U: np.ndarray  # (L, P) electrode potentials
    currents: np.ndarray  # (L, P) prescribed currents
    diagnostics: dict = field(default_factory=dict)


class CEMSystem:
    def __init__(self, nodes, tets, gamma_e, electrode_facets, z_contact, method="direct", tol=1e-10,
                 maxiter=2000):
        if method not in ("direct", "iterative"):
            raise ValueError(f"solver method must be 'direct' or 'iterative', got {method!r}")
        self.method, self.tol, self.maxiter = method, tol, maxiter
        self.nodes, self.tets = nodes, tets
        self.gamma_e = np.asarray(gamma_e, complex)
        self.L = len(electrode_facets)
        self.z = np.broadcast_to(np.asarray(z_contact, complex), (self.L,)).copy()
        n, L = len(nodes), self.L
        t0 = time.perf_counter()
        k = assemble_stiffness(nodes, tets, self.gamma_e).tocsr()
        self.b, self.area = [], np.zeros(L)
        cpl_r, cpl_c, cpl_v = [], [], []
        for li, f in enumerate(electrode_facets):
            m, b, a = electrode_terms(nodes, np.asarray(f))
            k = k + m / self.z[li]
            nz = np.nonzero(b)[0]
            cpl_r.append(nz)
            cpl_c.append(np.full(len(nz), n + li))
            cpl_v.append(-b[nz] / self.z[li])
            self.b.append(b)
            self.area[li] = a
        # Reference potential: ground the last electrode (U_L = 0) during the solve, then shift
        # the solution so that sum(U) = 0. This is equivalent to the constraint sum(U) = 0
        # (the field is defined up to a constant when sum(I) = 0) and keeps the matrix
        # complex-symmetric without a zero diagonal block.
        r = np.concatenate(cpl_r)
        c = np.concatenate(cpl_c)
        v = np.concatenate(cpl_v)
        diag_idx = np.arange(n, n + L)
        rows = np.concatenate([r, c, diag_idx])
        cols = np.concatenate([c, r, diag_idx])
        vals = np.concatenate([v, v, self.area / self.z])
        extra = sp.csr_matrix((vals, (rows, cols)), shape=(n + L, n + L))
        k.resize((n + L, n + L))
        self.A_full = (k + extra).tocsr()
        self.A = self.A_full[: n + L - 1, : n + L - 1].tocsc()
        self.assembly_time = time.perf_counter() - t0
        self._lu = None
        self._amg = None
        self.iterations = []

    def factorize(self):
        """Sparse LU (SuperLU) with a fill-reducing nested-dissection ordering (METIS).

        The matrix is complex symmetric with a dominant diagonal, so symmetric mode
        without partial pivoting is used; the residual is checked after every solve."""
        t0 = time.perf_counter()
        if self.method == "iterative":
            return self._setup_amg(t0)
        self.perm = nested_dissection_order(self.A)
        self.ordering = "metis_nd" if self.perm is not None else "mmd_at_plus_a"
        if self.perm is None:
            self._lu = spla.splu(self.A, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                                 options=dict(SymmetricMode=True))
        else:
            ap = self.A[self.perm][:, self.perm].tocsc()
            self._lu = spla.splu(ap, permc_spec="NATURAL", diag_pivot_thresh=0.0,
                                 options=dict(SymmetricMode=True))
        self.factor_time = time.perf_counter() - t0
        return self

    def _setup_amg(self, t0):
        """Smoothed-aggregation AMG on Re(A). Re(A) is SPD (sigma > 0, Re(1/z) > 0, one electrode
        grounded) and dominates Im(A) at kHz-MHz, so it is a good preconditioner for COCG."""
        import pyamg
        self._amg = pyamg.smoothed_aggregation_solver(self.A.real.tocsr(), symmetry="symmetric", max_coarse=500)
        self._prec = self._amg.aspreconditioner(cycle="V")
        self.ordering = "amg_cocg"
        self.factor_time = time.perf_counter() - t0
        return self

    def _precondition(self, r):
        out = np.empty_like(r)
        for j in range(r.shape[1]):
            out[:, j] = self._prec @ r[:, j].real + 1j * (self._prec @ r[:, j].imag)
        return out

    def _cocg_solve(self, b):
        """Preconditioned conjugate orthogonal CG for complex-symmetric A (unconjugated dot products),
        all right-hand sides advanced together. Stops at ||b - A x|| <= tol ||b|| per column."""
        a = self.A.tocsr()
        x = np.zeros_like(b)
        r = b.copy()
        bn = np.maximum(np.linalg.norm(b, axis=0), 1e-300)
        z = self._precondition(r)
        p = z.copy()
        rho = np.einsum("ij,ij->j", r, z)
        active = np.linalg.norm(r, axis=0) > self.tol * bn
        its = np.zeros(b.shape[1], int)
        for _ in range(self.maxiter):
            if not active.any():
                break
            q = a @ p
            pq = np.einsum("ij,ij->j", p, q)
            alpha = np.divide(rho, pq, out=np.zeros_like(rho), where=active)
            x += alpha * p
            r -= alpha * q
            its += active
            active &= np.linalg.norm(r, axis=0) > self.tol * bn
            if not active.any():
                break
            z = self._precondition(r)
            rho_new = np.einsum("ij,ij->j", r, z)
            beta = np.divide(rho_new, rho, out=np.zeros_like(rho), where=active)
            rho = np.where(active, rho_new, rho)
            p = z + beta * p
        if active.any():
            raise RuntimeError(f"COCG did not reach tol={self.tol:g} in {self.maxiter} iterations")
        self.iterations = its.tolist()
        return x

    def _lu_solve(self, b):
        if self.perm is None:
            return self._lu.solve(b)
        x = np.empty_like(b)
        x[self.perm] = self._lu.solve(b[self.perm])
        return x

    def solve(self, currents) -> CEMSolution:
        """currents: (L, P) electrode current patterns (A), each column summing to zero."""
        currents = np.asarray(currents, complex)
        if currents.ndim == 1:
            currents = currents[:, None]
        n, L = len(self.nodes), self.L
        if self._lu is None and self._amg is None:
            self.factorize()
        rhs = np.zeros((n + L, currents.shape[1]), complex)
        rhs[n:n + L] = currents
        t0 = time.perf_counter()
        x = np.zeros_like(rhs)
        inv = self._cocg_solve if self.method == "iterative" else self._lu_solve
        x[: n + L - 1] = inv(rhs[: n + L - 1])
        solve_time = time.perf_counter() - t0
        # residual of the full (ungrounded) system: checks the grounded electrode's equation too
        res = np.linalg.norm(self.A_full @ x - rhs, axis=0) / np.maximum(np.linalg.norm(rhs, axis=0), 1e-300)
        shift = x[n:n + L].mean(axis=0)
        x -= shift
        u, U = x[:n], x[n:n + L]
        # electrode currents recovered from the field: I_l = (A_l U_l - b_l . u) / z_l
        i_field = np.array([(self.area[li] * U[li] - self.b[li] @ u) / self.z[li] for li in range(L)])
        cons = np.abs(i_field - currents).max(axis=0) / np.maximum(np.abs(currents).max(axis=0), 1e-300)
        diag = dict(relative_residual=res.tolist(), current_conservation_error=cons.tolist(),
                    net_injected_current=np.abs(currents.sum(axis=0)).tolist(),
                    factor_time_s=getattr(self, "factor_time", None), solve_time_s=solve_time,
                    assembly_time_s=self.assembly_time, ndof=int(self.A.shape[0]),
                    solver=self.method,
                    lu_nnz=int(self._lu.L.nnz + self._lu.U.nnz) if self._lu is not None else None,
                    iterations=self.iterations if self.method == "iterative" else None)
        return CEMSolution(u, U, currents, diag)

    def element_fields(self, u):
        """Element-wise E = -grad u (V/m) and J = gamma E (A/m^2) for one solution column."""
        _, g = p1_gradients(self.nodes, self.tets)
        e = -np.einsum("eij,ei->ej", g, u[self.tets])
        return e, self.gamma_e[:, None] * e


def nested_dissection_order(a):
    try:
        import pymetis
    except ImportError:
        return None
    g = (a != 0).tocsr().astype(np.int8)
    g.setdiag(0)
    g.eliminate_zeros()
    if hasattr(pymetis, "CSRAdjacency"):
        perm, _ = pymetis.nested_dissection(pymetis.CSRAdjacency(g.indptr, g.indices))
    else:
        perm, _ = pymetis.nested_dissection(xadj=g.indptr, adjncy=g.indices)
    return np.asarray(perm)


def drive_pattern(L, inj_pos, inj_neg, amplitude):
    i = np.zeros(L, complex)
    i[inj_pos] += amplitude
    i[inj_neg] -= amplitude
    return i
