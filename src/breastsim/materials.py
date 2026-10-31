"""Frequency-dependent tissue electrical properties.

Every material returns a complex admittivity ``gamma(f) = sigma(f) + j*omega*eps(f)``
(S/m) under the phasor convention ``v(t) = Re{V exp(j omega t)}``. Each material
model produces gamma from a single representation, so dielectric loss is never
counted twice (see docs/material_library.md).
"""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.interpolate import PchipInterpolator

EPS0 = 8.8541878128e-12  # F/m

GABRIEL_TAU_UNITS = {"tau1_ps": 1e-12, "tau2_ns": 1e-9, "tau3_us": 1e-6, "tau4_ms": 1e-3}


class ExtrapolationError(ValueError):
    pass


def gabriel_complex_permittivity(f, p: dict, include_ionic: bool = True) -> np.ndarray:
    """Gabriel et al. (1996) four-term Cole-Cole relative complex permittivity."""
    f = np.asarray(f, dtype=float)
    w = 2 * np.pi * f
    eps = np.full(f.shape, p["ef"], dtype=complex)
    for n, (tau_key, scale) in enumerate(GABRIEL_TAU_UNITS.items(), start=1):
        d = p[f"del{n}"]
        if d == 0:
            continue
        tau = p[tau_key] * scale
        eps += d / (1 + (1j * w * tau) ** (1 - p[f"alf{n}"]))
    if include_ionic:
        eps += p["sig"] / (1j * w * EPS0)
    return eps


def cole_impedivity(f, rho0, rhoinf, fc, alpha) -> np.ndarray:
    """Cole impedivity rho*(f) = rho_inf + (rho0 - rho_inf) / (1 + (j f/fc)^(1-alpha)) in ohm*m."""
    f = np.asarray(f, dtype=float)
    return rhoinf + (rho0 - rhoinf) / (1 + (1j * f / fc) ** (1 - alpha))


@dataclass
class Material:
    key: str
    tissue: str
    pathology: str
    model: str
    params: dict
    valid_range_hz: tuple[float, float]
    provenance: dict = field(default_factory=dict)
    notes: str = ""

    def _raw_admittivity(self, f: np.ndarray) -> np.ndarray:
        w = 2 * np.pi * f
        p = self.params
        if self.model == "gabriel_cole_cole":
            return 1j * w * EPS0 * gabriel_complex_permittivity(f, p)
        if self.model == "cole_cole_single":
            # eps* = eps_inf + (eps_s - eps_inf)/(1 + (j w tau)^(1-alpha)) + sigma_dc/(j w eps0)
            eps = p["eps_inf"] + (p["eps_s"] - p["eps_inf"]) / (1 + (1j * w * p["tau_s"]) ** (1 - p["alpha"]))
            return p["sigma_dc_s_per_m"] + 1j * w * EPS0 * eps
        if self.model == "cole_impedivity":
            return 1.0 / cole_impedivity(f, p["rho0_ohm_m"], p["rhoinf_ohm_m"], p["fc_hz"], p["alpha"])
        if self.model == "constant":
            return p["sigma_s_per_m"] + 1j * w * EPS0 * p["eps_r"]
        if self.model == "tabulated":
            lf = np.log(np.asarray(p["freqs_hz"], float))
            s = PchipInterpolator(lf, np.log(np.asarray(p["sigma_s_per_m"], float)), extrapolate=True)
            e = PchipInterpolator(lf, np.log(np.asarray(p["eps_r"], float)), extrapolate=True)
            return np.exp(s(np.log(f))) + 1j * w * EPS0 * np.exp(e(np.log(f)))
        if self.model == "sigma_const_eps_gabriel":
            # Conductivity from a low-frequency measurement; only the real part of the
            # Gabriel permittivity (without its ionic term) supplies the capacitive part.
            eps_r = gabriel_complex_permittivity(f, p["gabriel"], include_ionic=False).real
            return p["sigma_s_per_m"] + 1j * w * EPS0 * eps_r
        raise ValueError(f"Unknown material model '{self.model}' for {self.key}")

    def admittivity(self, f, policy: str = "warn", sigma_scale: float = 1.0, eps_scale: float = 1.0):
        """Return (gamma, extrapolated_mask)."""
        f = np.atleast_1d(np.asarray(f, dtype=float))
        lo, hi = self.valid_range_hz
        extrap = (f < lo * (1 - 1e-9)) | (f > hi * (1 + 1e-9))
        if extrap.any():
            msg = (f"Material '{self.key}' evaluated outside its valid range "
                   f"[{lo:g}, {hi:g}] Hz at f={f[extrap].tolist()}")
            if policy == "error":
                raise ExtrapolationError(msg)
            if policy == "warn":
                warnings.warn(msg, stacklevel=2)
        g = self._raw_admittivity(f)
        g = sigma_scale * g.real + 1j * eps_scale * g.imag
        return g, extrap

    def sigma_eps(self, f, **kw):
        g, extrap = self.admittivity(f, **kw)
        w = 2 * np.pi * np.atleast_1d(np.asarray(f, float))
        return g.real, g.imag / w, extrap  # S/m, F/m


@dataclass
class MaterialLibrary:
    version: str
    materials: dict[str, Material]
    assignments: dict[str, dict[int, str]]
    label_names: dict[int, str]
    sha256: str
    path: str

    @classmethod
    def load(cls, path: str | Path) -> "MaterialLibrary":
        path = Path(path)
        text = path.read_text()
        d = yaml.safe_load(text)
        mats = {}
        for key, m in d["materials"].items():
            mats[key] = Material(
                key=key,
                tissue=m["tissue"],
                pathology=m.get("pathology", "normal"),
                model=m["model"],
                params=m["params"],
                valid_range_hz=tuple(float(x) for x in m["valid_range_hz"]),
                provenance=m.get("provenance", {}),
                notes=m.get("notes", ""),
            )
        assignments = {name: {int(k): v for k, v in a.items()} for name, a in d["assignments"].items()}
        for name, a in assignments.items():
            for lab, key in a.items():
                if key not in mats:
                    raise KeyError(f"Assignment '{name}' maps label {lab} to unknown material '{key}'")
        label_names = {int(k): v for k, v in d["labels"].items()}
        return cls(d["library_version"], mats, assignments, label_names,
                   hashlib.sha256(text.encode()).hexdigest(), str(path))

    def label_admittivity(self, assignment: dict[int, str], f: float, policy: str = "warn",
                          scales: dict[str, dict[str, float]] | None = None):
        """Admittivity per label at one frequency, plus the list of extrapolated materials."""
        scales = scales or {}
        for key, s in scales.items():
            if key not in self.materials:
                raise KeyError(f"materials.scales refers to unknown material '{key}'")
            bad = set(s) - {"sigma_scale", "eps_scale"}
            if bad:
                raise KeyError(f"materials.scales.{key}: unknown keys {sorted(bad)} (use sigma_scale / eps_scale)")
        out, flags = {}, []
        for lab, key in assignment.items():
            s = scales.get(key, {})
            g, ex = self.materials[key].admittivity(
                f, policy=policy, sigma_scale=s.get("sigma_scale", 1.0), eps_scale=s.get("eps_scale", 1.0))
            out[lab] = complex(g[0])
            if ex[0]:
                flags.append(key)
        return out, sorted(set(flags))

    def resolve_assignment(self, assignment: str | dict, overrides: dict | None = None) -> dict[int, str]:
        if isinstance(assignment, str):
            a = dict(self.assignments[assignment])
        else:
            a = {int(k): v for k, v in assignment.items()}
        for k, v in (overrides or {}).items():
            if v not in self.materials:
                raise KeyError(f"Unknown material '{v}'")
            a[int(k)] = v
        return a

    def summary_table(self, freqs) -> list[dict[str, Any]]:
        rows = []
        for key, m in self.materials.items():
            s, e, ex = m.sigma_eps(freqs, policy="allow")
            for fi, si, ei, xi in zip(np.atleast_1d(freqs), s, e, ex):
                rows.append(dict(material=key, tissue=m.tissue, pathology=m.pathology, freq_hz=float(fi),
                                 sigma_s_per_m=float(si), eps_r=float(ei / EPS0), extrapolated=bool(xi),
                                 data_type=m.provenance.get("data_type", "")))
        return rows
