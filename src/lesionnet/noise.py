"""Nuisance model for left/right bioimpedance pairs (what the simulator does not produce).

The simulator gives noise-free spectra and a mirrored (identical) contralateral breast. Every breast that goes into
a pair is passed through ``NoiseModel.perturb`` independently; the contralateral side additionally gets an
anatomical-asymmetry term. All magnitudes are relative (fraction of Z) and are *assumptions* to be calibrated
against hardware / volunteer data; see docs/nn_design.md section 4.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import numpy as np


@dataclass
class NoiseModel:
    meas_sigma: float = 1e-4            # complex white noise per reading (relative); 1e-4 = 80 dB SNR
    placement_sigma: float = 2e-3       # per-electrode re-placement log-gain (frequency-flat)
    placement_phase_frac: float = 0.1   # imaginary part of the placement log-gain relative to the real part
    contact_sigma: float = 2e-2         # contact-impedance drift on bipolar readings (relative to |Z| at f_ref)
    contact_alpha: tuple = (0.5, 0.9)   # constant-phase-element exponent range of the drift term
    contact_tetrapolar_leak: float = 0.01  # fraction of the drift seen by tetrapolar readings (finite CMRR etc.)
    anat_sigma: float = 3e-3            # left/right anatomical asymmetry (rms relative change, contralateral side)
    anat_smooth_frac: float = 0.5       # share of asymmetry from random smooth spectra (rest: anatomy directions)
    scale: float = 1.0                  # global multiplier on every sigma (noise sweeps)
    scale_jitter: tuple = (1.0, 1.0)    # per-pair log-uniform multiplier range (training-time domain randomisation)

    @classmethod
    def from_dict(cls, d: dict | None) -> "NoiseModel":
        d = dict(d or {})
        names = {f.name for f in fields(cls)}
        bad = set(d) - names
        if bad:
            raise KeyError(f"unknown noise keys {sorted(bad)}")
        for k in ("contact_alpha", "scale_jitter"):
            if k in d:
                d[k] = tuple(float(x) for x in d[k])
        return cls(**d)

    def to_dict(self) -> dict:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}

    def scaled(self, s: float) -> "NoiseModel":
        d = asdict(self)
        d["scale"] = self.scale * s
        return NoiseModel(**d)

    def replace(self, **kw) -> "NoiseModel":
        return NoiseModel(**{**asdict(self), **kw})

    def _pair_scale(self, n, rng):
        lo, hi = self.scale_jitter
        if hi <= lo:
            return np.full(n, self.scale * lo)
        return self.scale * np.exp(rng.uniform(np.log(lo), np.log(hi), n))

    def perturb(self, Z, rng, frequency_hz, measurement_type, measurement_electrodes, scale=None):
        """Device / electrode nuisance for a batch of breasts. Z: complex [n, F, M]. Returns a new array."""
        n, F, M = Z.shape
        s = self._pair_scale(n, rng) if scale is None else np.asarray(scale, float)
        s3 = s[:, None, None]
        logz = np.log(Z)
        # electrode re-placement: one log-gain per electrode, shared by every reading that uses it
        elec = sorted({e for es in measurement_electrodes for e in es})
        A = np.zeros((len(elec), M))
        for m, es in enumerate(measurement_electrodes):
            for e in es:
                A[elec.index(e), m] = 1 / np.sqrt(len(es))
        g = rng.standard_normal((n, len(elec))) @ A
        logz = logz + s3 * self.placement_sigma * (1 + 1j * self.placement_phase_frac) * g[:, None, :]
        out = np.exp(logz)
        # contact-impedance drift: additive constant-phase-element term, mainly on bipolar readings
        bip = np.asarray(measurement_type) == "bipolar"
        w = np.where(bip, 1.0, self.contact_tetrapolar_leak)
        f = np.asarray(frequency_hz, float)
        f_ref = np.sqrt(f.min() * f.max())
        alpha = rng.uniform(*self.contact_alpha, size=(n, 1, M))
        cpe = (f[None, :, None] / f_ref) ** (-alpha) * np.exp(-0.5j * np.pi * alpha)
        iref = int(np.argmin(np.abs(f - f_ref)))
        delta = rng.standard_normal((n, 1, M)) * self.contact_sigma * w[None, None, :]
        out = out + s3 * delta * np.abs(Z[:, iref:iref + 1, :]) * cpe
        # instrument noise
        e = (rng.standard_normal(Z.shape) + 1j * rng.standard_normal(Z.shape)) / np.sqrt(2)
        return out * (1 + s3 * self.meas_sigma * e)

    def asymmetry(self, n, rng, frequency_hz, anat_dirs=None, scale=None):
        """Multiplicative left/right anatomical asymmetry factors exp(beta*u), complex [n, F, M].

        u mixes (a) random combinations of ``anat_dirs`` (unit-rms log-spectrum differences between distinct
        anatomies) and (b) random spectra quadratic in log-frequency, each normalised to unit rms.
        """
        s = self._pair_scale(n, rng) if scale is None else np.asarray(scale, float)
        t = np.log(np.asarray(frequency_hz, float))
        t = (t - t.mean()) / (np.ptp(t) / 2 if np.ptp(t) > 0 else 1.0)
        basis = np.stack([np.ones_like(t), t, t ** 2 - (t ** 2).mean()])  # [3, F]
        if anat_dirs is None or anat_dirs.ndim != 3:
            raise ValueError("asymmetry() needs anat_dirs with shape [K, F, M] (K may be 0)")
        M = anat_dirs.shape[-1]
        c = rng.standard_normal((n, 3, M)) + 0.3j * rng.standard_normal((n, 3, M))
        smooth = _unit_rms(np.einsum("kf,nkm->nfm", basis, c))
        if anat_dirs.shape[0] > 0:
            a = rng.standard_normal((n, anat_dirs.shape[0]))
            struct = _unit_rms(np.einsum("nk,kfm->nfm", a, anat_dirs))
            u = np.sqrt(self.anat_smooth_frac) * smooth + np.sqrt(1 - self.anat_smooth_frac) * struct
        else:
            u = smooth
        beta = rng.standard_normal(n) * s * self.anat_sigma
        return np.exp(beta[:, None, None] * u)


def _unit_rms(x):
    r = np.sqrt(np.mean(np.abs(x) ** 2, axis=(1, 2), keepdims=True))
    return x / np.where(r > 0, r, 1.0)


def anatomy_directions(Z, composition, baselines):
    """Unit-rms log-spectrum differences between the mean baselines of each pair of compositions: [K, F, M]."""
    comps = sorted(set(composition[baselines]))
    means = {c: np.log(Z[baselines[composition[baselines] == c]]).mean(0) for c in comps}
    dirs = [means[a] - means[b] for i, a in enumerate(comps) for b in comps[i + 1:]]
    if not dirs:
        return np.zeros((0, *Z.shape[1:]), complex)
    return _unit_rms(np.stack(dirs))
