"""Load the lesion dataset (tensors.npz + records.csv) and write tiny synthetic datasets in the same schema."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class BreastDataset:
    """One row per simulated breast state.

    Z: complex [record, frequency, measurement] transfer impedance (ohm).
    baseline_index: index of the lesion-free record on the same mesh (itself for baselines).
    """

    Z: np.ndarray
    records: pd.DataFrame
    frequency_hz: np.ndarray
    measurement_id: np.ndarray
    measurement_type: np.ndarray
    measurement_electrodes: list
    baseline_index: np.ndarray
    lesion_present: np.ndarray
    composition: np.ndarray
    geometry_id: np.ndarray

    def __len__(self):
        return len(self.records)

    @property
    def baselines(self) -> np.ndarray:
        return np.flatnonzero(~self.lesion_present)

    @property
    def lesions(self) -> np.ndarray:
        return np.flatnonzero(self.lesion_present)

    def subset(self, idx) -> "BreastDataset":
        """Rows ``idx`` (must be closed under baseline_index)."""
        idx = np.asarray(idx)
        remap = -np.ones(len(self), int)
        remap[idx] = np.arange(len(idx))
        b = remap[self.baseline_index[idx]]
        if (b < 0).any():
            raise ValueError("subset drops the baseline of a kept lesion record")
        return BreastDataset(self.Z[idx], self.records.iloc[idx].reset_index(drop=True), self.frequency_hz,
                             self.measurement_id, self.measurement_type, self.measurement_electrodes, b,
                             self.lesion_present[idx], self.composition[idx], self.geometry_id[idx])


def _measurement_info(path: Path, mids):
    """(type per measurement, set of electrode ids per measurement) from measurement_definitions.csv."""
    f = path / "measurement_definitions.csv"
    md = pd.read_csv(f).set_index("measurement_id") if f.exists() else None
    types, elecs = [], []
    for m in mids:
        if md is not None and m in md.index:
            r = md.loc[m]
            types.append(r.measurement_type)
            elecs.append(sorted({r.inject_pos, r.inject_neg, r.sense_pos, r.sense_neg}))
        else:
            types.append("bipolar" if str(m).startswith("bipolar") else "tetrapolar")
            elecs.append([f"{m}_e{k}" for k in range(4)])  # unknown wiring: independent per measurement
    return np.array(types), elecs


def _tensors_from_csv(path: Path, rec: pd.DataFrame):
    z = pd.read_csv(path / "measurements.csv")
    freqs = np.sort(z.frequency_hz.unique())
    mids = np.array(list(dict.fromkeys(z.measurement_id)))
    Z = np.full((len(rec), len(freqs), len(mids)), np.nan + 0j)
    ri = {k: i for i, k in enumerate(rec.record_id)}
    mi = {k: i for i, k in enumerate(mids)}
    Z[z.record_id.map(ri).to_numpy(), np.searchsorted(freqs, z.frequency_hz),
      z.measurement_id.map(mi).to_numpy()] = (z.z_re_ohm + 1j * z.z_im_ohm).to_numpy()
    return Z, freqs, mids


def load_dataset(path) -> BreastDataset:
    """Load ``<path>/tensors.npz`` + ``records.csv`` (falls back to ``measurements.csv`` if the npz is missing)."""
    path = Path(path)
    rec = pd.read_csv(path / "records.csv", keep_default_na=False, na_values=[""])
    rec["lesion_present"] = rec.lesion_present.astype(str).str.lower().isin(["true", "1"])
    npz = path / "tensors.npz"
    if npz.exists():
        t = np.load(npz, allow_pickle=False)
        order = {k: i for i, k in enumerate(t["record_id"].astype(str))}
        idx = np.array([order[r] for r in rec.record_id])
        Z, freqs, mids = t["Z"][idx], t["frequency_hz"], t["measurement_id"].astype(str)
    else:
        Z, freqs, mids = _tensors_from_csv(path, rec)
    ri = {k: i for i, k in enumerate(rec.record_id)}
    base = rec.baseline_record_id.map(ri).to_numpy()
    ok = np.isfinite(Z).all(axis=(1, 2)) & (np.abs(Z) > 0).all(axis=(1, 2))
    if not ok.all():  # drop incomplete records, and lesion records whose baseline was dropped
        ok &= ok[base]
        keep = np.flatnonzero(ok)
        rec, Z = rec.iloc[keep].reset_index(drop=True), Z[keep]
        ri = {k: i for i, k in enumerate(rec.record_id)}
        base = rec.baseline_record_id.map(ri).to_numpy()
    mtype, melec = _measurement_info(path, mids)
    geom = rec["geometry_id"].to_numpy(str) if "geometry_id" in rec else rec.baseline_record_id.to_numpy(str)
    return BreastDataset(Z=Z.astype(np.complex128), records=rec, frequency_hz=np.asarray(freqs, float),
                         measurement_id=np.asarray(mids), measurement_type=mtype,
                         measurement_electrodes=melec, baseline_index=base.astype(int),
                         lesion_present=rec.lesion_present.to_numpy(bool),
                         composition=rec.composition.to_numpy(str), geometry_id=geom)


MEASUREMENTS = ["adj_E1E2_E3E4", "adj_E2E3_E4E1", "adj_E3E4_E1E2", "adj_E4E1_E2E3", "across_vertical",
                "bipolar_E1E3", "bipolar_E2E4"]
FREQS = [15625.0, 31250.0, 62500.0, 125000.0, 250000.0, 500000.0, 1000000.0]


def write_synthetic_dataset(out_dir, compositions=("fatty", "scattered", "hetero", "dense"), n_geometries=12,
                            materials=("carcinoma_a", "fibroadenoma_b"), effect_range=(1e-3, 1e-2),
                            mesh_jitter=2e-4, seed=0) -> Path:
    """Write a toy dataset with the generate_dataset.py schema (for tests / smoke runs; not physics).

    Each composition gets a Cole-like baseline spectrum per measurement; each geometry adds a small mesh
    perturbation; each lesion multiplies Z by (1 + effect * pattern) with a geometry-specific spatial pattern.
    """
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    f = np.array(FREQS)
    rec, rows = [], []
    for ci, comp in enumerate(compositions):
        r0 = rng.uniform(20, 60, len(MEASUREMENTS)) * (1 + 0.3 * ci)
        tau = 1 / (2 * np.pi * rng.uniform(5e4, 3e5))
        z0 = r0[None, :] * (0.6 + 0.4 / (1 + (1j * 2 * np.pi * f[:, None] * tau) ** 0.8))
        z0[:, 5:] += 80 * (1j * 2 * np.pi * f[:, None] * 1e-6) ** -0.7 / 50  # bipolar: contact term
        for g in range(n_geometries):
            gid = f"{comp}__shape{g % 3}__c{g}_s0p55_d{10 + 8 * (g % 2)}"
            zb = z0 * (1 + mesh_jitter * rng.standard_normal(z0.shape))
            pat = rng.standard_normal(len(MEASUREMENTS))
            spec = 1 + 0.5 * np.linspace(-1, 1, len(f))[:, None] * rng.standard_normal()
            base_id = f"{gid}__none"
            states = [(None, zb)]
            for m in materials:
                eff = rng.uniform(*effect_range) * (1 if "carcinoma" in m else -0.5)
                states.append((m, zb * (1 + eff * pat[None, :] * spec * (1 - 0.1j))))
            for m, z in states:
                rid = f"{gid}__{m or 'none'}"
                rec.append(dict(record_id=rid, geometry_id=gid, composition=comp, lesion_shape_id=f"shape{g % 3}",
                                lesion_clock_hour=float(g), lesion_depth_mm=10.0 + 8 * (g % 2),
                                lesion_present=m is not None, lesion_material=m or "",
                                lesion_class="none" if m is None else "x", baseline_record_id=base_id))
                for fi, fr in enumerate(f):
                    for mi, mid in enumerate(MEASUREMENTS):
                        rows.append((rid, fr, mid, z[fi, mi].real, z[fi, mi].imag))
    rec = pd.DataFrame(rec)
    z = pd.DataFrame(rows, columns=["record_id", "frequency_hz", "measurement_id", "z_re_ohm", "z_im_ohm"])
    zc = z.z_re_ohm + 1j * z.z_im_ohm
    z["z_mag_ohm"], z["z_phase_deg"] = np.abs(zc), np.degrees(np.angle(zc))
    rec.to_csv(out / "records.csv", index=False)
    z.to_csv(out / "measurements.csv", index=False, float_format="%.12g")
    pd.DataFrame(dict(measurement_id=MEASUREMENTS, measurement_type=["tetrapolar"] * 5 + ["bipolar"] * 2,
                      inject_pos=["E1", "E2", "E3", "E4", "E1", "E1", "E2"],
                      inject_neg=["E2", "E3", "E4", "E1", "E3", "E3", "E4"],
                      sense_pos=["E3", "E4", "E1", "E2", "E2", "E1", "E2"],
                      sense_neg=["E4", "E1", "E2", "E3", "E4", "E3", "E4"])).to_csv(
        out / "measurement_definitions.csv", index=False)
    ri = {k: i for i, k in enumerate(rec.record_id)}
    Z = np.zeros((len(rec), len(f), len(MEASUREMENTS)), complex)
    Z[z.record_id.map(ri).to_numpy(), np.searchsorted(f, z.frequency_hz),
      z.measurement_id.map({k: i for i, k in enumerate(MEASUREMENTS)}).to_numpy()] = zc.to_numpy()
    np.savez_compressed(out / "tensors.npz", Z=Z, record_id=rec.record_id.to_numpy(str),
                        baseline_index=rec.baseline_record_id.map(ri).to_numpy(),
                        lesion_present=rec.lesion_present.to_numpy(bool), composition=rec.composition.to_numpy(str),
                        frequency_hz=f, measurement_id=np.array(MEASUREMENTS))
    return out
