"""Fit per-sample Cole impedivity models to the UCI Breast Tissue features.

The UCI dataset (Jossinet group; doi:10.24432/C5P31H) does not publish spectra, only
features derived from 7-frequency spectra (15.625 kHz - 1 MHz). We recover a Cole
model rho*(f) = rho_inf + (rho0 - rho_inf)/(1 + (j f/fc)^(1-alpha)) per sample from:

  I0    -> rho0 (low-frequency-limit impedivity, used directly)
  DR    -> I0 - Re rho*(1 MHz)
  DA    -> |I0 - rho*(1 MHz)|            (interpretation; checked via fit residuals)
  MaxIP -> max over the 7 frequencies of -Im rho*(f)
  PA500 -> -arg rho*(500 kHz) in radians (interpretation; values 0.03-0.4)

Three unknowns (rho_inf, fc, alpha), four equations -> least squares. Residuals are
written out so poor fits can be excluded. Unit of I0 is stated as "ohm" in the dataset;
we treat it as ohm*cm (see docs/material_library.md for the plausibility argument).

Usage: python scripts/fit_uci_cole.py BreastTissue.xls data/materials/uci_cole_fits.csv
"""

import sys

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

FREQS = np.array([15.625e3, 31.25e3, 62.5e3, 125e3, 250e3, 500e3, 1000e3])


def model(f, r0, rinf, fc, alpha):
    return rinf + (r0 - rinf) / (1 + (1j * f / fc) ** (1 - alpha))


def fit_row(r):
    r0 = r.I0

    def resid(x):
        rinf, lfc, alpha = x
        fc = 10 ** lfc
        z1m = model(1e6, r0, rinf, fc, alpha)
        z = model(FREQS, r0, rinf, fc, alpha)
        z500 = model(500e3, r0, rinf, fc, alpha)
        return np.array([
            (r0 - z1m.real - r.DR) / r0,
            (abs(r0 - z1m) - r.DA) / r0,
            ((-z.imag).max() - r["Max IP"]) / r0,
            (-np.angle(z500) - r.PA500),
        ])

    best = None
    for lfc0 in (4.5, 5.0, 5.5, 6.0, 6.5):
        for a0 in (0.2, 0.4):
            x0 = [min(max(r0 - 1.5 * r.DR, 0.05 * r0), 0.99 * r0), lfc0, a0]
            try:
                s = least_squares(resid, x0, bounds=([1e-3 * r0, 3.0, 0.0], [r0, 8.0, 0.9]))
            except Exception:
                continue
            if best is None or s.cost < best.cost:
                best = s
    rinf, lfc, alpha = best.x
    return dict(rhoinf=rinf, fc_hz=10 ** lfc, alpha=alpha, rms_resid=float(np.sqrt(np.mean(best.fun ** 2))))


def main(xls, out):
    d = pd.read_excel(xls, sheet_name="Data")
    rows = []
    for _, r in d.iterrows():
        fr = fit_row(r)
        rows.append(dict(case=int(r["Case #"]), cls=r.Class, I0=r.I0, PA500=r.PA500, DA=r.DA, DR=r.DR,
                         MaxIP=r["Max IP"], rho0=r.I0, **fr))
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False, float_format="%.6g")
    good = df[df.rms_resid < 0.05]
    print(f"{len(good)}/{len(df)} fits with rms residual < 0.05")
    summ = good.groupby("cls").agg(
        n=("rho0", "size"), rho0_med=("rho0", "median"), rho0_q25=("rho0", lambda x: x.quantile(.25)),
        rho0_q75=("rho0", lambda x: x.quantile(.75)), rhoinf_med=("rhoinf", "median"),
        rhoinf_q25=("rhoinf", lambda x: x.quantile(.25)), rhoinf_q75=("rhoinf", lambda x: x.quantile(.75)),
        fc_med=("fc_hz", "median"), fc_q25=("fc_hz", lambda x: x.quantile(.25)),
        fc_q75=("fc_hz", lambda x: x.quantile(.75)), alpha_med=("alpha", "median"),
        alpha_q25=("alpha", lambda x: x.quantile(.25)), alpha_q75=("alpha", lambda x: x.quantile(.75)))
    summ.to_csv(out.replace(".csv", "_summary.csv"), float_format="%.6g")
    print(summ.to_string())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
