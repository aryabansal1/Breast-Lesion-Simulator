"""Impedance-spectrum plots and 3D field/anatomy renders."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def plot_spectra(out_dir, measurements=None):
    out_dir = Path(out_dir)
    df = pd.read_csv(out_dir / "measurements.csv")
    if measurements:
        df = df[df.measurement_id.isin(measurements)]
    else:
        df = df[df.measurement_type != "adjacent"] if (df.measurement_type != "adjacent").any() else df
    files = []
    for side, ds in df.groupby("side"):
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        for (mid, case), g in ds.groupby(["measurement_id", "case"]):
            g = g.sort_values("frequency_hz")
            ls = "-" if case == "absent" else "--"
            lab = f"{mid} ({case})"
            ax[0].semilogx(g.frequency_hz, g.z_mag_ohm, ls, marker="o", ms=3, label=lab)
            ax[1].semilogx(g.frequency_hz, g.z_phase_deg, ls, marker="o", ms=3, label=lab)
        ax[0].set(xlabel="frequency (Hz)", ylabel="|Z_transfer| (ohm)", title=f"{side}: magnitude")
        ax[1].set(xlabel="frequency (Hz)", ylabel="phase (deg)", title=f"{side}: phase")
        for a in ax:
            a.grid(True, which="both", alpha=0.3)
        ax[1].legend(fontsize=7, ncol=1, loc="best")
        fig.tight_layout()
        f = out_dir / f"spectra_{side}.png"
        fig.savefig(f, dpi=130)
        plt.close(fig)
        files.append(f)
    diff_path = out_dir / "tumour_difference.csv"
    if diff_path.exists():
        d = pd.read_csv(diff_path)
        if measurements:
            d = d[d.measurement_id.isin(measurements)]
        for side, ds in d.groupby("side"):
            fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
            for mid, g in ds.groupby("measurement_id"):
                g = g.sort_values("frequency_hz")
                ax[0].semilogx(g.frequency_hz, 100 * g.rel_d_mag, marker="o", ms=3, label=mid)
                ax[1].semilogx(g.frequency_hz, g.d_phase_deg, marker="o", ms=3, label=mid)
            ax[0].set(xlabel="frequency (Hz)", ylabel="(|Z_present| - |Z_absent|)/|Z_absent| (%)",
                      title=f"{side}: tumour effect on magnitude")
            ax[1].set(xlabel="frequency (Hz)", ylabel="phase_present - phase_absent (deg)",
                      title=f"{side}: tumour effect on phase")
            for a in ax:
                a.grid(True, which="both", alpha=0.3)
            ax[0].legend(fontsize=6, ncol=2)
            fig.tight_layout()
            f = out_dir / f"tumour_effect_{side}.png"
            fig.savefig(f, dpi=130)
            plt.close(fig)
            files.append(f)
    return files


def render_fields(out_dir, side="left", case="present", window_size=(1400, 600)):
    """Off-screen PyVista render: tissue labels + tumour + electrodes | |J| on a clipped half."""
    import pyvista as pv

    pv.OFF_SCREEN = True
    out_dir = Path(out_dir)
    grid = pv.read(out_dir / f"fields_{side}_{case}.vtu")
    elec = pv.read(out_dir / f"electrodes_{side}.vtu")
    tissue = grid.threshold([0.5, 1e9], scalars="label")
    clipped = tissue.clip(normal="z", origin=tissue.center, invert=True)
    pl = pv.Plotter(shape=(1, 2), window_size=window_size, off_screen=True)
    pl.subplot(0, 0)
    pl.add_text("tissue labels (clipped), tumour, electrodes", font_size=9)
    pl.add_mesh(clipped, scalars="label", cmap="tab20", show_scalar_bar=True, scalar_bar_args=dict(title="label"))
    if "tumour" in grid.cell_data and grid.cell_data["tumour"].any():
        pl.add_mesh(grid.threshold([0.5, 1.5], scalars="tumour"), color="red")
    pl.add_mesh(elec, color="gold")
    pl.view_isometric()
    pl.subplot(0, 1)
    pl.add_text("|J| (A/m^2), log scale, clipped", font_size=9)
    j = np.asarray(clipped.cell_data["J_abs_A_per_m2"])
    pl.add_mesh(clipped, scalars="J_abs_A_per_m2", log_scale=True, cmap="inferno",
                clim=[max(j[j > 0].min(), j.max() * 1e-4), j.max()], scalar_bar_args=dict(title="|J|"))
    pl.add_mesh(elec, color="cyan")
    pl.view_isometric()
    pl.link_views()
    f = out_dir / f"render_{side}_{case}.png"
    pl.screenshot(str(f))
    pl.close()
    return f
