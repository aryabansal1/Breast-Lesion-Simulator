"""Block-majority downsample of an uncompressed VICTRE/breastPhantom label volume.

The 0.05 mm sample phantoms are 2-3 GB; the FEM never needs that resolution. This
streams the raw file through a memmap and writes a compact .npz (labels + header)
at an integer multiple of the original voxel size. Compression status is checked
from the file name/header: '_pc_' / '_pcl_' (compressed+cropped) files are refused.

Usage: python scripts/preprocess_victre.py dense_p_-321964974.mhd [raw path] out.npz --factor 10
"""

import argparse
import re
from pathlib import Path

import numpy as np

LABELS = np.array([0, 1, 2, 29, 33, 40, 88, 95, 125, 150, 200, 225, 250], dtype=np.uint8)


def read_mhd(path):
    h = {}
    for ln in Path(path).read_text().splitlines():
        if "=" in ln:
            k, v = ln.split("=", 1)
            h[k.strip()] = v.strip()
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mhd")
    ap.add_argument("raw", nargs="?")
    ap.add_argument("out")
    ap.add_argument("--factor", type=int, default=10)
    a = ap.parse_args()
    name = Path(a.mhd).name
    if re.search(r"_pcl?_", name) or "compress" in name.lower():
        raise SystemExit(f"{name}: compressed/cropped phantom refused (use the uncompressed '_p_' volume)")
    h = read_mhd(a.mhd)
    dims = [int(v) for v in h["DimSize"].split()]
    spacing = np.array([float(v) for v in h["ElementSpacing"].split()])
    offset = np.array([float(v) for v in h["Offset"].split()])
    assert h["ElementType"] == "MET_UCHAR"
    raw = a.raw or str(Path(a.mhd).with_name(h["ElementDataFile"]))
    if raw.endswith(".gz"):
        raise SystemExit("decompress the .raw.gz first (gunzip -k) so it can be memory-mapped")
    # MetaImage stores x fastest; numpy order (z, y, x) -> transpose to (x, y, z)
    vol = np.memmap(raw, dtype=np.uint8, mode="r", shape=tuple(dims[::-1]))
    k = a.factor
    nz, ny, nx = (d // k for d in dims[::-1])
    lut = np.full(256, len(LABELS), np.int64)
    lut[LABELS] = np.arange(len(LABELS))
    nl = len(LABELS) + 1
    out = np.zeros((nx, ny, nz), np.uint8)
    frac_tissue = np.zeros((nx, ny, nz), np.float32)
    yb = (np.arange(ny * k) // k)[None, :, None]
    xb = (np.arange(nx * k) // k)[None, None, :]
    for iz in range(nz):
        slab = np.asarray(vol[iz * k:(iz + 1) * k, :ny * k, :nx * k])
        c = lut[slab]
        block = (yb * nx + xb)
        cnt = np.bincount((np.broadcast_to(block, c.shape) * nl + c).ravel(), minlength=ny * nx * nl)
        cnt = cnt.reshape(ny, nx, nl)
        tissue = cnt[..., 1:len(LABELS)].sum(-1)
        frac_tissue[:, :, iz] = (tissue / k ** 3).T
        maj = cnt[..., 1:len(LABELS)].argmax(-1) + 1
        lab = np.where(tissue * 2 >= k ** 3, LABELS[np.minimum(maj, len(LABELS) - 1)], 0)
        # preserve thin skin: a tissue block containing any skin on an air interface keeps skin
        out[:, :, iz] = lab.T
        if iz % 20 == 0:
            print(f"slab {iz}/{nz}", flush=True)
    unknown = (np.asarray(vol[::max(k, 1) * 4, ::k * 4, ::k * 4]) [..., None] == LABELS).any(-1)
    if not unknown.all():
        print("WARNING: labels outside the breastPhantom table were found and treated as air")
    np.savez_compressed(a.out, labels=out, tissue_fraction=frac_tissue.astype(np.float16),
                        spacing_mm=spacing * k, offset_mm=offset + spacing * (k - 1) / 2,
                        source=name, source_spacing_mm=spacing, factor=k,
                        axes="x: chest wall->nipple, y: lateral, z: inferior->superior")
    u, n = np.unique(out, return_counts=True)
    print(dict(zip(u.tolist(), n.tolist())), out.shape)


if __name__ == "__main__":
    main()
