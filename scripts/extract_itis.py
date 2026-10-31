"""Extract Gabriel Cole-Cole parameters and LF conductivity rows from the IT'IS
Tissue Properties Database V5.0 ASCII export (doi:10.13099/VIP21000-05-0).

Usage: python scripts/extract_itis.py "Thermal_dielectric_acoustic_MR properties_database_V5.0(ASCII).txt"
Prints YAML for the tissues used by the simulator so values in
data/materials/library.yaml can be re-derived from the original download.
"""

import sys

import yaml

TISSUES = ["Breast Fat", "Breast Gland", "Skin", "Muscle", "Blood", "Connective Tissue", "Fat"]
GAB = ["ef", "del1", "tau1_ps", "alf1", "del2", "tau2_ns", "alf2", "sig",
       "del3", "tau3_us", "alf3", "del4", "tau4_ms", "alf4"]


def main(path):
    lines = open(path, encoding="latin1").read().splitlines()
    head = lines[2].split("\t")
    i_ef = head.index("ef")
    i_src = head.index("source")
    out = {}
    for ln in lines[3:]:
        c = ln.split("\t")
        if len(c) <= i_src + 5 or c[1].strip() not in TISSUES:
            continue
        name = c[1].strip()
        gab = {k: float(c[i_ef + n]) for n, k in enumerate(GAB)}
        lf = dict(substituted_from=c[i_src].strip(), average_s_per_m=float(c[i_src + 1]),
                  sd=float(c[i_src + 2] or "nan"), n=float(c[i_src + 3] or "nan"),
                  min=float(c[i_src + 4] or "nan"), max=float(c[i_src + 5] or "nan"))
        out[name] = dict(gabriel=gab, lf_conductivity=lf)
    print(yaml.safe_dump(out, sort_keys=False))


if __name__ == "__main__":
    main(sys.argv[1])
