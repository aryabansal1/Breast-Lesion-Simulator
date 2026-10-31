"""Regenerate data/materials/library.yaml from the extracted source tables.

Inputs (both committed):
  data/materials/uci_cole_fits_summary.csv  (scripts/fit_uci_cole.py)
  data/materials/itis_v5_extract.yaml       (scripts/extract_itis.py)
"""

from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1] / "data" / "materials"
ITIS_DOI = "10.13099/VIP21000-05-0"
UCI_DOI = "10.24432/C5P31H"
UCI_NAMES = {"adi": ("adipose", "normal"), "gla": ("glandular", "normal"), "con": ("connective", "normal"),
             "car": ("carcinoma", "malignant"), "fad": ("fibroadenoma", "benign"), "mas": ("mastopathy", "benign")}


def plain(o):
    if isinstance(o, dict):
        return {k: plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [plain(v) for v in o]
    if hasattr(o, "item"):
        return o.item()
    return o


def main():
    uci = pd.read_csv(ROOT / "uci_cole_fits_summary.csv", index_col=0)
    itis = yaml.safe_load((ROOT / "itis_v5_extract.yaml").read_text())
    mats = {}
    for code, (tissue, path) in UCI_NAMES.items():
        r = uci.loc[code]
        mats[f"{tissue}_uci"] = dict(
            tissue=tissue, pathology=path, model="cole_impedivity",
            params=dict(rho0_ohm_m=round(r.rho0_med / 100, 5), rhoinf_ohm_m=round(r.rhoinf_med / 100, 5),
                        fc_hz=float(f"{r.fc_med:.4g}"), alpha=round(r.alpha_med, 4)),
            valid_range_hz=[15.625e3, 1.0e6],
            provenance=dict(
                source="UCI Machine Learning Repository, Breast Tissue (Jossinet group, Univ. Lyon)",
                doi=UCI_DOI, related_publication="Jossinet 1996 Med Biol Eng Comput; Jossinet 1998 Physiol Meas "
                "doi:10.1088/0967-3334/19/1/006", dataset_fields="I0, PA500, DA, DR, Max IP",
                data_type="fitted (Cole model fitted per sample to derived spectral features; class median "
                          "parameters)", n_samples=int(r.n), temperature="not stated (ex vivo, freshly excised)",
                in_vivo="ex vivo", measurement_geometry="Jossinet impedance probe, 7 frequencies 15.625 kHz-1 MHz",
                variability=dict(rho0_ohm_m_iqr=[r.rho0_q25 / 100, r.rho0_q75 / 100],
                                 rhoinf_ohm_m_iqr=[r.rhoinf_q25 / 100, r.rhoinf_q75 / 100],
                                 fc_hz_iqr=[r.fc_q25, r.fc_q75], alpha_iqr=[r.alpha_q25, r.alpha_q75]),
                units_assumption="I0 interpreted as ohm*cm (dataset says 'ohm'); see docs/material_library.md"),
            notes="Impedivity is an intrinsic (geometry-normalised) property, so it can be assigned to a region; "
                  "per-sample Cole fits are under-determined at high frequency (rho_inf) — use the IQR for "
                  "sensitivity studies.")
    itis_map = {"Breast Fat": ("fat_itis", "adipose"), "Breast Gland": ("gland_itis", "glandular"),
                "Skin": ("skin_itis", "skin"), "Muscle": ("muscle_itis", "muscle"),
                "Blood": ("blood_itis", "blood"), "Connective Tissue": ("connective_itis", "connective")}
    for name, (key, tissue) in itis_map.items():
        e = itis[name]
        lf = e["lf_conductivity"]
        prov = dict(source=f"IT'IS Tissue Properties Database V5.0, '{name}'", doi=ITIS_DOI,
                    data_type="fitted (Gabriel 1996 4-term Cole-Cole parameters)",
                    original_reference="Gabriel S, Lau RW, Gabriel C 1996 Phys Med Biol 41:2271",
                    temperature="mostly 37 C (Gabriel compilation)", in_vivo="mixed, mostly ex vivo")
        if name == "Breast Gland":
            prov["warning"] = "IT'IS LF conductivity for breast gland is substituted from thyroid gland"
        mats[key] = dict(tissue=tissue, pathology="normal", model="gabriel_cole_cole", params=e["gabriel"],
                         valid_range_hz=[1.0e3, 1.0e9], provenance=prov,
                         notes="Gabriel parametrisation is least reliable below ~1 MHz; LF tail is fit, not data.")
        mats[key + "_lf"] = dict(
            tissue=tissue, pathology="normal", model="sigma_const_eps_gabriel",
            params=dict(sigma_s_per_m=lf["average_s_per_m"], gabriel=e["gabriel"]),
            valid_range_hz=[1.0, 1.0e6],
            provenance=dict(prov, data_type="measured LF conductivity (IT'IS average) + Gabriel permittivity",
                            lf_substituted_from=lf["substituted_from"], lf_sd=lf["sd"], lf_n=lf["n"]),
            notes="Conductivity from the IT'IS low-frequency compilation (frequency independent); permittivity "
                  "from the real part of the Gabriel model only, so losses are not double counted.")
    mats["skin_itis_lf"]["notes"] += " Used as default skin: approximates hydrated skin under gel electrodes."
    mats["carcinoma_halter"] = dict(
        tissue="carcinoma", pathology="malignant", model="constant",
        params=dict(sigma_s_per_m=0.145, eps_r=0.0), valid_range_hz=[1.0e4, 1.0e4],
        provenance=dict(source="Halter et al. 2009, Table (IDCa/DCIS, 10 kHz)", pmcid="PMC2792899",
                        data_type="measured (ex vivo probe, mean)", sd=0.012, in_vivo="ex vivo",
                        temperature="room temperature (ex vivo)",
                        measurement_geometry="bipolar ex vivo probe"),
        notes="Single-frequency conductivity. eps_r=0 is an ASSUMPTION (unknown at 10 kHz). Use only at 10 kHz.")
    sur = pd.read_csv(ROOT / "surowiec1988_table1.csv", comment="#")
    # The paper describes peripheral / low-tumour-involvement samples as having LF conductivity
    # below ~1 mS/cm and eps_s < ~500-800 at 100 kHz (examples E1, F3, G1, G2). We split on
    # sigma_dc <= 1.05 mS/cm and take medians of each group.
    groups = {"carcinoma_surowiec": (sur.sigma_dc_mS_per_cm > 1.05, "carcinoma", "malignant",
                                     "samples with sigma_dc > 1.05 mS/cm (tumour centre and infiltrating margin)"),
              "peritumoral_low_surowiec": (sur.sigma_dc_mS_per_cm <= 1.05, "peritumoral", "mixed",
                                           "samples with sigma_dc <= 1.05 mS/cm (peripheral, low tumour involvement)")}
    for key, (sel, tissue, path, desc) in groups.items():
        g = sur[sel]
        mats[key] = dict(
            tissue=tissue, pathology=path, model="cole_cole_single",
            params=dict(eps_s=float(g.eps_s.median()), eps_inf=float(g.eps_inf.median()),
                        tau_s=float(g.tau_us.median()) * 1e-6, alpha=float(g.alpha.median()),
                        sigma_dc_s_per_m=float(g.sigma_dc_mS_per_cm.median()) * 0.1),
            valid_range_hz=[1.0e5, 1.0e8],
            provenance=dict(source="Surowiec et al. 1988, IEEE Trans Biomed Eng 35(4), Table I (p. 262)",
                            doi="not recorded in supplied PDF",
                            data_type=f"fitted (Cole-Cole per sample; median over {len(g)} {desc})",
                            n_samples=int(len(g)), temperature="37 C", in_vivo="ex vivo",
                            measurement_geometry="coaxial/parallel sample holder (see paper Sec. II)",
                            variability=dict(sigma_dc_s_per_m_range=[float(g.sigma_dc_mS_per_cm.min()) * 0.1,
                                                                     float(g.sigma_dc_mS_per_cm.max()) * 0.1],
                                             eps_s_range=[float(g.eps_s.min()), float(g.eps_s.max())])),
            notes="Fit range 100 kHz-100 MHz (measured from 20 kHz). Median of per-sample parameters is not the "
                  "same as the median spectrum; group split is our classification of the table.")
    mats["perfect_insulator_test"] = dict(
        tissue="test", pathology="none", model="constant", params=dict(sigma_s_per_m=1e-9, eps_r=1.0),
        valid_range_hz=[0.0, 1e12], provenance=dict(data_type="assumed (verification only)"))
    labels = {0: "air", 1: "fat", 2: "skin", 29: "glandular", 33: "nipple", 40: "muscle", 88: "ligament",
              95: "TDLU", 125: "duct", 150: "artery", 225: "vein", 200: "tumour", 250: "calcification",
              41: "chest_wall_extension"}
    common = {2: "skin_itis_lf", 33: "skin_itis_lf", 40: "muscle_itis_lf", 88: "connective_itis_lf",
              150: "blood_itis_lf", 225: "blood_itis_lf", 41: "muscle_itis_lf"}
    assignments = {
        "uci_breast": {**common, 1: "adipose_uci", 29: "glandular_uci", 95: "glandular_uci", 125: "glandular_uci",
                       250: "glandular_uci", 200: "carcinoma_uci"},
        "itis_only": {**common, 1: "fat_itis", 29: "gland_itis", 95: "gland_itis", 125: "gland_itis",
                      250: "gland_itis", 200: "carcinoma_uci"},
    }
    out = dict(library_version="0.1.0", labels=labels, assignments=assignments, materials=mats)
    (ROOT / "library.yaml").write_text(
        "# GENERATED by scripts/build_library.py — edit the generator, not this file.\n"
        + yaml.safe_dump(plain(out), sort_keys=False, width=110))


if __name__ == "__main__":
    main()
