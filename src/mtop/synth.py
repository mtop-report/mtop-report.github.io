"""Synthetic registry with the raw schema and Czech coding of the FNO export.

Every value is drawn from hand-set distributions; nothing is fitted to or copied from real
patients. Use it to run the whole pipeline (CI, demos, onboarding) without access to the data.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from .clean import RAW_COLUMNS
from .paths import SYNTHETIC


def _yn(rng, p, n):
    return np.where(rng.random(n) < p, "ANO", "NE").astype(object)


def make_synthetic(n: int = 600, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(70, 12, n).round(), 20, 95)
    sex = np.where(rng.random(n) < 0.53, "Muž", "Žena")
    nihss = np.clip(rng.normal(15, 6, n).round(), 0, 40)
    occl = rng.choice(["M1-ACM", "ACI-Intracranial", "ACI+ACM", "AB", "M2-ACM", "ACI-odstup", "AV"], n,
                      p=[0.45, 0.15, 0.1, 0.1, 0.1, 0.06, 0.04])
    post = np.isin(occl, ["AB", "AV"])
    hemi = np.where(post, "zadní povodí", np.where(rng.random(n) < 0.5, "levá hemisféra", "pravá hemisféra"))
    years = rng.integers(2010, 2026, n)
    dates = [dt.datetime(int(y), int(rng.integers(1, 13)), int(rng.integers(1, 28))) for y in years]
    aspects = np.where(years >= 2016, np.clip(rng.normal(8.5, 1.5, n).round(), 0, 10), np.nan)
    otp = np.clip(rng.gamma(4, 50, n).round(), 40, 1400)
    ivt = rng.random(n) < 0.65
    proc = rng.choice(["ADAPT", "Solumbra", "MT-stent retriever", "kombinovaný výkon", "Stent", "PAG"], n,
                      p=[0.3, 0.25, 0.2, 0.12, 0.08, 0.05])
    device = np.where(proc == "ADAPT", "PENUMBRA", np.where(proc == "MT-stent retriever", "Catch",
                      np.where(proc == "Solumbra", "PENUMBRA + Catch", np.where(proc == "PAG", "žádný", "Preset"))))
    tici = rng.choice(["0", "1", "2a", "2b", "3"], n, p=[0.06, 0.04, 0.08, 0.22, 0.60])
    tici_num = pd.Series(tici).map({"0": 0, "1": 1, "2a": 2, "2b": 3, "3": 4}).to_numpy()
    dur = np.clip(rng.gamma(3, 16, n).round(), 8, 200)
    glu = np.clip(rng.normal(7.3, 2.0, n), 3, 25).round(1)
    sbp = np.clip(rng.normal(150, 22, n).round(), 85, 240)
    htn = rng.random(n) < 0.78
    dm = rng.random(n) < 0.27
    af = rng.random(n) < 0.42
    # latent severity drives follow-up NIHSS and the 90-day mRS
    z = (-0.045 * (age - 70) - 0.09 * (nihss - 15) + 0.35 * (tici_num - 3) - 0.003 * (otp - 200)
         - 0.15 * (glu - 7.3) + 0.25 * ivt + 0.12 * np.nan_to_num(aspects - 8.5) + rng.normal(0, 0.9, n))
    n24 = np.clip((nihss * 0.6 - 3 * z + rng.normal(0, 3, n)).round(), 0, 42)
    n7 = np.clip((n24 * 0.75 - 2 * z + rng.normal(0, 2.5, n)).round(), 0, 42)
    mrs = np.clip(np.round(2.6 - 1.35 * z + rng.normal(0, 0.9, n)), 0, 6)
    died_early = (mrs == 6) & (rng.random(n) < 0.45)
    n7 = np.where(died_early, 43, n7)
    n24 = np.where(died_early & (rng.random(n) < 0.15), 43, n24)
    sich = np.where((rng.random(n) < 0.05) | ((mrs >= 5) & (rng.random(n) < 0.08)), "ANO", "NE")
    df = pd.DataFrame({
        "Patient ID": np.arange(1, n + 1),
        "Date of stroke onset": sorted(dates),
        "Time of stroke onset": [f"{int(h)}:{int(m):02d}" for h, m in zip(rng.integers(0, 24, n), rng.integers(0, 60, n))],
        "Age": age, "Sex": sex, "ASPECTS": aspects, "Occlusion site": occl, "Hemisphere": hemi,
        "Intervention procedure": proc,
        "Stent combo": np.where(proc == "Stent", "Wallstent", np.where(proc == "kombinovaný výkon", "Roadsaver", "žádný")),
        "Extraction system": device,
        "Thrombolytics": np.where(ivt, "Actilyse", "žádné"),
        "Procedure duration": dur, "TICI": tici,
        "IVT": np.where(ivt, "ANO", "NE"),
        "IVT in different hospital": np.where(ivt, np.where(rng.random(n) < 0.4, "Karviná", "FNO"), "žádné"),
        "Transfer from different hospital": np.where(rng.random(n) < 0.4, "Třinec", "FNO"),
        "IAT": _yn(rng, 0.05, n),
        "Stroke onset to IVT (min)": np.where(ivt, np.clip(otp - rng.integers(30, 90, n), 20, None), np.nan),
        "Stroke onset to puncture (min)": otp,
        "Stroke onset to recanalization (min)": np.where(tici_num >= 2, otp + dur, np.nan),
        "Admission NIHSS": nihss, "NIHSS 24h": n24, "NIHSS 7d": n7,
        "SICH": sich, "non-SICH": _yn(rng, 0.12, n),
        "mRS before": rng.choice([0, 0, 0, 0, 1, 1, 2], n).astype(float),
        "90-day mRS": mrs,
        "Anti-thrombotics before": rng.choice(["žádná", "AT-ASA", "NOAC-apixaban", "VKA-Warfarin"], n, p=[0.55, 0.25, 0.1, 0.1]),
        "Statins before": _yn(rng, 0.45, n), "Hypertension": np.where(htn, "ANO", "NE"),
        "DM": np.where(dm, "ANO", "NE"), "Hyperlipidemia": _yn(rng, 0.45, n), "Smoking": _yn(rng, 0.2, n),
        "Alcohol abusus": _yn(rng, 0.06, n), "BMI": np.clip(rng.normal(28, 4.5, n), 16, 50).round(2),
        "Arrhythmia": np.where(af, "ANO", "NE"), "TIA before": _yn(rng, 0.04, n), "CMP before": _yn(rng, 0.12, n),
        "Heart condition": rng.choice(["žádné", "ICHS", "Ao. chlopeň"], n, p=[0.65, 0.3, 0.05]),
        "Systolic BP": sbp, "Diastolic BP": np.clip(sbp * 0.55 + rng.normal(0, 8, n), 40, 140).round(),
        "Glycemia": glu, "Cholesterol": np.clip(rng.normal(4.6, 1.1, n), 2, 9).round(2),
        "Etiology": rng.choice(["kardioembolická", "aterotrombotická", "kryptogenní"], n, p=[0.55, 0.38, 0.07]),
        "Cerebral edema": _yn(rng, 0.15, n), "Decompression surgery": _yn(rng, 0.02, n),
        "Anesthesia": rng.choice(["Celková anestezie", "Analgosedace", "bez ARO"], n, p=[0.55, 0.35, 0.1]),
    })
    # realistic gaps
    for col, frac in [("90-day mRS", 0.06), ("NIHSS 24h", 0.12), ("Glycemia", 0.15), ("Anesthesia", 0.08)]:
        df.loc[rng.random(n) < frac, col] = np.nan
    return df[RAW_COLUMNS]


def write_synthetic(n: int = 600, seed: int = 1):
    SYNTHETIC.mkdir(parents=True, exist_ok=True)
    out = SYNTHETIC / "registry_synthetic.xlsx"
    make_synthetic(n, seed).to_excel(out, index=False, sheet_name="data")
    return out
