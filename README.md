# MTOP: staged, explainable outcome prediction after mechanical thrombectomy

MTOP estimates the chance that a patient treated with mechanical thrombectomy (MT) for acute
ischaemic stroke will be **functionally independent (mRS 0–2) at 90 days**, and the chance of
death. The estimate is updated at four points along the care pathway, using only what is
known at each point:

| Stage | Adds |
|---|---|
| 1 · Admission (before MT) | age, sex, pre-stroke mRS, NIHSS, ASPECTS, occlusion site, side, onset-to-puncture, IVT, drip-and-ship, glucose, SBP, hypertension, diabetes, AF, previous stroke, anticoagulation |
| 2 · End of MT | final mTICI, procedure duration, first-line technique, acute stenting, anaesthesia |
| 3 · 24 hours | NIHSS at 24 h (and its change), symptomatic ICH |
| 4 · Day 7 | NIHSS on day 7, cerebral oedema |

The models come from the University Hospital Ostrava (FNO) thrombectomy registry (grant
SGS18/LF/2026, Faculty of Medicine, University of Ostrava). The web app runs entirely in the
browser: no server, no patient data, nothing typed in is sent anywhere.

> **Research tool, not a medical device.** Do not use it on its own to decide treatment.

**Live app:** https://viktorgajdosik.github.io/mt/ (published by GitHub Actions on every push to `main`).
It can be installed on a phone or laptop and works offline. The page is a single light-themed file
with its typeface bundled, so opening it contacts no third-party server.

## What the app contains

- **Calculator.** Enter a patient stage by stage and watch the estimate move: three-band bars
  (independent / dependent / dead) per stage, the factors that move the estimate (exact
  TreeSHAP, computed in the browser), a what-if curve for any numeric input, and a reality check
  ("of recent registry patients with a similar estimate, X % reached mRS 0–2"). Unknown inputs
  are averaged over the registry's values. English and Czech.
  *New in v1.1 (beta):* the full predicted 90-day mRS distribution with a 100-patient view,
  a bootstrap uncertainty range, the same patient scored by logistic regression and six published
  scores or predictor sets, and clearly marked previews of candidate predictors (GFAP, imaging).
- **Registry.** 90-day mRS distributions (Grotta bars) by 13 subgroups, outcome by time to
  puncture, two decades of trends (volume, age, reperfusion, technique, anaesthesia), data
  completeness by year, and Table 1. Aggregates only; groups under 10 patients are hidden.
- **Validation.** Temporal validation of every stage against logistic regression, XGBoost, an
  MLP and published scores, with ROC, calibration, decision curves, subgroup performance and an
  explanation of the post-2020 outcome drift.
- **Adaptive** *(v1.1 beta).* How the model is kept up to date: six update strategies compared year
  by year 2018–2025 (frozen, intercept or logistic recalibration, refit, refit + recalibration,
  sliding window), a drift monitor with control limits, data availability by year, a draft update
  policy and the model version history with its SHA-256 fingerprint.
- **Multicentre** *(v1.1 beta).* Any centre can validate MTOP on its own registry export inside the
  browser (CSV or Excel; nothing is uploaded) and export a results file with aggregate metrics
  only. Results from several sites are pooled in a random-effects meta-analysis (forest plot);
  until partners join, the four FNO treatment eras (internal–external validation) stand in.
- **Roadmap** *(v1.1).* What is live, what is new, what is in progress and what is proposed.
- **About.** Methods, limits, privacy, authors, QR code.

## Results (temporal validation, patients treated 2021–2025)

**mRS 0–2 at 90 days** — AUROC (95% CI)

| Stage | n (events) | LightGBM | Logistic regression | XGBoost | MLP | Published score |
|---|---|---|---|---|---|---|
| 1 · Admission | 378 (110) | 0.771 (0.72–0.82) | 0.737 (0.68–0.79) | 0.769 (0.72–0.82) | 0.726 (0.67–0.78) | THRIVE 0.677; SPAN-100 0.574 |
| 2 · End of MT | 370 (108) | 0.805 (0.76–0.85) | 0.765 (0.71–0.81) | 0.795 (0.74–0.84) | 0.745 (0.69–0.80) | THRIVE 0.673 |
| 3 · 24 h | 206 (73) | 0.859 (0.81–0.91) | 0.838 (0.78–0.89) | 0.856 (0.81–0.90) | 0.804 (0.74–0.86) | NIHSS 24 h 0.844 |
| 4 · Day 7 | 135 (62) | 0.896 (0.84–0.94) | 0.831 (0.75–0.90) | 0.898 (0.84–0.94) | 0.818 (0.74–0.89) | NIHSS day 7 0.882 |

**Death within 90 days** — AUROC (95% CI)

| Stage | n (events) | LightGBM | Logistic regression | XGBoost | MLP | Published score |
|---|---|---|---|---|---|---|
| 1 · Admission | 378 (76) | 0.780 (0.72–0.83) | 0.744 (0.68–0.80) | 0.774 (0.71–0.83) | 0.708 (0.64–0.77) | THRIVE 0.605; SPAN-100 0.565 |
| 2 · End of MT | 370 (73) | 0.805 (0.75–0.86) | 0.752 (0.68–0.82) | 0.800 (0.74–0.85) | 0.699 (0.63–0.77) | THRIVE 0.601 |
| 3 · 24 h | 206 (27) | 0.904 (0.84–0.95) | 0.883 (0.82–0.93) | 0.911 (0.85–0.96) | 0.857 (0.78–0.92) | NIHSS 24 h 0.852 |
| 4 · Day 7 | 135 (7) | 0.878 (0.74–0.97) | 0.856 (0.70–0.97) | 0.871 (0.72–0.97) | 0.892 (0.73–0.99) | NIHSS day 7 0.896 |

The day-7 mortality model rests on 7 deaths in the validation years; treat it as indicative only.

Calibration of the development model in 2021–2025 (mRS 0–2): 1 · Admission: intercept −0.51, slope 1.24, deployed shift −0.26; 2 · End of MT: intercept −0.60, slope 1.09, deployed shift −0.32; 3 · 24 h: intercept −0.74, slope 0.89, deployed shift −0.45; 4 · Day 7: intercept −0.75, slope 1.15, deployed shift −0.54.

**Keeping the model up to date** (admission stage, mRS 0–2, test years 2018–2025, models trained
only on earlier years): a model frozen in 2015 drifted (first signal 2019, 3σ alarm 2023) with
calibration-in-the-large −0.59 and AUROC 0.733; refitting on all earlier years and shifting the
intercept to the last two years (the policy used for v1.0) gave −0.03 and 0.768. See the Adaptive tab.

Full tables (Brier, calibration, ΔAUROC with paired-bootstrap CIs, subgroups) are produced by
`python -m mtop report` in `reports/validation_summary.md`.

## How it works

1. **Cleaning** (`src/mtop/clean.py`). Czech registry coding is mapped to standard variables.
   Every change is logged with the patient ID in `reports/data_quality.xlsx` so the source file
   can be corrected. Implausible values become missing (with a suggested correction, never applied
   silently). The only automatic repair is an isolated ±1-year typo in the onset date, detected
   from neighbouring patient IDs. **NIHSS 43 is the registry's code for death and 99 for "not
   assessable"**; they are never used as scores, because doing so would leak the outcome.
2. **Cohort.** Adults with endovascular treatment (diagnostic angiography only is excluded) and a
   known 90-day mRS. Stage 3 includes patients alive and assessable at 24 h, stage 4 at day 7, so
   later estimates are conditional on reaching that point.
3. **Models** (`src/mtop/train.py`). One LightGBM model per stage and outcome, with **monotone
   constraints** (for example a higher NIHSS or a longer time to puncture can never raise the
   chance of independence), tuned with Optuna on cross-validated log-loss in the development
   cohort only. Every tree leaf is guaranteed to summarise at least 20 patients.
4. **Validation.** Temporal split: development ≤ 2020, validation 2021–2025. Bootstrap 95 % CIs
   (2,000 resamples), calibration intercept and slope, ICI, decision curves, subgroup AUROCs.
5. **Deployment.** Models are refitted on all years. Because recent outcomes were worse than the
   development data predicted, each model's intercept is re-estimated on 2021–2025 patients from
   cross-validated predictions (temporal recalibration).
6. **Browser** (`web/`). The trees are exported as compact JSON. `web/model.js` reproduces
   LightGBM's decision rules and TreeSHAP exactly; `tests/test_parity.py` checks predictions and
   every SHAP value against LightGBM to 1e-6.

## Data protection

- Patient-level files never enter git: `data/raw/`, `data/processed/`, `artifacts/private/` and
  `reports/` are ignored, and CI fails if a tracked file could contain patient data
  (`scripts/check_no_patient_data.sh`).
- Only `artifacts/public/` is published: tree structure and leaf values (every leaf ≥ 20
  patients), aggregate statistics with small groups suppressed, and validation summaries.
  `tests/test_privacy.py` checks these rules before every deployment.
- The page makes no third-party requests: fonts and icons are bundled, and values typed into the
  calculator stay in the browser. The only exception is the multicentre tool, which loads the
  SheetJS library from cdnjs when a partner opens an Excel file (CSV needs nothing); the file
  itself is still read locally.
- Before making the repository or the page public, confirm with the FNO ethics committee / data
  protection officer that publishing the trained model and aggregate results is covered.

## Quick start

```bash
git clone https://github.com/viktorgajdosik/mt && cd mt
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# put the registry export at data/raw/mt_patients_master.xlsx, then:
python -m mtop all            # clean → train → export → explorer → report → site  (~15 min)
open site/index.html          # the app; works offline from the file

# no data at hand? use the synthetic registry
python -m mtop all --input data/synthetic/registry_synthetic.xlsx --quick
```

Individual steps: `python -m mtop clean | train [--quick] | export | extras [--quick] | adaptive | explorer | report | site | synth`.

## Updating with new patients

1. Add rows to the master Excel file and put it in `data/raw/`.
2. `python -m mtop all`, then read `reports/data_quality.xlsx` and fix the source where needed.
3. Check `reports/validation_summary.md`, then commit `artifacts/public/` and push. GitHub Actions
   re-runs the tests and redeploys the app.

## Repository layout

```
config/            features.yaml (predictors, stages, UI labels EN/CZ), settings.yaml
src/mtop/          clean · cohort · features · train · lr · metrics · scores · export · extras · adaptive · explorer · report · site · synth · cli
web/               index.html template, styles.css, app.js + view_*.js, charts.js, model.js (trees + TreeSHAP),
                   models_extra.js (comparators, mRS layer), metrics.js (in-browser validation), i18n.js,
                   assets/ (logo, app icons, bundled Instrument Sans font)
scripts/           check_no_patient_data.sh (CI guard), make_brand.py (regenerates logo and icons)
artifacts/public/  published model bundle, validation and explorer aggregates (committed)
data/synthetic/    synthetic registry for CI and demos (committed)
tests/             cleaning rules, JS↔LightGBM parity, privacy rules, end-to-end synthetic run
docs/              model card (TRIPOD+AI)
.github/workflows/ ci.yml (tests), pages.yml (build and deploy the app)
```

## Things worth knowing about the data

- ASPECTS is recorded from 2016 and anaesthesia type from about 2012.
- Since 2021 about half of patients have no 24-hour NIHSS, and day-7 NIHSS is often missing;
  missing 90-day mRS rose from about 3 % to about 20 %. Filling these gaps would strengthen the
  24-h and day-7 models more than any modelling change.
- The share of patients reaching mRS 0–2 fell from about 44 % (2015–2017) to 29 % (2021–2025),
  with more mRS 4–5 and unchanged mortality, beyond what age and severity explain.

## Citation and licence

Gajdošík V, Procházka V. *MTOP: staged, explainable prediction of 90-day outcome after
mechanical thrombectomy.* University Hospital Ostrava / Faculty of Medicine, University of
Ostrava, 2026. Code: MIT licence. The registry data are not included and not licensed.
The bundled typeface, Instrument Sans, is under the SIL Open Font License 1.1 (`web/assets/fonts/OFL.txt`).
