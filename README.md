# MTOP · Mechanical Thrombectomy Outcome Predictor

Staged, explainable outcome prediction after mechanical thrombectomy.

MTOP estimates the chance that a patient treated with mechanical thrombectomy (MT) for acute
ischaemic stroke will be **functionally independent (mRS 0–2) at 90 days**, and the chance of
death. The estimate is updated at four points along the care pathway, using only what is
known at each point:

| Stage | Adds |
|---|---|
| 1 · Before thrombectomy | the 33 pre-procedural predictors of the 2026 FNO analysis: age, sex, pre-stroke mRS, NIHSS, ASPECTS, occlusion site, side, IVT (and whether it was given elsewhere, onset to IVT, IVT to puncture), transfer, onset to puncture, onset hour (night), weekend, glucose (hyperglycaemia), systolic and diastolic BP (pulse pressure), BMI, hypertension, diabetes, hyperlipidaemia, arrhythmia, previous stroke and TIA, smoking, alcohol, statin, heart disease, antithrombotics |
| 2 · End of procedure | final mTICI, technique (aspiration, stent retriever, Solumbra, combined procedure, no thrombectomy, diagnostic angiography only), stenting, angioplasty, intra-arterial lysis, procedure duration, onset to recanalisation, anaesthesia |
| 3 · 24 hours | NIHSS at 24 h, its change, early neurological improvement, symptomatic and asymptomatic ICH |
| 4 · Day 7 | NIHSS on day 7, cerebral oedema, decompressive craniectomy |

Every procedure is included: patients who had stenting, angioplasty or intra-arterial lysis without
thrombectomy, and patients whose angiography showed nothing to treat, because before the procedure
nobody knows which it will be.

The models come from the University Hospital Ostrava (FNO) thrombectomy registry (grant
SGS18/LF/2026, Faculty of Medicine, University of Ostrava). The web app runs entirely in the
browser: no server, no patient data, nothing typed in is sent anywhere.

> **Research tool, not a medical device.** Do not use it on its own to decide treatment.

**Live app:** https://viktorgajdosik.github.io/mt/ (published by GitHub Actions on every push to `main`).
It can be installed on a phone or laptop and works offline. The page is a single file, light or
dark, with its typeface bundled, so opening it contacts no third-party server.

## What the app contains

- **Calculator.** Enter a patient stage by stage and watch the estimate move: three-band bars
  (independent / dependent / dead) per stage, the factors that move the estimate (exact
  contributions computed in the browser: TreeSHAP for the tree models, coefficient × value for the
  regressions), a what-if curve for any numeric input, and a reality check ("of recent registry
  patients with a similar estimate, X % reached mRS 0–2"). Unknown inputs are averaged over the
  registry's values. By default each stage uses the model chosen by the pre-specified rule; any
  other published model can be selected, and every model and published score is shown side by
  side for the same patient. Also the full 90-day mRS distribution with a 100-patient view and a
  bootstrap uncertainty range. English and Czech; light and dark themes (follows the system setting
  until one is chosen with the switch in the header).
- **Registry.** *Outcomes and trends:* 90-day mRS distributions (Grotta bars) by 21 subgroups,
  outcome by age and NIHSS, factors associated with the outcome (unadjusted and adjusted odds
  ratios, multiple imputation), outcome by time to puncture (and what each hour costs, adjusted),
  why outcomes were worse in 2023–2025, and two decades of trends.
  *Data and preprocessing:* the full data report in the style of the earlier MTOP pipeline:
  CONSORT-style patient flow and flow table, stage cohorts with exclusion reasons, every cleaning
  rule with the patients it touched (by variable), how undocumented registry codes were read, all
  48 registry columns with their role and the coding of every recorded value, the 49 model inputs
  with missing data overall, by split, by era, by sex and age and by outcome, 90-day mRS
  availability and who lacks it, distributions, why cholesterol is not used, and Table 1 by split
  and by outcome with standardised mean differences. Aggregates only; groups under 10 are hidden.
- **Validation.** Both temporal splits, every stage and outcome, in three groups: the eight
  candidates for the calculator, the learners validated for comparison only, and every published
  or bedside score the calculator shows at that stage. AUROC, ΔAUROC against the reference with its
  paired-bootstrap CI and DeLong's p, Brier, calibration intercept, slope and O/E (bootstrap CIs),
  nested cross-validation, the pre-specified model choice (Holm-adjusted, amendment 1), how each
  learner was tuned, ROC and decision curves, recalibration on newer patients for every model and
  score, SHAP for every model on the same test patients side by side, and for any one model:
  calibration, sensitivity/specificity/PPV/NPV at thresholds, subgroups with CIs and what the model
  relies on. Also how the results would change if the test-year patients without a 90-day outcome
  were counted.
- **Adaptive.** How the models are kept up to date: six update strategies compared year by year
  2018–2025 for every publishable learner, a drift monitor with control limits, leave-one-era-out
  validation, data availability by year, a draft update policy and the version history.
- **Multicentre.** *Learn together:* a working federated network in which hospitals train and
  validate shared models (boosted trees, regression with splines, linear regression) without
  moving patient data (secure aggregation; the result equals pooled training), demonstrated on four
  virtual centres made of FNO's own patients by treatment era ([docs/federated.md](docs/federated.md)).
  *Validate on your data:* any centre can validate every published MTOP model on its own registry
  export inside the browser and export aggregate results; results from several sites are pooled
  in a random-effects meta-analysis, model by model.
- **Standards.** TRIPOD+AI item by item (status, where it is addressed, what is left), TRIPOD+AI
  for Abstracts, a PROBAST+AI self-assessment, the Riley sample-size criteria for development and
  validation, the standards that apply later (TRIPOD-Cluster, DECIDE-AI, CONSORT-AI/SPIRIT-AI,
  FUTURE-AI), EU MDR and AI Act notes, and a prioritised to-do list.
- **Roadmap** and **About** (methods, limits, privacy, authors, QR code).

## Results

Temporal validation, primary split as in the 2026 FNO analysis: models developed on 1,225 patients
treated 2005–2022 and tested on 226 treated 2023–2025. AUROC (95% CI); **bold**: the model the
pre-specified rule chose for the calculator.

**mRS 0–2 at 90 days**

| Stage | n (events) | Regression + splines | Linear regression | LightGBM | XGBoost | MLP | Best published score |
|---|---|---|---|---|---|---|---|
| 1 · Before thrombectomy | 226 (60) | **0.809 (0.74–0.87)** | 0.803 (0.74–0.86) | 0.800 (0.73–0.86) | 0.797 (0.72–0.86) | 0.786 (0.72–0.85) | bedside items as a regression 0.794 |
| 2 · End of procedure | 221 (59) | **0.842 (0.79–0.90)** | 0.838 (0.78–0.89) | 0.829 (0.77–0.88) | 0.815 (0.75–0.87) | 0.816 (0.75–0.88) | FNO bedside score 0.776 |
| 3 · 24 hours | 120 (36) | **0.883 (0.82–0.94)** | 0.882 (0.82–0.94) | 0.890 (0.83–0.94) | 0.885 (0.82–0.94) | 0.889 (0.83–0.94) | MR PREDICTS@24H 0.904 |
| 4 · Day 7 | 139 (52) | **0.884 (0.83–0.93)** | 0.885 (0.83–0.93) | 0.921 (0.88–0.96) | 0.922 (0.88–0.96) | 0.858 (0.80–0.91) | NIHSS day 7 0.924 |

**Death within 90 days**

| Stage | n (events) | Regression + splines | Linear regression | LightGBM | XGBoost | MLP | Best published score |
|---|---|---|---|---|---|---|---|
| 1 · Before thrombectomy | 226 (40) | **0.734 (0.64–0.82)** | 0.726 (0.63–0.82) | 0.748 (0.66–0.83) | 0.761 (0.67–0.84) | 0.733 (0.64–0.82) | bedside items as a regression 0.760 |
| 2 · End of procedure | 221 (38) | **0.746 (0.65–0.83)** | 0.737 (0.64–0.83) | 0.777 (0.69–0.86) | 0.778 (0.69–0.86) | 0.742 (0.65–0.83) | FNO bedside score 0.686 |
| 3 · 24 hours | 120 (13) | **0.859 (0.73–0.95)** | 0.853 (0.71–0.95) | 0.860 (0.70–0.96) | 0.855 (0.70–0.96) | 0.852 (0.71–0.96) | MR PREDICTS@24H 0.849 |
| 4 · Day 7 | 139 (6) | **0.805 (0.57–0.98)** | 0.817 (0.55–0.98) | 0.837 (0.65–0.98) | 0.852 (0.68–0.97) | 0.812 (0.64–0.96) | NIHSS day 7 0.856 |

The rule (from the 2026 analysis): regression with splines is the reference, and another model
replaces it only if its paired 95% CI of the AUROC difference on the temporal test lies above zero,
or in nested cross-validation (5 folds × 3 repeats on all years, admission stage) it gains ≥ 0.01
AUROC on average and is better in ≥ 80% of folds, without worse calibration-in-the-large. Nested
cross-validation, mRS 0–2 before thrombectomy: splines 0.762, linear 0.758, LightGBM 0.756,
XGBoost 0.756, MLP 0.758. The test years are few (60 good outcomes, 6 deaths by day 7), so the
sensitivity split (development 2005–2020, test 2021–2025: 379 patients, 110 good outcomes) is also
reported; there the tree models lead at the later stages (day 7, mRS 0–2: LightGBM 0.917 vs
splines 0.813), which the Validation tab shows next to the primary split.

Calibration: the development models over-predicted good outcomes in 2023–2025 (regression with
splines before thrombectomy: calibration intercept −0.41, slope 1.47, O/E 0.78). The published
models are refitted on all years and their intercepts re-estimated on the pooled years 2023–2025.

**Keeping the model up to date** (regression with splines, before thrombectomy, mRS 0–2, test years
2018–2025, models trained only on earlier years): a model frozen in 2015 had calibration-in-the-large
−0.42 and AUROC 0.718; refitting on all earlier years and re-estimating the intercept on the last
years gave −0.03 and 0.769. Every learner is compared on the Adaptive tab.

`python -m mtop report` writes the full tables to `reports/validation_report.md` (local only).

## How it works

1. **Cleaning** (`src/mtop/clean.py`). Czech registry coding is mapped to standard variables.
   Every change is logged with the patient ID in `reports/data_quality.xlsx` so the source file
   can be corrected. Implausible values and text in numeric fields become missing (with a
   suggested correction, never applied silently). The only automatic repair is an isolated ±1-year
   typo in the onset date, detected from neighbouring patient IDs. Cross-field checks list records
   for review without changing them. **NIHSS 43 is read as death and 99 as "not assessable"**
   (70 of the 71 patients coded 43 on day 7 had died by 90 days); they are never used as scores,
   because doing so would leak the outcome. Impossible time orders (IVT after puncture,
   recanalisation before puncture) set that time to missing and keep the patient. Codings the
   export does not document are marked "our reading" on the Registry page, to be confirmed with
   the registry team. Occlusion site: tandem (ACI+ACM, ACI-odstup +T typ, ACC+ACI+ACM) is its own
   category, apart from cervical ICA/CCA (ACI-odstup, ACC) and intracranial ICA; every recorded
   value keeps its own detailed code, and a site needs at least 10 patients for its own model term
   (M3 counts with M2; the ACA, too rare and part of no modelled group, is not used by the models
   yet and gets no calculator estimate). Procedure: Solumbra (aspiration + stent retriever) and the
   combined procedure ("kombinovaný výkon", two or more techniques) are separate categories; any
   PENUMBRA + stent retriever device means Solumbra. Procedure labels that disagree with the
   extraction device, stent or IAT columns are listed for review.
2. **Cohort.** 1,577 registry records → 1,572 adults → 1,451 with a known 90-day mRS, whatever the
   procedure (88 diagnostic angiographies only). The end-of-procedure stage needs the final mTICI,
   the 24-h stage a 24-h NIHSS score and the day-7 stage a day-7 NIHSS score.
3. **Models** (`src/mtop/learners.py`, `train.py`, `extra_learners.py`, `compare.py`). Five learners
   per stage and outcome, each tuned with the same budget (50 candidate settings on the same
   cross-validation folds of the development years): logistic regression with restricted cubic
   splines (the reference), linear logistic regression, LightGBM with **monotone constraints**,
   XGBoost and an MLP. [Amendment 1](docs/amendment_1.md), committed before the runs, adds four
   candidates that can be published with exact explanations: an explainable boosting machine (EBM)
   with at least 20 patients per leaf, the reference with two interactions, the reference fitted on
   10 multiple imputations, and stacking of the reference, XGBoost and EBM. CatBoost, the default
   EBM, averaged probabilities and FT-Transformers (default and tuned) are validated for comparison
   only, like the MLP: nothing guarantees 20 patients behind their parts, or they have no exact
   explanation. TabPFN v2 and TabICL v2 are listed too and run only from local weight files.
4. **Validation.** Temporal splits (primary: test 2023–2025; sensitivity: test 2021–2025), nested
   cross-validation on all years, bootstrap 95% CIs (2,000 resamples) for AUROC, Brier,
   calibration intercept and slope and the AUROC difference against the reference, decision
   curves, thresholds and subgroups.
5. **Model choice.** The reference stays unless a candidate beats it on the temporal test with a
   Holm-adjusted paired-bootstrap p below 0.05 (over the seven alternatives), or in nested
   cross-validation gains ≥ 0.01 AUROC on average and is better in ≥ 80% of folds, without worse
   calibration-in-the-large.
6. **Deployment.** Every publishable learner (regression with splines, linear regression,
   LightGBM, XGBoost, EBM, regression with interactions, regression with multiple imputation,
   stacking) is refitted on all years with its development settings, and its intercept is
   re-estimated on the pooled years 2023–2025 from cross-validated predictions.
7. **Explanations** (`src/mtop/shap_all.py`). In the calculator, exact attributions for every
   published learner, computed in the browser. On the Validation tab, SHAP values of every
   validated model on the same test patients: exact where the model allows it, sampled
   (interventional, antithetic permutations) for the MLP and the averaged probabilities, expected
   gradients for the FT-Transformers.
8. **Browser** (`web/`). `web/model.js` reproduces every published model and its explanations
   exactly; `tests/test_parity.py` checks predictions and contributions against Python.

## Federated network

`python -m mtop.fed` runs the network itself (Python standard library plus `cryptography`):

```bash
pip install -e ".[federated]"
python -m mtop.fed enrol A B C                    # tokens for three centres; network.yaml keeps hashes only
python -m mtop.fed coordinator --network network.yaml --wait 3          # dashboard on :8750
python -m mtop.fed site --coordinator http://HOST:8750 --site-id A --token-file A.token --data a.csv
python -m mtop fed-demo                          # the four-centre demonstration behind the Federated tab
```

How it works, what leaves a hospital, its limits and how a centre joins: [docs/federated.md](docs/federated.md).

## Local data studio (FNO only)

`python -m mtop studio` opens the registry at every stage of the cleaning on the analysis computer:
the export as recorded, the cleaned registry, the analysed patients and the model inputs of each
stage. Column names are in English or Czech. Every value the cleaning set to missing, repaired or
flagged for review is marked in place, and any value can be corrected (apply the suggestion, type
the right value, or confirm it is correct). Corrections go to a corrections file with the author,
time and reason and are applied at the next cleaning; the Excel file is never overwritten. It is
password-protected, listens on this computer only and is not part of the public app. See
[docs/studio.md](docs/studio.md).

## Data protection

- Patient-level files never enter git: `data/raw/`, `data/processed/`, `data/studio/`,
  `artifacts/private/` and `reports/` are ignored, and CI fails if a tracked file could contain patient data
  (`scripts/check_no_patient_data.sh`).
- Only `artifacts/public/` is published: tree structure and leaf values (every leaf ≥ 20
  patients), aggregate statistics with small groups suppressed, and validation summaries.
  `tests/test_privacy.py` checks these rules before every deployment.
- The page makes no third-party requests: fonts and icons are bundled, and values typed into the
  calculator stay in the browser. The only exception is the multicentre tool, which loads the
  SheetJS library from cdnjs when a partner opens an Excel file (CSV needs nothing); the file
  itself is still read locally.
- The federated network never moves patient rows: centres send only masked sums, and the
  coordinator can read only network-wide totals (`docs/federated.md` lists exactly what it sees).
  `federated.json` holds aggregates for groups of at least 10 patients and 10 events.
- Before making the repository or the page public, confirm with the FNO ethics committee / data
  protection officer that publishing the trained model and aggregate results is covered.

## Quick start

```bash
git clone https://github.com/viktorgajdosik/mt && cd mt
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# put the registry export at data/raw/mt_patients_master.xlsx, then:
python -m mtop all            # clean → train → export → … → explorer → associations → report → site
open site/index.html          # the app; works offline from the file

# no data at hand? use the synthetic registry
python -m mtop all --input data/synthetic/registry_synthetic.xlsx --quick
```

Individual steps: `python -m mtop clean | nested | train [--split primary|sensitivity] | deploy | export | extras | adaptive | fed-demo | explorer | preprocessing | reporting | associations | analyses | shap | report | site | synth | studio`.

The amendment-1 learners run separately, in chunks that can go in parallel, after `train`
(`pip install -e ".[compare]"` adds CatBoost and the FT-Transformer):

```bash
python -m mtop compare --split primary --tag a        # every new learner, every stage (also --stage, --learner)
python -m mtop compare --split sensitivity --tag b
python -m mtop compare --nested 0-4 --tag c           # nested CV folds 1-5 (… 5-9, 10-14)
python -m mtop compare --merge                        # into the training results
python -m mtop deploy --learner EBM --learner "LR + interactions" --learner "LR + multiple imputation" --learner Stacking
python -m mtop shap && python -m mtop export && python -m mtop site
```

## Updating with new patients

1. Add rows to the master Excel file and put it in `data/raw/`.
2. `python -m mtop studio` to review the findings and record corrections (or read
   `reports/data_quality.xlsx`), then `python -m mtop all`, and the amendment-1 runs above (chunks
   computed on earlier training results are not merged).
3. Check `reports/validation_report.md`, then commit `artifacts/public/` and push. GitHub Actions
   re-runs the tests and redeploys the app.

## Repository layout

```
config/            features.yaml (predictors, stages, UI labels EN/CZ), settings.yaml, federated.yaml
src/mtop/          clean · corrections · cohort · features · learners · lgbm · tuning · lr · splines · treejson · train ·
                   nested · metrics · scores · export · extras · adaptive · explorer · preprocessing · reporting ·
                   associations · report · site · synth · studio · studio_app · studio_data · cli
src/mtop/fed/      federated network: schema · bins · secagg · gbdt · logreg · site · coordinator · server · client · demo
web/               index.html template, styles.css, app.js + view_*.js, charts.js, model.js (all models + explanations),
                   models_extra.js (comparators, mRS layer), metrics.js (in-browser validation), i18n.js,
                   assets/ (logo, app icons, bundled Instrument Sans font)
scripts/           check_no_patient_data.sh (CI guard), make_brand.py (regenerates logo and icons)
artifacts/public/  published model bundle, validation and explorer aggregates (committed)
data/synthetic/    synthetic registry for CI and demos (committed)
tests/             cleaning rules, JS↔Python parity for every model, privacy rules, federated network, end-to-end synthetic run
docs/              model card (TRIPOD+AI), data dictionary, federated network, local data studio
.github/workflows/ ci.yml (tests), pages.yml (build and deploy the app)
```

## Things worth knowing about the data

- ASPECTS is recorded from 2016 and anaesthesia type from about 2012.
- Since 2021 about half of patients have no 24-hour NIHSS, and day-7 NIHSS is often missing;
  missing 90-day mRS rose from about 3 % to about 20 % (21.5 % in 2023–2025). Filling these gaps
  would strengthen the 24-h and day-7 models more than any modelling change.
- The share of patients reaching mRS 0–2 fell to about 27 % in 2023–2025 (40 % in 2005–2022),
  beyond what age and severity explain.
- With 33 predictors before the procedure, the development data fall short of the Riley sample-size
  criteria for the regression (Standards tab); the models are penalised and validated on later
  years for that reason.
- Procedure labels and the device, stent and IAT columns disagree in several dozen records (for
  example "ADAPT" with a stent retriever as the device, or a thrombectomy label with the extraction
  system "žádný"). The label is kept, and the records are listed for review in the data studio.

## Citation and licence

Gajdošík V, Procházka V. *MTOP: staged, explainable prediction of 90-day outcome after
mechanical thrombectomy.* University Hospital Ostrava / Faculty of Medicine, University of
Ostrava, 2026. Code: MIT licence. The registry data are not included and not licensed.
The bundled typeface, Instrument Sans, is under the SIL Open Font License 1.1 (`web/assets/fonts/OFL.txt`).
