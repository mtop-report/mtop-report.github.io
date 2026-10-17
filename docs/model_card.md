# MTOP model card

Structured after TRIPOD+AI (Collins et al., BMJ 2024). Numbers are filled from
`artifacts/public/validation.json`; regenerate with `python -m mtop report`.

## 1. Purpose and intended use

- **Predicts:** probability of functional independence (modified Rankin Scale 0–2) at 90 days
  after mechanical thrombectomy for acute ischaemic stroke (primary), and of death by 90 days
  (secondary).
- **When:** four prediction moments: before groin puncture, at the end of the procedure,
  24 hours after treatment (alive and assessable), and day 7 (alive and assessable).
- **Who for:** research, teaching, prognostic discussion and trial planning at FNO Ostrava.
- **Not for:** deciding whether to treat an individual patient, or use outside the population it
  was developed in, unless it has been externally validated there. It is not a medical device under
  EU MDR 2017/745.

## 2. Data

- Source: prospective mechanical-thrombectomy registry, University Hospital Ostrava (FNO),
  patients 2005–2025 (one export, October 2025).
- Inclusion: age ≥ 18, endovascular treatment performed (diagnostic angiography "PAG" excluded),
  90-day mRS recorded. Flow: 1,577 registry rows → 1,565 adults → 1,475 treated → 1,361 analysed.
- Stage cohorts: stage 2 requires a final mTICI; stage 3 an assessed 24-h NIHSS; stage 4 an
  assessed day-7 NIHSS. Patients who died earlier leave the later cohorts.
- Outcome assessment: 90-day mRS as recorded in the registry (assessor and method not recorded).

## 3. Predictors

See `config/features.yaml` (definitions, units, allowed ranges, monotone direction, stage).
Exclusions made on purpose:

- **NIHSS codes 43 (= died) and 99 (= not assessable)** are never used as scores.
- **Cholesterol** (measured later in the stay; its missingness tracks early death) and
  **etiology** (known only after work-up) are not used.
- **Year of treatment** is not a predictor (it cannot be extrapolated); time trends are handled by
  recalibration.

## 4. Missing data

LightGBM handles missing values natively during training. In the calculator, an input marked
"unknown" is **averaged over registry values** (20 quantiles, or the category mix, among patients
treated since 2016) rather than sent down the trees' missing-value branch, because missingness in
the registry is strongly tied to era (ASPECTS absent before 2016, anaesthesia before about 2012)
and to severity (SBP or onset time missing in sicker patients).

## 5. Model development

- LightGBM, binary log-loss, one model per stage × outcome (8 models).
- Monotone constraints for clinically ordered predictors (`features.yaml`, flipped for mortality).
- Hyperparameters tuned with Optuna (TPE, 50 trials) on 5-fold cross-validated log-loss within the
  development cohort only (onset ≤ 2020), early stopping on the number of trees.
- Minimum leaf size: each leaf's hessian sum ≥ 5, which guarantees ≥ 20 patients per leaf
  (each patient contributes at most 0.25 for log-loss). This is both regularisation and a
  disclosure-control measure.
- **Comparison learners, tuned with the same budget:** XGBoost, a multilayer perceptron, and
  penalised logistic regression, linear and with restricted cubic splines. Every learner,
  LightGBM included, gets 50 candidate settings, each scored by 5-fold stratified cross-validated
  log-loss on the development cohort with the same folds (`src/mtop/tuning.py`). Search spaces
  and the chosen settings are in section 5a.
- **Benchmarks without settings:** THRIVE, SPAN-100, HIAT2, PRE, the FNO 7-item bedside score and
  NIHSS at 24 h or day 7 alone (points mapped to probabilities by one logistic fit on the
  development cohort); the THRIVE-c and MR PREDICTS@24H predictor sets refitted with the
  logistic-regression pipeline.

## 5a. Comparison models and their settings

Every learner had the same budget: 50 candidate settings, each scored by 5-fold stratified cross-validated log-loss on the development patients (onset ≤ 2020), with the same folds for every learner (seed 20261016). The 2021–2025 patients were not used for any choice. Published point scores have no settings; their points are mapped to probabilities by one logistic fit on the development patients.

| Learner | Searched |
|---|---|
| LightGBM | Optuna TPE: learning rate 0.01–0.08 (log), leaves 3–15, depth 2–5, patients per leaf 20–80, column fraction 0.5–1, row fraction 0.6–1, L2 0.001–30 (log); trees by early stopping (100 rounds, max 3000); monotone constraints; every leaf ≥ 20 patients |
| XGBoost | Optuna TPE: learning rate 0.01–0.08 (log), depth 2–5, minimum child weight 1–20 (log), row fraction 0.6–1, column fraction 0.5–1, L2 0.001–30 (log), minimum split gain (gamma) 0.001–5 (log); trees by early stopping (100 rounds, max 3000) |
| MLP | Optuna TPE: 1–3 hidden layers, first layer 8–128 neurons (log; each further layer half as wide), ReLU or tanh, weight decay 1e-5–10 (log), Adam learning rate 1e-4–1e-2 (log), batch 32/64/128/256; early stopping on 20 % of the training fold (patience 25, max 1000 epochs); median imputation with missing indicators and standardisation; the best setting trained from 5 random starts and averaged |
| Logistic regression | L2 penalty strength C over 50 values (1e-4–1e2, log grid); median imputation with missing indicators and standardisation |
| Logistic regression (splines) | as logistic regression, plus restricted cubic splines (3, 4 or 5 knots at Harrell's percentiles) for age, NIHSS, onset to puncture, glucose, procedure duration and later NIHSS; 50 values of C for each number of knots |

**Chosen settings** (development cohort, onset ≤ 2020)

*LightGBM*

| Stage | Outcome | Learning rate | Leaves | Depth | Min leaf | Columns | Rows | L2 | Trees | CV log-loss |
|---|---|---|---|---|---|---|---|---|---|---|
| admission | mRS 0–2 | 0.0616 | 5 | 3 | 25 | 0.505 | 0.639 | 0.0124 | 73 | 0.581 |
| admission | death | 0.0379 | 9 | 2 | 25 | 0.517 | 0.653 | 0.707 | 176 | 0.448 |
| procedure | mRS 0–2 | 0.0144 | 12 | 5 | 27 | 0.513 | 0.747 | 0.0762 | 315 | 0.548 |
| procedure | death | 0.0292 | 7 | 5 | 45 | 0.922 | 0.631 | 0.0383 | 137 | 0.440 |
| h24 | mRS 0–2 | 0.0309 | 6 | 2 | 32 | 0.526 | 0.767 | 0.162 | 177 | 0.377 |
| h24 | death | 0.0598 | 8 | 3 | 40 | 0.631 | 0.985 | 2.96 | 105 | 0.298 |
| d7 | mRS 0–2 | 0.0158 | 13 | 3 | 74 | 0.751 | 0.986 | 0.00861 | 375 | 0.331 |
| d7 | death | 0.0218 | 6 | 2 | 34 | 0.58 | 0.844 | 0.139 | 311 | 0.246 |

*XGBoost*

| Stage | Outcome | Learning rate | Depth | Min child weight | Rows | Columns | L2 | Gamma | Trees | CV log-loss |
|---|---|---|---|---|---|---|---|---|---|---|
| admission | mRS 0–2 | 0.023 | 2 | 2.86 | 0.623 | 0.54 | 0.042 | 0.2 | 288 | 0.584 |
| admission | death | 0.0589 | 4 | 6.61 | 0.681 | 0.642 | 1.58 | 0.0759 | 62 | 0.448 |
| procedure | mRS 0–2 | 0.0221 | 2 | 5.7 | 0.621 | 0.555 | 0.00737 | 0.028 | 334 | 0.552 |
| procedure | death | 0.0164 | 5 | 5.33 | 0.627 | 0.593 | 0.024 | 0.875 | 248 | 0.442 |
| h24 | mRS 0–2 | 0.0125 | 2 | 1.02 | 0.882 | 0.539 | 0.00163 | 0.0779 | 463 | 0.374 |
| h24 | death | 0.0198 | 2 | 1.04 | 0.998 | 0.5 | 0.0588 | 0.0257 | 411 | 0.290 |
| d7 | mRS 0–2 | 0.0441 | 2 | 3.42 | 0.871 | 0.786 | 0.307 | 0.00605 | 136 | 0.331 |
| d7 | death | 0.0232 | 3 | 1.11 | 0.764 | 0.594 | 0.0244 | 0.484 | 195 | 0.242 |

*MLP*

| Stage | Outcome | Hidden layers | Activation | Weight decay | Learning rate | Batch | CV log-loss |
|---|---|---|---|---|---|---|---|
| admission | mRS 0–2 | 30 | tanh | 0.000636 | 0.00276 | 256 | 0.581 |
| admission | death | 100 | relu | 1.44 | 0.00589 | 32 | 0.465 |
| procedure | mRS 0–2 | 46 | tanh | 1.15 | 0.00275 | 32 | 0.547 |
| procedure | death | 113→56 | tanh | 0.00488 | 0.00046 | 128 | 0.460 |
| h24 | mRS 0–2 | 74→37 | tanh | 0.577 | 0.0049 | 32 | 0.389 |
| h24 | death | 72 | relu | 0.587 | 0.00408 | 32 | 0.317 |
| d7 | mRS 0–2 | 119 | relu | 4.94 | 0.006 | 256 | 0.384 |
| d7 | death | 72 | relu | 0.455 | 0.00311 | 32 | 0.262 |

*Logistic regression*

| Stage | Outcome | C (linear) | CV log-loss | Knots (splines) | C (splines) | CV log-loss |
|---|---|---|---|---|---|---|
| admission | mRS 0–2 | 0.0281 | 0.581 | 3 | 0.016 | 0.579 |
| admission | death | 0.0281 | 0.457 | 3 | 0.016 | 0.456 |
| procedure | mRS 0–2 | 0.0281 | 0.550 | 3 | 0.0212 | 0.549 |
| procedure | death | 0.016 | 0.453 | 3 | 0.0121 | 0.451 |
| h24 | mRS 0–2 | 0.0869 | 0.383 | 3 | 0.0869 | 0.385 |
| h24 | death | 0.0494 | 0.313 | 3 | 0.0281 | 0.310 |
| d7 | mRS 0–2 | 0.153 | 0.356 | 3 | 0.0869 | 0.360 |
| d7 | death | 0.0655 | 0.250 | 3 | 0.0373 | 0.244 |

C is the inverse penalty strength: smaller C means stronger shrinkage.

Regenerated by `python -m mtop report` as `reports/model_settings.md`.

## 6. Validation

- **Temporal validation** (primary): models fitted on onset ≤ 2020 and applied unchanged to
  2021–2025. AUROC and Brier with percentile bootstrap 95 % CIs (2,000 resamples), paired
  bootstrap for ΔAUROC, calibration intercept (calibration-in-the-large) and slope, integrated
  calibration index, calibration by deciles, decision curves, subgroup AUROC (sex, age,
  circulation, IVT, admission route).
- **Model comparison:** LightGBM against XGBoost, an MLP and logistic regression (linear and with
  splines), every learner tuned with the same budget on the same folds (section 5a), and against
  published scores.
- **Internal validation:** 5-fold × 3 repeated cross-validation in the development cohort.
- **Updating check:** the intercept re-estimated on 2021–2022 patients and evaluated on 2023+.

## 7. Deployed models

- Refitted on all years with the tuned hyperparameters (number of trees re-chosen by CV).
- Intercept shifted to the 2021–2025 outcome level using out-of-fold predictions (temporal
  recalibration; Booth et al., Int J Epidemiol 2020; Steyerberg, *Clinical Prediction Models*,
  ch. 20).
- Browser implementation verified against LightGBM (predictions and TreeSHAP to 1e-6).

## 7a. Additions in v1.1 (beta, same deployed models)

- **Comparators for the calculator:** penalised logistic regression on the MTOP predictors, linear
  and with restricted cubic splines, tuned as in section 5a but on all years;
  THRIVE, SPAN-100, HIAT2 and PRE mapped to probabilities by logistic recalibration on the
  registry; THRIVE-c and MR PREDICTS@24H predictor sets refitted on the registry (published
  coefficients not used; MR PREDICTS@24H without the CTA collateral score, not recorded). All are
  shifted to the 2021–2025 outcome level. Browser arithmetic verified against Python
  (`tests/test_extras_parity.py`).
- **Full 90-day mRS distribution:** a cumulative-logit layer on the two LightGBM scores (mRS 0–2
  and death), fitted on out-of-fold scores with thresholds re-estimated on 2021–2025. It only
  splits the independent and dependent bands into grades, so headline probabilities are those of
  the binary models. Temporal validation (development models applied to 2021–2025): ordinal
  c-index and AUROC at every mRS cut point, reported on the Validation tab.
- **Uncertainty:** 100 bootstrap refits of each deployed model (Riley & Collins, BMJ 2023);
  the calculator shows the median 95 % range for the estimate's risk bin; mean absolute
  prediction error is reported per stage.
- **Adaptive updating experiment:** rolling-origin evaluation 2018–2025 of six update strategies
  with fixed generic hyperparameters, a drift monitor for a model frozen in 2015, and
  leave-one-era-out internal–external validation pooled by random-effects meta-analysis.
- **In-browser validation for partner centres:** the page computes AUROC (bootstrap CI),
  Brier score, calibration intercept and slope and grouped calibration on a partner's file
  without uploading it, and exports aggregate results for pooling.

## 8. Limitations

- Single centre, retrospective use of a registry; external validation is needed.
- Since 2021, 24-hour NIHSS is missing in about half of patients and day-7 NIHSS often, so the
  stage 3 and 4 temporal validation rests on a selected subset; the day-7 mortality model rests on
  few late deaths.
- Imaging (core volume, collaterals), number of passes and door-to-needle data are not in the
  registry.
- Observational associations (for example anaesthesia type) are not causal effects.
- The adaptive experiment uses generic hyperparameters for every strategy, so its absolute
  performance differs slightly from the tuned deployed models; it is a comparison of strategies.
- Previews of candidate predictors in the calculator are illustrations and do not affect any
  estimate.

## 9. Privacy

Published artifacts contain no patient-level data: tree leaves summarise ≥ 20 patients; aggregate
tables hide groups under 10 (full mRS distributions need ≥ 30); no dates, IDs, hospital names
or value extremes are exported. Enforced by `tests/test_privacy.py` and by the CI data guard.

## 10. Maintenance

Re-run `python -m mtop all` when new registry data are available; review
`reports/data_quality.xlsx`; check calibration drift on the newest year before republishing.
