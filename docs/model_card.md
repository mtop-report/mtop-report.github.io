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
- Comparators: L2 logistic regression (median imputation + missing indicators, standardised,
  C by CV), XGBoost (grid on depth / min child weight, early stopping), MLP (grid on width /
  weight decay, 5-seed average), THRIVE, SPAN-100, NIHSS at 24 h alone, NIHSS on day 7 alone.

## 6. Validation

- **Temporal validation** (primary): models fitted on onset ≤ 2020 and applied unchanged to
  2021–2025. AUROC and Brier with percentile bootstrap 95 % CIs (2,000 resamples), paired
  bootstrap for ΔAUROC, calibration intercept (calibration-in-the-large) and slope, integrated
  calibration index, calibration by deciles, decision curves, subgroup AUROC (sex, age,
  circulation, IVT, admission route).
- **Internal validation:** 5-fold × 3 repeated cross-validation in the development cohort.
- **Updating check:** the intercept re-estimated on 2021–2022 patients and evaluated on 2023+.

## 7. Deployed models

- Refitted on all years with the tuned hyperparameters (number of trees re-chosen by CV).
- Intercept shifted to the 2021–2025 outcome level using out-of-fold predictions (temporal
  recalibration; Booth et al., Int J Epidemiol 2020; Steyerberg, *Clinical Prediction Models*,
  ch. 20).
- Browser implementation verified against LightGBM (predictions and TreeSHAP to 1e-6).

## 8. Limitations

- Single centre, retrospective use of a registry; external validation is needed.
- Since 2021, 24-hour NIHSS is missing in about half of patients and day-7 NIHSS often, so the
  stage 3 and 4 temporal validation rests on a selected subset; the day-7 mortality model rests on
  few late deaths.
- Imaging (core volume, collaterals), number of passes and door-to-needle data are not in the
  registry.
- Observational associations (for example anaesthesia type) are not causal effects.

## 9. Privacy

Published artifacts contain no patient-level data: tree leaves summarise ≥ 20 patients; aggregate
tables hide groups under 10 (full mRS distributions need ≥ 30); no dates, IDs, hospital names
or value extremes are exported. Enforced by `tests/test_privacy.py` and by the CI data guard.

## 10. Maintenance

Re-run `python -m mtop all` when new registry data are available; review
`reports/data_quality.xlsx`; check calibration drift on the newest year before republishing.
