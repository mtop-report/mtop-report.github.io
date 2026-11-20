# MTOP version 1.1: development and evaluation plan

Plan, 17 November 2026. Committed to the repository before any version 1.1 code is written or any
version 1.1 model is fitted; the commit's time on GitHub is its time stamp. Any later change is
added at the end as a dated amendment with its reason.

## 1. Purpose and what stays fixed

Version 1.1 refines the reference model of MTOP (Mechanical Thrombectomy Outcome Predictor):
penalised logistic regression with restricted cubic splines. It fixes weaknesses found after
version 1.0.0 was frozen, in the temporal test, the model diagnostics, the bootstrap internal
validation and the two dry runs on synthetic MR CLEAN data (MRS CLEAN).

- **MTOP v1.0.0 (fingerprint 77bb37a3…) stays the pre-specified primary model.** It is the model of
  the first manuscript and of the external validations under way, and it stays in the app until
  version 1.1 is released. After that it stays archived and selectable in the app, so every
  published v1.0.0 result can be reproduced.
- **Version 1.1 is a refinement, not a new model search.** It keeps the same outcomes (90-day
  modified Rankin Scale (mRS) 0–2 and death by 90 days), the same four stages (before thrombectomy,
  end of procedure, 24 hours, day 7) and the same predictors. The only exceptions are inputs that
  are exact functions of other inputs (change 2.3).
- **Nothing in this plan is implemented yet.** Implementation starts only after this file is
  committed.

## 2. What was known when this plan was written

The changes below were motivated by results that are already known. Version 1.1 is therefore not
blind to them, and section 6 sets out how it is evaluated so that this cannot flatter it.

- **Temporal test (development 2005–2022, test 2023–2025), v1.0.0 before recalibration.**
  Before thrombectomy, mRS 0–2: AUROC (area under the receiver operating characteristic curve)
  0.809, calibration-in-the-large −0.41, slope 1.47.
  - At 24 hours and day 7, the NIHSS (National Institutes of Health Stroke Scale) at that time alone
    did as well as or better than the full model: day 7, mRS 0–2, 0.924 against 0.884 (DeLong
    p = 0.15). In the sensitivity split (test 2021–2025) it was 0.914 against 0.813 (p < 0.001).
  - No alternative learner met the pre-specified rule at any stage.
- **Model diagnostics (docs/model_diagnostics.md).**
  - Missing-value terms make up about half of every model's coefficients (45 of 91 before
    thrombectomy).
  - 19 inputs are missing in some patients but in fewer than 20, 13 of them in fewer than 10.
  - Every model chose the smallest number of knots (3) except death at day 7.
- **Bootstrap internal validation (docs/registrations/bootstrap_optimism_plan.md), before
  thrombectomy.**
  - Optimism-corrected AUROC 0.761 (mRS 0–2) and 0.790 (death).
  - Optimism-corrected calibration slope 1.09 and 1.10: the penalty shrinks about 10 % more than
    the data need.
  - Each patient's risk varied across the bootstrap models by a median of 21 and 15 percentage
    points.
- **MRS CLEAN dry runs (6 November 2026).** The missing-value term for where intravenous
  thrombolysis was given (+2.18 log-odds of death at 24 hours, learned from 4 of 1,235 patients)
  applied to every thrombolysed synthetic patient. The app's "Validate on your data" has averaged
  unknown inputs since then; the models were not changed.
- **New FNO patients.** A file of 72 patients treated in 2025–2026 was run through v1.0.0 with
  "Validate on your data" on 5 November 2026. Of the 36 treated in 2026, 27 had a 90-day outcome,
  and the tool withheld performance as insufficient. No outcome-based result for 2026 patients has
  been looked at.

## 3. Changes

### 3.1 Data

**1.1 Complete the follow-up.**
- Fill in the 90-day mRS where it can be found in the hospital records. It is missing in 21.5 % of
  adults treated in 2023–2025: 62 of 288 recent patients.
- Fill in the 24-hour and day-7 NIHSS where they are documented.
- Record every completed value in the registry's audit trail (data studio).
- Before any version 1.1 fit, publish the SHA-256 of the frozen registry export in this folder.

**1.2 Add patients treated in 2026, once their 90-day outcome is complete.** This is the
"scheduled recalibration after the 2026 follow-up" already announced on the app's Adaptive tab.
- The 2026 patients are first used as a small temporal check of both versions (section 6, step 5).
- Only after that check are they added to the data the deployed models are fitted on.

### 3.2 Specification of the reference model

**2.1 One missing-value indicator per input, not per model column.**
- Version 1.0.0 adds an indicator to every expanded column. An input that spans several columns
  therefore carries several identical indicators. Before thrombectomy this means 13 exact copies:
  7 for occlusion site (8 columns), 1 each for heart condition and antithrombotics, and 4 for the
  spline terms.
- Under an L2 penalty, k identical columns act as one term penalised k times more weakly. The
  missing-value term for occlusion site is therefore shrunk 8 times less than intended.
- Version 1.1 builds one indicator per input.

**2.2 A missing-value term only when at least 20 development patients lack the input.**
- Below that, a missing value is set to the development median with no indicator.
- Why: version 1.0.0 standardises indicators like any other column. For an indicator held by 4 of
  1,235 patients the standard deviation is 0.057, so a modest penalised coefficient becomes a large
  log-odds term (the +2.18 above).
- Diagnostics count 19 such inputs. The model card announced this rule as the intended version 1.1
  change.

**2.3 Drop inputs that are exact functions of others, so no measurement enters twice.**
- Pulse pressure (systolic − diastolic): systolic and diastolic blood pressure stay.
- Intravenous thrombolysis to puncture (onset to puncture − onset to thrombolysis): both onset
  times stay.
- NIHSS change in 24 hours (24-hour − admission NIHSS): both NIHSS values stay.
- Glucose above 7.8 mmol/L: glucose stays, with its spline.
- Time of onset as a number: onset at night stays. Clock time is circular, so a linear term in the
  hour has no clinical meaning.
- Early neurological improvement (a threshold on the two NIHSS values): both NIHSS values stay, with
  their splines.

These derived inputs came from the 2026 FNO analysis. With them removed and changes 2.1–2.2
applied, the model before thrombectomy has about 60 coefficients instead of 91. The exact count is
reported when implemented.

**2.4 At least 20 development patients for a category's own term** (raised from 10, matching
change 2.2). At present this affects no category that has a term: only the anterior cerebral
artery (ACA) is rarer, and it already has none.

**2.5 Steadier tuning, and a pre-specified slope correction.**
- Knots (3, 4 or 5) and the penalty C (the same 50 values, 10⁻⁴ to 10²) are chosen by 5-fold
  cross-validation repeated 5 times (mean log-loss), not by one 5-fold split. This makes the choice
  depend less on one assignment of patients to folds.
- After tuning, the full-pipeline bootstrap (section 6) estimates the optimism-corrected
  calibration slope.
- If that slope is outside 0.9–1.1, the linear predictor is multiplied by it and the intercept is
  re-estimated: a uniform shrinkage or expansion factor (van Houwelingen and le Cessie, 1990).
  Otherwise nothing is applied.

### 3.3 The later stages

**3.1 A candidate "update" model at 24 hours and day 7.**
- Version 1.0.0 re-enters every earlier input at each later stage: 127 coefficients at 24 hours,
  135 at day 7. The penalty then shrinks the dominant late predictor, the current NIHSS, together
  with everything else.
- The candidate is: logit = α + β·LP + f(NIHSS now) + the stage's events, where:
  - LP is the cross-fitted linear predictor of the previous stage's model (end of procedure for
    24 hours; 24 hours for day 7);
  - f is a restricted cubic spline with 3 knots;
  - the events are symptomatic and other intracranial haemorrhage at 24 hours, and oedema and
    decompressive craniectomy on day 7.
- The candidate has about 7 coefficients, tuned in the same way (change 2.5). It is compared with
  the version 1.1 full structure and with the NIHSS alone.
- It replaces the full structure at a stage only under the rule in section 6, step 3c.

### 3.4 Deployment and reporting

**4.1 Refit and recalibration.**
- The deployed models are refitted on all patients with a complete outcome, including 2026 after
  its check.
- Their intercepts are re-estimated on the most recent three complete years, from cross-validated
  predictions, as in version 1.0.0.
- The calibration slope is updated only under the app's update policy: slope outside 0.8–1.2 in
  the recent years, with a 95 % interval that excludes 1.

**4.2 Death on day 7 labelled as limited evidence.**
- The temporal test had fewer than 10 deaths at this stage.
- The calculator shows the death estimate on day 7 with that label until a test with at least
  10 deaths supports it.
- This changes the display, not the model.

## 4. What does not change

- Outcomes, stages, eligibility, the temporal split and the predictor set, apart from change 2.3.
- The candidate learners and the selection rule of amendment 1. They are refitted on the version
  1.1 data and reported for the record. Version 1.1 changes the reference specification, and the
  rule decides as before whether any alternative replaces it.
- Missing-data handling in the calculator: unknown optional inputs are averaged over 20 registry
  values. Missing-data handling in "Validate on your data": any unknown input is averaged.
- Privacy: tree leaves of at least 20 patients, aggregates only, counts under 10 hidden.

**Not in version 1.1:**
- New predictors: serum glial fibrillary acidic protein (GFAP), CT perfusion volumes and the CT
  angiography collateral score. They are tested under amendment 1's rule (100 + 100 outcomes) and
  would start version 2.0.
- New learner families and changes to the outcome definitions.

## 5. Order of work

1. Data freeze and its SHA-256 (change 1.1).
2. **Pipeline check.** The version 1.1 code with every change switched off must reproduce v1.0.0
   exactly, with the same fingerprint, before any change is evaluated.
3. Changes 2.1–2.4, then 2.5, then 3.1, each evaluated as in section 6, in this order.
4. Deployment (4.1–4.2), model card, diagnostics and changelog.

## 6. Evaluation

1. **Development years only (2005–2022) decide.** Each change is compared with the specification
   before it on:
   - nested cross-validation (5 outer folds × 3 repeats; the same folds as v1.0.0) at every stage
     and outcome;
   - the full-pipeline bootstrap of the bootstrap plan (500 resamples, seed 2026), now at all four
     stages and both outcomes.

   Reported: optimism-corrected AUROC, Brier score, log-loss, calibration-in-the-large and slope,
   the median width of patients' risk ranges, and the number of coefficients.
2. **Primary comparisons:** mRS 0–2 before thrombectomy and at 24 hours. The other stages and death
   are secondary.
3. **Rules:**
   - (a) A simplification (2.1–2.4) is kept unless, at a primary comparison, optimism-corrected
     AUROC falls by more than 0.01 or optimism-corrected log-loss rises by more than 0.005.
   - (b) The slope correction (2.5) applies by its own rule.
   - (c) The update model (3.1) replaces the full structure at a stage only if its nested
     cross-validated AUROC is higher by at least 0.01 and its log-loss is not worse.
4. **The temporal test 2023–2025 is reported for version 1.1 next to v1.0.0,** labelled as not
   independent: version 1.1 was designed after these results were known. It decides nothing.
5. **The 2026 check.** Before the 2026 patients join the deployed fit, both versions are scored on
   them (observed/expected ratio, AUROC with 95 % interval). This is descriptive: about 70 patients
   cannot decide between versions.
6. **External validation.**
   - Both bundles (v1.0.0 and v1.1) go to the external partners under one analysis plan.
   - v1.0.0 is the pre-specified primary model; v1.1 is secondary, compared on the same patients.
   - MRS CLEAN is re-run for version 1.1 under the rules of plan version 2, as a check of the
     pipeline, not as evidence. It would be the third look at those synthetic data.

## 7. Release

- Version 1.1.0 gets its own fingerprint, changelog entry, model card and diagnostics. The app
  offers it by default and keeps v1.0.0 selectable.
- It is released after steps 1–5 of section 6. External results are added when they arrive.
- Housekeeping done with the release (no effect on the models):
  - the app's own QR code points to mtop-report.github.io;
  - a favicon.ico;
  - the Zenodo archive of both versions, when the authors decide.

## 8. Amendments

### Amendment 1 (20 November 2026, before any version 1.1 code)

Committed before any version 1.1 code is written or any version 1.1 model is fitted.

**What changes: the plan is split into two releases.**

- **Version 1.1: specification only, on the data of v1.0.0.**
  - It contains changes 2.1–2.5 (reference model), 3.1 (update model at 24 hours and day 7) and
    4.2 (day-7 death label).
  - It is developed and evaluated on exactly the data of v1.0.0: the same registry export, cohorts,
    temporal split and cross-validation folds. No data are added or completed.
  - Section 6, steps 1–4, apply as written. The deployed models are refitted on the same patients
    as v1.0.0, with intercepts re-estimated as in change 4.1 on the most recent three complete years
    (2023–2025).
  - It is released after section 6, steps 1–4.
- **Version 1.2: data update, with the frozen version 1.1 pipeline.**
  - It contains changes 1.1 (complete the follow-up) and 1.2 (add the patients treated in 2026),
    and the refit and recalibration of change 4.1 on these data.
  - The frozen version 1.1 pipeline, including its tuning procedure, is run unchanged on the
    updated data. No specification change is allowed in version 1.2.
  - Data: all patients treated up to 31 December 2026. The data are frozen once the 90-day follow-up
    of the last 2026 patients is complete (expected April 2027), and the SHA-256 of the frozen
    export is published in this folder before any fit.
  - The 2026 check (section 6, step 5) moves to version 1.2: before the 2026 patients join the fit,
    versions 1.0.0 and 1.1 are scored on them. It stays descriptive.
- **External validation (section 6, step 6).**
  - Every version is frozen before any external result for it is seen.
  - Version 1.1 joins an external analysis plan, as the secondary model next to the primary v1.0.0,
    only if it is frozen before that analysis starts.
  - Version 1.2 is validated on data that did not influence it.
- **Release.** Each version gets its own fingerprint, changelog entry, model card and diagnostics,
  and a GitHub release that creates a new version under the all-versions Zenodo DOI
  (10.5281/zenodo.24396392). Earlier versions stay archived and selectable in the app.

**Why.**

- Comparing v1.0.0 with version 1.1 on identical data shows the effect of the specification changes
  alone; when data and methods change together, their effects cannot be separated.
- Version 1.1 no longer waits for the follow-up to be completed or for the 2026 outcomes, so it can
  be frozen before external results arrive and be validated next to v1.0.0.
- Version 1.2 is the scheduled recalibration that the app's update policy calls for.

**What was known when this was written.** Nothing beyond section 2: no version 1.1 code exists, no
new model has been fitted, no MR CLEAN Registry or trial data have been received, and no
outcome-based result for patients treated in 2026 has been looked at.
