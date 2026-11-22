# MTOP ordinal model: development and evaluation plan

Plan, 22 November 2026. Committed to the repository before any ordinal-model code is written or
any ordinal model is fitted; the commit's time on GitHub is its time stamp. Any later change is
added at the end as a dated, numbered amendment with its reason, committed before the work it
changes.

## 1. Purpose and what stays fixed

MTOP (Mechanical Thrombectomy Outcome Predictor) estimates two binary outcomes at 90 days:
functional independence (modified Rankin Scale, mRS, 0–2) and death (mRS 6). The calculator splits
these into the seven mRS grades with a proportional-odds layer fitted on the two binary models'
scores. This plan adds a **full ordinal model**: a proportional-odds (cumulative logit) regression
trained directly on all seven grades with all inputs, as MR PREDICTS is.

Why:

- It uses all of the outcome information, as the ordinal ("shift") analyses of stroke trials do.
- It is the like-for-like comparator of MR PREDICTS, which predicts the whole mRS distribution, in
  the planned external validation on MR CLEAN Registry data.
- It may give a better distribution over the seven grades than the layer.

What stays fixed:

- **The binary models are not changed.** MTOP v1.0.0 (fingerprint 77bb37a3…) stays the
  pre-specified primary model of every external validation; v1.1 (841d810c…) stays the calculator's
  default. Their headline estimates of independence and death stay those of the binary models,
  whatever this plan decides.
- **The ordinal model is a new model, released as its own bundle** ("MTOP ordinal", version 1.0.0,
  with its own fingerprint and Zenodo version under the MTOP all-versions DOI). It is not part of
  version 1.1 and does not change version 1.2 (the planned data update).
- **Same data, patients and splits as v1.0.0 and v1.1** (section 3).
- **One machine-learning model is evaluated for comparison only** (section 9). It cannot replace the
  regression, is not deployed and is not sent to external validation.
- **Nothing in this plan is implemented yet.**

## 2. What is known when this plan is written

The ordinal model is not blind to earlier results, and the evaluation (sections 6–8) is set up so
that this cannot flatter it.

- The temporal test of v1.0.0 and v1.1 (development 2005–2022, test 2023–2025), including the
  over-prediction of independence in 2023–2025 and the shift towards mRS 4–5.
- The proportional-odds layer's test results: generalised concordance 0.72 before thrombectomy and
  0.83 at 24 hours; mean predicted utility-weighted mRS 0.44 against 0.38 observed before
  thrombectomy.
- The nested cross-validation and bootstrap results of v1.1, and the MRS CLEAN results of v1.0.0
  and v1.1.
- The marginal distribution of the 90-day mRS by treatment era (published in the manuscript's
  tables).
- No ordinal model with patient inputs has been fitted on these data, and no ordinal-model result of
  any kind has been looked at.

## 3. Data, cohorts and outcome

- **Registry export:** the frozen export of v1.1, SHA-256
  `1a19f809ed688a3940c38ed4d0b1198f0a978b806f9ed7b50721777d980d50f7`. Same cleaning, same data
  checks.
- **Patients:** adults with a recorded 90-day mRS. **Stage cohorts** as in v1.1: before
  thrombectomy (all analysed patients); end of procedure (final mTICI grade recorded); 24 hours and
  day 7 (assessable NIHSS at that time).
- **Splits:** primary development 2005–2022, test 2023–2025; sensitivity development 2005–2020,
  test 2021–2025. Cross-validation folds and seeds of v1.0.0 (seed 20261016).
- **Outcome:** the 90-day mRS as recorded, seven ordered grades 0–6. **Sparse grades:** at a stage,
  a grade with fewer than 20 development patients is merged with a neighbour in the same band
  (independent, mRS 0–2; dependent, mRS 3–5), so the band boundaries stay intact: mRS 0 with 1, 2
  with 1, 3 with 4, 5 with 4, and mRS 1 or 4 with the smaller of its two neighbours. Death (mRS 6) is
  never merged. Any merge is reported.

## 4. Model specification

- **Model:** cumulative logit, logit P(mRS ≤ k | x) = θ_k − x′β for k = 0…5; one coefficient vector
  shared by all cut-points (proportional odds); thresholds θ ordered by construction.
- **Penalty and solver:** L2 penalty on β only (C × sum of the negative log-likelihood + ½‖β‖²; the
  thresholds are not penalised), all columns standardised on the training data, fitted by
  Newton–Raphson to convergence, as the v1.1 reference.
- **Inputs:** the v1.1 reference specification at each stage, unchanged: restricted cubic splines
  for age, NIHSS (admission, 24 hours, day 7), onset-to-puncture time, glucose and procedure
  duration; one missing-value indicator per input, only for inputs that at least 20 development
  patients lack; a category's own term only with at least 20 patients; no inputs that are exact
  functions of others and no clock hour of onset; each stage re-enters all earlier inputs (the v1.1
  full structure).
- **Tuning:** number of knots (3, 4 or 5) and penalty (the 50 values of C from 10⁻⁴ to 10², log
  spaced) chosen by mean ordinal log-loss over 5-fold cross-validation repeated 5 times, folds
  stratified by mRS grade.
- **Explanations:** exact additive explanations on the linear predictor (coefficient × centred
  value), which apply to every cut-point because the coefficients are shared.

## 5. Order of work

1. Commit this plan.
2. **Code and unit tests**, before any registry data are used: the penalised cumulative-logit
   solver must (a) reproduce statsmodels' unpenalised OrderedModel (logit link) on simulated data to
   10⁻⁶, (b) reduce exactly to the v1.1 binary Newton solver when there are two categories, (c) pass a
   numerical gradient and Hessian check, and (d) give the same predictions when rows are reordered.
3. **Pipeline check on the registry:** data checks, cohorts and model columns identical to v1.1
   (same export SHA-256); a small unpenalised ordinal model matches statsmodels to 10⁻⁶.
4. **Development-year evaluation** (section 6, and the comparison-only learner of section 9) and the
   **decisions** of section 7. The decisions are written to the results file and committed **before
   any 2023–2025 or 2021–2025 result of the ordinal model or the comparison-only learner is
   computed**.
5. **Temporal and sensitivity tests** (section 8), reported, not deciding.
6. **Deployment, freezing and release** (section 11).
7. **External validation registration** (section 12).

## 6. Evaluation within the development years (2005–2022), which decides

- **Nested cross-validation:** 5 outer folds × 3 repeats (the v1.0.0 folds), the whole pipeline
  (layout, knots, penalty, thresholds) refitted inside each outer fold, at all four stages.
- **Full-pipeline bootstrap:** 500 resamples (seed 2026), optimism by Harrell's method, at all four
  stages; inner folds keep copies of a patient together.
- **Primary stages:** before thrombectomy and 24 hours. The others are secondary.

Measures (all pre-specified):

- **Overall accuracy:** ranked probability score (RPS, the mean over patients of
  (1/6) Σₖ₌₀…₅ (F̂ₖ − 1[mRS ≤ k])², where F̂ₖ is the predicted P(mRS ≤ k); lower is better), the
  primary measure; and ordinal log-loss (mean −log of the probability given to the observed grade).
- **Discrimination:** generalised concordance, the probability that of two patients with different
  mRS the one with the better predicted expected utility-weighted mRS had the better outcome (ties
  in the prediction count ½; utility weights 1.00, 0.91, 0.76, 0.65, 0.33, 0, 0 for mRS 0–6, as in
  the calculator); the same with the expected mRS grade as the ranking score; and the area under the
  ROC curve (AUROC) for each dichotomy implied by the model: mRS 0–1, 0–2, 0–3 and death.
- **Calibration:** for each implied dichotomy, calibration-in-the-large and slope; for the whole
  distribution, mean predicted against observed proportion in each grade, the ordinal calibration
  slope (the coefficient of the linear predictor when a proportional-odds model with free thresholds
  is refitted on it) and the mean predicted against observed utility-weighted mRS.
- **Stability:** across the 500 bootstrap models, the median width of each patient's 95 % range of
  P(mRS 0–2), P(death) and expected utility-weighted mRS.
- **Consistency with the binary models:** AUROC, log-loss and calibration of the implied P(mRS 0–2)
  and P(death) against the v1.1 binary models, on the same folds.
- **Comparison with the layer:** RPS, ordinal log-loss and generalised concordance against the v1.1
  proportional-odds layer, refitted inside each outer fold on inner out-of-fold scores.
- **Proportional-odds assumption** (descriptive, section 7, rule O4).
- **Sample size:** the minimum sample size of Riley et al., applied at each cut-point of the
  outcome, reported for each stage (descriptive, not deciding).

## 7. Decision rules (development years only)

- **O1. The calculator's distribution.** The binary models keep the headline estimates. The
  ordinal model replaces the layer as the source of the shares that split the independent (mRS 0–2)
  and dependent (mRS 3–5) bands into single grades if, in nested cross-validation, the resulting
  distribution has a lower mean RPS than with the layer's shares at both primary stages. Otherwise
  the calculator keeps the layer.
- **O2. External validation.** The ordinal model is frozen and goes to external validation as a
  secondary model whatever O1 decides, unless the pipeline check fails or, at a primary stage, the
  nested cross-validated AUROC of its implied P(mRS 0–2) is more than 0.02 below that of the v1.1
  binary model. In that case it is reported and not sent.
- **O3. Slope correction.** If the optimism-corrected ordinal calibration slope at a stage lies
  outside 0.9–1.1, the linear predictor is multiplied by it and the thresholds are re-estimated
  (as v1.1 rule b). Otherwise nothing is applied.
- **O4. Proportional odds.** Not a decision rule. The assumption is examined by fitting a separate
  penalised binary model at each cut-point with the same inputs and penalty and comparing their
  coefficients with the shared ones (a Brant-type comparison), and by the calibration of each implied
  dichotomy. A partial proportional-odds model is not part of this version.

## 8. Temporal and sensitivity tests (reported, not deciding)

- The development-year model is tested on 2023–2025 and, in the sensitivity split, on 2021–2025,
  with the measures of section 6 and 2,000 paired bootstrap resamples of the test patients for
  intervals and differences.
- **Labelled as not independent:** the specification is that of v1.1, which was designed with the
  2023–2025 results known.
- **Comparators, on the same patients:** the v1.1 binary models (implied dichotomies); the v1.1
  layer; the MR PREDICTS predictor set refitted as a proportional-odds model on the development data
  without the collateral score (before thrombectomy and end of procedure); the MR PREDICTS@24H
  predictor set refitted likewise (24 hours and day 7); and the current NIHSS alone as a
  proportional-odds model with a 3-knot spline (24 hours and day 7). Differences are reported with
  unadjusted intervals; no comparator can replace anything.
- Subgroups by sex and age (under 80, 80 or older), where a subgroup has at least 10 patients in
  each of three grade groups (mRS 0–2, 3–5, 6); otherwise not shown.

## 9. Comparison-only learner: ordinal gradient boosting

In the sensitivity split of v1.0.0, gradient-boosted trees did better than the regression after the
procedure. To learn whether this holds for the ordinal outcome, one ordinal gradient-boosting model
is evaluated alongside the regression, for comparison only.

- **Stages:** end of procedure, 24 hours and day 7.
- **Model:** six LightGBM models, one for each cut-point P(mRS ≤ k), k = 0…5, on the stage's inputs
  as the v1.0.0 LightGBM receives them (missing values handled by the trees), with its monotone
  constraints (signs reversed where needed so that each constraint points towards a better outcome)
  and at least 20 patients per leaf. Each patient's cumulative probabilities are made monotone by
  taking their running maximum over k (capped at 1); the probability of each grade is the difference
  of consecutive cumulative probabilities.
- **Settings:** the hyperparameters tuned for the v1.0.0 LightGBM for mRS 0–2 at that stage on the
  development years of the respective split, reused for all six cut-points with no further tuning.
  Because these were tuned on all development years, the nested cross-validation slightly favours
  this learner; this is reported with its results.
- **Evaluation:** the same folds, measures and test periods as the regression (sections 6 and 8).
- **Role:** comparison only. It cannot replace the regression, is not deployed and is not sent to
  external validation. If it is better than the regression on the ranked probability score in both
  the nested cross-validation and the temporal test, this is noted as a reason to pre-specify it as
  a candidate in a later version.

## 10. What this version does not include

- Ordinal machine-learning models as candidates (one is evaluated for comparison only, section 9).
- Partial proportional-odds models.
- New predictors (GFAP, platelet-to-white-cell ratio), 2026 patients or completed follow-up: these
  belong to later versions under their own plans.
- Any change to the binary models.

## 11. Deployment, freezing and release

- **Deployed model:** refitted on all patients of each stage's cohort with the tuned settings and,
  if rule O3 applies, the slope factor. Its common threshold shift is re-estimated on the most recent
  three complete years (2023–2025) from cross-validated predictions, as the binary models' intercepts
  are. The slope is updated only under the calculator's update policy (slope outside 0.8–1.2 with a
  95 % interval that excludes 1). Death on day 7 carries the "limited evidence" label of v1.1.
- **Bundle:** coefficients, thresholds, knot positions, medians, scaling and layout only; SHA-256
  fingerprint of the bundle; no patient-level data; counts from 1 to 9 shown as fewer than 10.
- **Browser:** the calculator reimplements the model in JavaScript; automated tests require its
  predictions and explanations to match the Python model within 10⁻⁷ on every development patient
  and on the synthetic demonstration cohort.
- **Release:** results file, model card, diagnostics and changelog entry; a GitHub release that
  creates a new version under the MTOP all-versions Zenodo DOI (10.5281/zenodo.24396392). The binary
  versions stay unchanged and selectable.

## 12. External validation and synthetic data

- **Before any MR CLEAN Registry data are received,** the frozen ordinal model's fingerprint is added
  to the external analysis plan (the CONTRAST data request) as a secondary model: v1.0.0 primary,
  v1.1 and the ordinal model secondary. That plan, not this one, sets the comparison with MR PREDICTS
  (published ordinal formula with the collateral score): generalised concordance, RPS and the
  calibration of the distribution, with its margins.
- **MRS CLEAN:** the ordinal model is run on the synthetic data only after it is frozen, only with
  the MR CLEAN team's agreement, through the same tool, as a check of the pipeline rather than as
  evidence. The results are shared with the MR CLEAN team before any other use.
- Other centres validate it through the validation tool once it is released, with results files
  tied to its fingerprint.

## 13. Reporting

- TRIPOD+AI items for model development and evaluation; every result reported, including those
  that do not favour the ordinal model.
- The results file is written by the analysis code from its outputs, not by hand.
- Aggregates only; raw and patient-level data stay outside the repository.

## 14. Amendments

### Amendment 1 (22 November 2026): how the plan is carried out

Written after steps 2 and 3 (code, unit tests, pipeline check: passed; no grade has fewer than 20
development patients at any stage, so no grades are merged) and before any ordinal-model or
comparison-learner result has been computed. It fixes details the plan leaves open; it changes no
decision rule.

1. **Folds.** The outer folds are v1.1's nested folds for mRS 0–2 (5 × 3, seed 20261016,
   stratified by mRS 0–2), used for every model, the death model included, so that all comparisons
   are on identical patients. In outer fold k the inner seed is 20261016 + 100 + k, as in v1.1; the
   ordinal model's inner tuning folds are stratified by mRS grade.
2. **v1.1 binary models in each fold:** the v1.1 full structure tuned on the outer training part,
   with rule (b)'s factor (1.10193, 24 hours, death) applied on that part, as deployed.
3. **The v1.1 layer in each fold:** refitted on the training part's out-of-fold log-odds of both
   binary models (5 folds, seed + 1, settings fixed), as the deployed layer is built; within the
   development years its thresholds are not re-estimated on recent years.
4. **Ordinal log-loss** floors the probability of the observed grade at machine epsilon (2.2 ×
   10⁻¹⁶), as the tuning code does; the number of patients whose observed grade had a probability
   below 10⁻⁶ is reported, because the running maximum of the comparison-only learner can give a
   grade a probability of exactly 0. The ranked probability score is unaffected.
5. **The ordinal calibration slope** is computed for models with a linear predictor (the ordinal
   regression and the proportional-odds comparators). The layer, the bands with ordinal shares and
   the comparison-only learner are calibrated through their dichotomies.
6. **Comparison-only learner:** also in the full-pipeline bootstrap (settings fixed, seed 20261016),
   so its optimism-corrected measures and stability can be reported with the same method.
7. **Bootstrap:** apparent measures on the resample, copies counted; test measures on the original
   stage cohort; stability from the 2.5th and 97.5th percentiles of each patient's 500 estimates.
8. **Proportional odds (rule O4):** at each cut-point k, a penalised binary model of mRS > k with
   the development model's layout and penalty; reported are the slope and correlation of its linear
   predictor on the shared one (both 1 under proportional odds, apart from the penalty) and the
   calibration of each implied dichotomy.
9. **Sample size:** at each cut-point, Riley et al.'s three criteria (shrinkage ≥ 0.9, optimism in
   Nagelkerke R² ≤ 0.05, intercept within ± 0.05) with the development model's number of
   coefficients; the anticipated Cox–Snell R² is 15 % of its maximum at every cut-point (their
   conservative default) and, at mRS 0–2 and death, also the value implied by v1.1's nested
   cross-validated AUROC (Riley et al. 2021).
10. **Temporal comparators:** the MR PREDICTS and MR PREDICTS@24H predictor sets are refitted as
    proportional-odds models with the pipeline of the binary refits (median imputation with
    missing-value indicators, standardisation, L2 penalty chosen by 5-fold cross-validation, here on
    ordinal log-loss); NIHSS alone uses a 3-knot spline (knots at the 10th, 50th and 90th
    percentiles) in the same pipeline. Rule O3's factor, decided in 2005–2022, is used in both
    splits, as v1.1 did. The learner's settings in the sensitivity split are those tuned for v1.0.0
    in 2005–2020.
11. **Deployment:** the threshold shift is the maximum-likelihood common shift of the 2023–2025
    out-of-fold cumulative log-odds (5 folds, seed + 1, stratified by grade, settings fixed); the
    slope check of the update policy uses the ordinal calibration slope of the same estimates with a
    95 % Wald interval.
12. **Computation:** the solver's per-patient arithmetic runs through numba when it is installed;
    a unit test requires it to match the plain NumPy code to 10⁻⁹ (relative).
