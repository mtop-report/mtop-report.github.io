# Internal validation of the MTOP reference model: bootstrap optimism and stability

Analysis plan, 12 November 2026. Committed to the repository before the analysis is run; the
commit's time on GitHub is its time stamp. Any later change is added below as a dated amendment
with its reason.

## Purpose

A supplementary internal validation of MTOP v1.0.0 (model fingerprint 77bb37a3…). It answers two
questions the temporal validation and the nested cross-validation do not:

1. **Optimism.** How much better does the reference model look on the patients it was developed on
   than on new patients from the same population, once every modelling decision is repeated?
   In particular, does the tuned penalty shrink the estimates too much or too little
   (optimism-corrected calibration slope above or below 1)?
2. **Stability.** How stable are the tuned settings (number of spline knots, penalty C) and each
   patient's estimated risk when the development data are resampled?

**Nothing in this analysis changes MTOP v1.0.0**, its fingerprint, the app or the external
validations under way. The results are reported in the model card and inform version 1.1.

## Model and data

- **Model:** the reference learner, logistic regression with restricted cubic splines and an L2
  penalty, at the Admission stage, for 90-day mRS 0–2 (primary) and death (secondary).
- **Data:** the development years of the temporal split, patients treated 2005–2022
  (n = 1,225; 493 with mRS 0–2, 269 deaths). The temporal test years 2023–2025 are not used.

## Procedure (Harrell's bootstrap optimism)

1. **Check of the pipeline first.** Refit on the development data exactly as published (same
   seed and folds); the chosen knots and C must equal the saved settings and the test-year
   predictions must equal the stored ones (largest difference below 10⁻⁶). Otherwise stop.
2. **Apparent performance:** that model evaluated on the development data.
3. **500 bootstrap samples** of patients, drawn with replacement (n = 1,225 each; seed 2026,
   all samples drawn before any fit). In each sample the **whole pipeline is repeated**:
   knot placement at Harrell's percentiles of the sample (3, 4 and 5 knots), median imputation with
   missing-value indicators and standardisation fitted on the sample, and the penalty tuned over the
   same 50 values (10⁻⁴ to 10²) by 5-fold cross-validated log-loss, choosing the knots–penalty pair
   with the lowest log-loss.
   - The inner folds keep all copies of a patient in the same fold (stratified group folds, project
     seed). With ordinary folds, copies of one patient would sit on both sides of a fold and favour
     weak penalties. On data without copies this differs from the published folds only in how
     patients are assigned to folds; its choice on the original data is reported alongside.
4. Each bootstrap model is evaluated on its own sample (apparent) and on the original development
   data (test). **Optimism = apparent − test** for each measure; the optimism-corrected value is the
   original apparent value minus the mean optimism.
5. **Measures:** AUROC, Brier score, log-loss, calibration-in-the-large, calibration slope and
   observed/expected ratio. Reported with the mean optimism, its Monte Carlo standard error and the
   2.5th–97.5th percentiles of the 500 optimism values.
6. **Stability:**
   - the number of knots chosen (share of 3, 4 and 5);
   - the chosen C: median, interquartile range and share at either end of the grid;
   - each development patient's estimated risk across the 500 bootstrap models: the width of its
     2.5th–97.5th percentile range, summarised by the median and 90th percentile over patients, and
     the share of patients whose range is wider than 10 percentage points;
   - convergence warnings.

## Reading the results (descriptive; no decision for v1.0.0)

- An optimism-corrected calibration slope **above 1** means the tuned penalty shrinks the estimates
  more than the development data need (too timid); **below 1**, too little (too extreme).
- Optimism in AUROC is read against the temporal test (0.809 for mRS 0–2 at Admission).
- Settings and risks that vary widely between bootstrap samples argue for a simpler model in
  version 1.1; stable ones support the current specification.

## Reporting and data handling

- Aggregates only. Per-resample results and predictions stay in the git-ignored `reports/` and
  `artifacts/private/` folders; the summary goes into the model card.
- Code: `src/mtop/optimism.py` (`python -m mtop.optimism`), committed with this plan.
