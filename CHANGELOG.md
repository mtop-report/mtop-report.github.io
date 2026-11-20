# Changelog

Every published model version has its own fingerprint (SHA-256 of the models in
`artifacts/public/model_bundle.json`), shown on the app's About tab. Earlier versions stay archived
and selectable in the app, so every published result can be reproduced.

## 1.1.0 (planned release: November 2026)

Fingerprint `841d810c28097e00d32ea2e0de6ea6f6b432f235a5a8e228ed7021b955556296`. Plan: [docs/registrations/v1_1_plan.md](docs/registrations/v1_1_plan.md),
registered before any version 1.1 code, with amendments 1 and 2 committed before any change was
evaluated. Results: [docs/registrations/v1_1_results.md](docs/registrations/v1_1_results.md).

Same data, cohorts, outcomes, stages and temporal split as 1.0.0. Only the reference model
(penalised logistic regression with restricted cubic splines) changes:

- **One missing-value indicator per input** (2.1), and **only for inputs that at least 20 of the
  fitting patients lack** (2.2); below that, a missing value is set to the median. v1.0.0 had one
  indicator per model column, including copies for inputs spanning several columns.
- **No input that is an exact function of others** (2.3): pulse pressure, IV thrombolysis to
  puncture, NIHSS change in 24 hours, glucose above 7.8 mmol/L, clock time of onset and early
  neurological improvement are left out; their source inputs stay.
- **A category needs at least 20 fitting patients for its own term** (2.4): at 24 hours and on
  day 7, basilar + vertebral occlusion joins basilar.
- **Steadier tuning** (2.5): knots and penalty chosen by 5-fold cross-validation repeated 5 times;
  a pre-specified slope correction where the optimism-corrected calibration slope is outside 0.9–1.1, which applies only to death at 24 hours (× 1.102).
- **Exact solver**: the same penalised likelihood fitted by Newton–Raphson to convergence
  (amendment 2).
- **Not adopted** (3.1): an "update" model at 24 hours and day 7 (previous stage's log-odds, current
  NIHSS, the stage's events) gained less than the required 0.01 AUROC in nested cross-validation.
- **Day-7 death estimate labelled as limited evidence** (4.2).
- Coefficients of the reference model: 91 → 59 before thrombectomy, 119 → 77 at the end of the procedure, 127 → 76 at 24 hours, 135–148 → 82 on day 7 (deployed models).
- Other learners (linear regression, LightGBM, XGBoost, EBM, regression with interactions,
  stacking) are carried over from 1.0.0 unchanged. The selection rule of amendment 1, applied again
  with version 1.1 as the reference, chooses the reference at every stage and outcome.
- App: version 1.1 by default, v1.0.0 selectable as "archived"; the Validation tab shows version 1.1
  as its own row (its 2023–2025 results are not independent) and a version 1.1 card; the app's QR
  code points to mtop-report.github.io; favicon.ico.

## 1.0.0 (models frozen 5 November 2026; GitHub release and Zenodo archive 17 November 2026)

Fingerprint `77bb37a365930cdf3e5551d000f8ae40fdfa41f50a44c0593bdc7fea80ba4425`. Zenodo DOI
[10.5281/zenodo.24396393](https://doi.org/10.5281/zenodo.24396393). The pre-specified primary model
of the first manuscript and of the external validations under way.
