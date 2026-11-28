# Changelog

Every published model version has its own fingerprint (SHA-256 of the models in
`artifacts/public/model_bundle.json`), shown on the app's About tab. Earlier versions stay archived
and selectable in the app, so every published result can be reproduced.

## Unreleased: version 2.0 imaging module, registered (28 November 2026)

No model changes. Plan [docs/registrations/v2_imaging_plan.md](docs/registrations/v2_imaging_plan.md),
committed before the ISLES'24 data are downloaded: a fixed CT perfusion volume extractor (core,
Tmax > 6 s, mismatch) checked against the follow-up infarct, and a technical demonstration node
that writes the ISLES'24 cases in the network template. Code `src/mtop/isles.py`
(`python -m mtop isles`), settings `config/isles24.yaml`, tests on synthetic maps
`tests/test_isles.py`. Optional dependencies `nibabel` and `pydicom` (`pip install .[imaging]`).

Amendment 1 of the same plan: a four-centre network simulation (`python -m mtop fed-multi`, code
`src/mtop/fed/demo_multi.py`, tests `tests/test_fed_multi.py`): Ostrava, two disjoint random
samples of 1,200 synthetic MR CLEAN patients, and ISLES'24. A software test, not a validation;
shown in the app only after the MR CLEAN team has seen it. The extractor reads NIfTI and greyscale
DICOM maps (`python -m mtop perfusion`); it is offered in the app only if it passes the plan's check.

## MTOP ordinal 1.0.0 (22 November 2026)

A new, separate model: a proportional-odds regression of all seven 90-day mRS grades at each stage,
with the inputs of the v1.1 reference. Fingerprint `78436c388af37724da897a7d29e95f103b12b2a3a8c5de47d30b4286b393ea46`
(`artifacts/public/ordinal_bundle.json`). Zenodo DOI
[10.5281/zenodo.24463897](https://doi.org/10.5281/zenodo.24463897) (release ordinal-v1.0.0). Plan: [docs/registrations/ordinal_plan.md](docs/registrations/ordinal_plan.md),
committed before any of its code, with amendment 1 committed before any result. Results:
[docs/registrations/ordinal_results.md](docs/registrations/ordinal_results.md); model card:
[docs/model_card_ordinal.md](docs/model_card_ordinal.md).

- **The binary models (1.1.0) are unchanged** and keep the headline estimates of independence and death.
- **Calculator (rule O1):** the independent and dependent bands are now split into single grades by
  the ordinal model's shares instead of the v1.1 layer (lower nested cross-validated ranked
  probability score at both primary stages).
- **External validation (rule O2):** frozen and registered as a secondary model next to v1.0.0
  (primary) and v1.1 ([external_validation_models.md](docs/registrations/external_validation_models.md)).
- **Slope correction (rule O3):** day 7 only (× 0.887).
- **Validation tab:** an MTOP ordinal card (development years and the 2023–2025 test, which is not
  independent). **Multicentre tab:** the validation tool also evaluates the ordinal model and writes
  its results, with its own fingerprint, into the results file.
- **Known weakness:** death. The implied P(death) is poorly calibrated at 24 hours and day 7, and the
  recent-era threshold shift over-predicts it in 2023–2025; this is why the binary models keep the
  headline estimates.
- A comparison-only ordinal LightGBM did slightly better at the end of the procedure in both the
  nested cross-validation and the 2023–2025 test; it is not deployed.

## 1.1.0 (21 November 2026)

Fingerprint `841d810c28097e00d32ea2e0de6ea6f6b432f235a5a8e228ed7021b955556296`. Zenodo DOI
[10.5281/zenodo.24420424](https://doi.org/10.5281/zenodo.24420424); all versions: 10.5281/zenodo.24396392. Plan: [docs/registrations/v1_1_plan.md](docs/registrations/v1_1_plan.md),
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
