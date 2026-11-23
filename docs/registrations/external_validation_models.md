# MTOP models frozen for external validation

Recorded 2026-11-22, before any MR CLEAN Registry data have been received. Every external validation names the fingerprint of each model it tests; a results file is accepted only for a fingerprint listed here.

| Model | Version | Fingerprint (SHA-256 of the models block) | Role | Archive |
|---|---|---|---|---|
| MTOP (binary models: mRS 0–2, death) | 1.0.0 | `77bb37a365930cdf3e5551d000f8ae40fdfa41f50a44c0593bdc7fea80ba4425` | **Primary** | [10.5281/zenodo.24396393](https://doi.org/10.5281/zenodo.24396393) |
| MTOP (binary models) | 1.1.0 | `841d810c28097e00d32ea2e0de6ea6f6b432f235a5a8e228ed7021b955556296` | Secondary | [10.5281/zenodo.24420424](https://doi.org/10.5281/zenodo.24420424) |
| MTOP ordinal (all seven mRS grades) | 1.0.0 | `78436c388af37724da897a7d29e95f103b12b2a3a8c5de47d30b4286b393ea46` | Secondary | Zenodo version under 10.5281/zenodo.24396392 (at release) |

Plans: [v1_1_plan.md](v1_1_plan.md), [ordinal_plan.md](ordinal_plan.md). The external analysis plan (the CONTRAST data request) sets the analyses and margins, including the comparison of MTOP ordinal with MR PREDICTS (its published ordinal formula with the collateral score): generalised concordance, ranked probability score and the calibration of the distribution. MRS CLEAN (synthetic data) is a pipeline check only, run with the MR CLEAN team's agreement.

## Fixed for the MR CLEAN Registry validation (added 23 November 2026)

Decided before any MR CLEAN Registry data have been received; the same for all three models.

1. **Primary analysis: the models as frozen.** Each model exactly as archived, with its recent-era intercept (binary models) or threshold shift (MTOP ordinal), run through the validation tool.
2. **Secondary analysis: after a baseline update.** The patients are split at random into two halves (seed 2026). On the first half, one shift of the baseline is estimated with the model's estimates held fixed: for each binary model its intercept (logistic regression with the model's log-odds as offset); for MTOP ordinal one common shift of all six thresholds (maximum likelihood). On the second half, each model is evaluated before and after the shift with the measures of the primary analysis. The slope is not changed.
3. **Reading,** as in the MRS CLEAN dry-run plan (version 2): a baseline update is needed when the observed/expected ratio of mRS 0–2 or of death lies outside 0.9–1.1 with a 95 % interval that excludes 1, and it is sufficient when, after it, the calibration slope lies within 0.8–1.2 (for MTOP ordinal, the slopes of mRS 0–2 and of death).
4. **Against MR PREDICTS** in both analyses: as frozen on every patient the comparator applies to, and after the update on the second half. MR PREDICTS keeps its published formula and MR CLEAN Registry intercepts.
5. The update is an analysis, not a new version: no shift estimated on MR CLEAN data enters the published models.

