# MTOP ordinal model card

**MTOP ordinal 1.0.0** (fingerprint `78436c388af37724da897a7d29e95f103b12b2a3a8c5de47d30b4286b393ea46`): a proportional-odds model of the whole 90-day modified Rankin Scale (mRS), developed under a plan committed before any of its code ([plan](registrations/ordinal_plan.md), [results](registrations/ordinal_results.md), [diagnostics](model_diagnostics_ordinal.md), [changelog](../CHANGELOG.md)). It is a separate model next to the binary models of MTOP 1.1.0, which it does not change. Structured after TRIPOD+AI (Collins et al., BMJ 2024).

## 1. Purpose and intended use

- **Predicts:** the probability of each mRS grade 0–6 at 90 days after endovascular treatment for acute ischaemic stroke, at four moments (before the procedure, at its end, at 24 hours and on day 7), each from what is known then.
- **Role in the calculator:** it splits the two binary models' bands (independent, mRS 0–2; dependent, 3–5) into single grades (rule O1). The headline estimates of independence and death stay those of the binary models.
- **External validation:** frozen and sent as a secondary model (rule O2): v1.0.0 stays the primary model, v1.1 and this model are secondary.
- **Who for:** research, teaching and trial planning (for example the ordinal "shift" outcomes of stroke trials). **Not for** deciding whether or how to treat an individual patient. Not a medical device under EU MDR 2017/745.

## 2. Data and outcome

The registry export, cleaning, patients and stage cohorts of v1.1 (SHA-256 `1a19f809…`; see the [main model card](model_card.md), sections 2–4). Outcome: the 90-day mRS as recorded, seven grades; no grade had fewer than 20 development patients, so none was merged. Development 2005–2022; test 2023–2025 (sensitivity split 2005–2020 against 2021–2025); the deployed model is refitted on 2005–2025.

## 3. Model

Cumulative logit: logit P(mRS ≤ k) = θₖ − x′β, one coefficient vector for all six cut-points, thresholds ordered. Inputs: the v1.1 reference specification at each stage (restricted cubic splines for age, NIHSS, onset-to-puncture time, glucose and procedure duration; one missing-value indicator per input lacked by at least 20 patients). L2 penalty on β, fitted by Newton–Raphson; knots and penalty tuned by repeated cross-validation on ordinal log-loss. Explanations are exact: each input's coefficient × standardised value, the same at every cut-point.

| Stage | Patients | Coefficients | Knots | O3 slope factor | Recent-era threshold shift |
|---|---|---|---|---|---|
| Before thrombectomy | 1451 | 59 | 3 | 1.000 | −0.101 |
| End of procedure | 1431 | 77 | 3 | 1.000 | −0.110 |
| 24 hours | 1235 | 76 | 3 | 1.000 | −0.396 |
| Day 7 | 1139 | 82 | 3 | 0.887 | −0.284 |

## 4. Performance in the development years (2005–2022)

Nested cross-validation (15 outer folds) and the optimism-corrected bootstrap (500 resamples). RPS: ranked probability score (lower is better); C (utility): generalised concordance on the expected utility-weighted mRS.

| Stage | RPS, nested | RPS, corrected | C (utility), nested | AUROC mRS 0–2, nested (v1.1 binary model) | AUROC death, nested (v1.1 binary model) | Ordinal calibration slope, corrected |
|---|---|---|---|---|---|---|
| Before thrombectomy | 0.1651 | 0.1635 | 0.711 | 0.760 (0.753) | 0.768 (0.774) | 1.083 |
| End of procedure | 0.1550 | 0.1531 | 0.741 | 0.793 (0.789) | 0.793 (0.794) | 1.087 |
| 24 hours | 0.1040 | 0.1020 | 0.853 | 0.906 (0.893) | 0.899 (0.905) | 0.906 |
| Day 7 | 0.0897 | 0.0882 | 0.876 | 0.928 (0.914) | 0.909 (0.914) | 0.887 |

## 5. Temporal test (2023–2025; not independent)

| Stage | Test patients | RPS (95 % CI) | C (utility) | Mean utility predicted − observed |
|---|---|---|---|---|
| Before thrombectomy | 226 | 0.1491 (0.1373–0.1611) | 0.727 | +0.070 |
| End of procedure | 221 | 0.1419 (0.1300–0.1552) | 0.747 | +0.076 |
| 24 hours | 120 | 0.1060 (0.0910–0.1224) | 0.861 | +0.092 |
| Day 7 | 139 | 0.1018 (0.0885–0.1160) | 0.891 | +0.094 |

The inputs are those of v1.1, which was designed with the 2023–2025 results known, so this test is not independent.

## 6. Limitations

- One centre; no external validation yet (planned on the MR CLEAN Registry, section 12 of the plan).
- Proportional odds is an assumption: separate models at each cut-point differ from the shared one (results, section 4); each implied dichotomy's calibration is reported.
- Sample size: with 58, 76, 74, 77 coefficients in the development models, Riley et al.'s criteria call for more patients than the registry has at every cut-point (4,352–8,193 with the conservative anticipated R², against 1,000–1,225 patients; results, section 5). The penalty and the optimism-corrected checks are what guard against overfitting.
- **Death is the model's weak point.** At 24 hours and on day 7 the implied P(death) is poorly calibrated (nested slope 0.74, 0.67), and the recent-era threshold shift, which moves every patient towards worse grades, over-predicts death in 2023–2025 (results, section 8). Headline estimates of independence and death therefore come from the binary models.
- The 90-day mRS is missing in about one in five patients of 2023–2025; complete-case analysis.
- Death on day 7 carries the "limited evidence" label of v1.1 (fewer than 10 deaths in that stage's temporal test).

## 7. Privacy and maintenance

The bundle holds coefficients, thresholds, knots, medians and scaling only, no patient-level data; counts from 1 to 9 are shown as fewer than 10. The browser reimplementation (web/model.js) matches the Python model within 10⁻⁷ on every patient and on the synthetic demonstration cohort. A new version gets a new fingerprint, a changelog entry and a Zenodo version under the MTOP all-versions DOI (10.5281/zenodo.24396392).

