# MTOP model card

Structured after TRIPOD+AI (Collins et al., BMJ 2024). The full item-by-item checklist, with what
is still to do, is on the app's **Standards** tab. Numbers come from `artifacts/public/validation.json`;
`python -m mtop report` writes the complete tables to `reports/validation_report.md` (local).

## 1. Purpose and intended use

- **Predicts:** probability of functional independence (modified Rankin Scale 0–2) at 90 days
  after endovascular treatment for acute ischaemic stroke (primary), and of death by 90 days.
- **When:** four moments: before the procedure, at its end, 24 hours after treatment and on day 7.
  Each stage uses only what is known at that moment.
- **Who for:** research, teaching, prognostic discussion and trial planning by stroke neurologists
  and interventional radiologists at FNO Ostrava.
- **Not for:** deciding whether or how to treat an individual patient, or use outside the population
  it was developed in unless externally validated there. Not a medical device under EU MDR 2017/745.

## 2. Data

- Source: mechanical-thrombectomy registry of University Hospital Ostrava (FNO), 2005–2025, one
  registry export (48 columns, 1,577 records).
- Eligibility: adults (age ≥ 18 or not recorded) with a recorded 90-day mRS, whatever the procedure
  turned out to be: 1,577 → 1,572 adults → 1,451 analysed, including 88 diagnostic angiographies
  only, 29 patients with an impossible time order (that time set to missing) and 2 without age.
- Stage cohorts (not cumulative): end of procedure needs the final mTICI (1,431); 24 hours a 24-h
  NIHSS score (1,235); day 7 a day-7 NIHSS score (1,139).
- Outcome: 90-day mRS as recorded (assessor and method are not recorded in the export; Standards
  tab, items 8a–8c). Missing in 7.7 % of adults overall and 21.5 % in 2023–2025; complete-case,
  with an analysis of who lacks it on the Registry page.
- Cleaning: range checks, logged numeric conversion, onset-year repair from neighbouring IDs and
  cross-field checks (including the procedure label against the extraction device, stent and IAT
  columns); flagged values are kept and listed for review at FNO, where the local data studio
  ([studio.md](studio.md)) shows them in every version of the data and records corrections. NIHSS 43 (died) and 99
  (not assessable) are status codes, never scores. The full data report (flow, checks, codings,
  missing data, Table 1) is on the Registry page; the dictionary is in [data_dictionary.md](data_dictionary.md).

## 3. Predictors

49 inputs in `config/features.yaml`: the 33 pre-procedural predictors of the 2026 FNO analysis
(including derived night onset, IVT-to-puncture time, pulse pressure and hyperglycaemia); at the
end of the procedure mTICI, technique (aspiration, stent retriever, Solumbra, combined procedure,
no thrombectomy, diagnostic only), stenting, angioplasty, intra-arterial lysis, procedure duration,
onset to recanalisation and anaesthesia; at 24 hours NIHSS, its change, early neurological improvement and
haemorrhage; on day 7 NIHSS, oedema and decompressive craniectomy. Not used: thrombolytic agent
(the IVT variables carry it), cholesterol (missingness tied to outcome) and aetiology (known only
after the decisions the model supports). Occlusion site in the models: intracranial ICA, tandem
(cervical ICA/CCA lesion + intracranial occlusion), cervical ICA/CCA, MCA M1, MCA M2/M3, basilar,
basilar + vertebral, vertebral/PCA. Each detailed registry code is kept; a site needs at least 10
patients for its own term, so the ACA (fewer) is not used by the models yet and the calculator
gives no estimate for it.

## 4. Missing data

No predictor values are filled in before modelling. The regressions use the development median
plus a missing-value indicator; the tree models learn which branch missing values take. In the
calculator an unknown input is averaged over 20 registry values (2016 onwards). Inputs that do not
apply (onset to IVT without IVT) stay missing, as in the registry.

The missing-value indicators encode who had missing data at FNO, and several rest on a handful of
patients: at most 9 development patients lack age, sex, affected side, IVT, where IVT was given,
hypertension, atrial fibrillation / arrhythmia, intra-arterial lysis or procedure duration, and their
terms reach 1–4 log-odds. The calculator never uses them: these inputs always have a value there or,
if optional, are averaged over registry values. A partner's file can lack a whole column, so since November 2026
*Validate on your data* averages any unknown value over registry values too (every input has 20; for
where IVT was given, from thrombolysed patients). The models and their fingerprint are unchanged.
This was found in a pre-specified dry run on synthetic registry data (MRS CLEAN, DOI
10.5281/zenodo.18846304): where IVT was given is not recorded there, and the term for it (+2.18
log-odds of death at 24 hours, learned from 4 of 1,235 patients) applied to every thrombolysed
patient but none without IVT. The next
model version should give an input a missing-value term only when at least 20 development patients
lack it. Counted over all years (the data the deployed models were fitted on), 19 inputs from
admission to day 7 have fewer than 20 patients without them (13 of them fewer than 10); over the
development years alone the number can only be the same or larger. Counts, outcomes among the
patients without each input, and rare categories: [model diagnostics](model_diagnostics.md).

## 5. Model development

- Learners per stage and outcome: logistic regression with restricted cubic splines (reference;
  3–5 knots), linear logistic regression, LightGBM (monotone constraints), XGBoost and an MLP,
  each with 50 candidate settings on the same 5 cross-validation folds of the development years.
- [Amendment 1](amendment_1.md) (19 October 2026, committed before the runs) adds four candidates
  that can be published with exact explanations: an EBM with at least 20 patients per leaf (10 pair
  terms on 32 bins), the reference with age × NIHSS and onset-to-puncture × age, the reference fitted
  on 10 chained-equation imputations, and stacking (logistic meta-model on the out-of-fold log-odds of
  the reference, XGBoost and an EBM). Validated for comparison only, never published: the MLP,
  CatBoost, the default EBM, averaged probabilities and FT-Transformers (default; tuned on the
  admission stage only). TabPFN v2 and TabICL v2 are listed but not yet run (local weights only).
- Every tree leaf and EBM bin summarises at least 20 patients; comparison-only learners lack that
  guarantee or an exact explanation.
- Model choice, fixed before testing (2026 analysis; amendment 1): the reference stays unless a
  candidate's AUROC on the temporal test is higher with a Holm-adjusted paired-bootstrap p below
  0.05 (over the seven alternatives), or in nested cross-validation it gains ≥ 0.01 AUROC on average
  and is better in ≥ 80 % of folds, in both cases with calibration-in-the-large at most 0.05 worse.
  Chosen: regression with splines for every stage and outcome. No alternative met the Holm-adjusted
  temporal criterion; in nested cross-validation the closest was regression with multiple
  imputation (+0.005 AUROC on average, better in 9 of 15 folds; the rule asks for +0.01 and 12).
- Sample size: with 33 pre-procedural predictors (92 regression parameters) the development data
  fall short of the Riley criteria (Standards tab); estimation is penalised (L2) and the expected
  shrinkage is reported.
- Tuned settings of the reference ([model diagnostics](model_diagnostics.md)): 3 knots at every
  stage and outcome except death at day 7 (4 knots), so the smallest option was chosen almost
  everywhere; penalty C 0.009–0.037 on a grid of 10⁻⁴–10², none at either end of the grid.

## 6. Validation

- Primary temporal split: development 2005–2022 (1,225), test 2023–2025 (226; 60 good outcomes,
  40 deaths). Sensitivity split: test 2021–2025 (379; 110 and 77).
- Nested cross-validation on all years (5 folds × 3 repeats, admission stage, mRS 0–2).
- Measures: AUROC (paired-bootstrap difference and DeLong's test against the reference), Brier,
  calibration intercept and slope, O/E, ICI, decision curves, thresholds, subgroups (with bootstrap
  CIs), recalibration on newer patients for every model and score, SHAP for every model on the same
  test patients; 2,000 bootstrap resamples. Every published or bedside score the calculator shows is
  tested at that stage (including MR PREDICTS refitted without collaterals, THRIVE-EVT, CLEAR, BET).
- Sensitivity of the test result to the 62 test-year patients without a 90-day outcome (extremes and
  20 imputations): Validation tab.
- Results: README and the Validation tab. Before thrombectomy, mRS 0–2: regression with splines
  AUROC 0.809 (0.743–0.867). Calibration of the development model on the test years, before any
  recalibration: calibration-in-the-large −0.41 (−0.72 to −0.12), i.e. good outcomes over-predicted
  in 2023–2025, and slope 1.47 (1.06–2.03). A slope above 1 means the estimates are too close
  together (too timid), the opposite of the usual sign of overfitting; linear regression without
  splines gives 1.44, so the splines are not the cause. Every stage: [model diagnostics](model_diagnostics.md).
- Bootstrap internal validation (plan registered before the run:
  [bootstrap_optimism_plan.md](registrations/bootstrap_optimism_plan.md); 500 samples of the
  development patients with the whole pipeline repeated), before thrombectomy:
  - AUROC: mRS 0–2 0.796 apparent, **0.761 optimism-corrected** (nested cross-validation: 0.762);
    death 0.826 and **0.790**.
  - Calibration slope, optimism-corrected: **1.09** (mRS 0–2) and **1.10** (death). The tuned
    penalty shrinks the estimates about 10 % more than the development data need. The test years' 1.47 for mRS 0–2 is
    larger than this, so it reflects change over time, or chance (its interval reaches 1.06), more
    than the penalty. For death the test-year slope was 0.94 (0.53–1.47).
  - Stability: 3 knots in 83 % and 86 % of samples, the penalty within a narrow range and never
    at the end of the grid, no convergence warnings. Each patient's estimated risk, however, varied
    across the bootstrap models by a median of 21 (mRS 0–2) and 15 (death) percentage points
    (2.5th–97.5th percentile range), as expected with about 3–5 events per parameter (Riley criteria
    not met). Individual estimates are therefore uncertain; version 1.1 should have fewer
    parameters, more patients, or both. MTOP v1.0.0 is unchanged.

## 7. Deployed models

Every publishable learner (regression with splines, linear regression, LightGBM, XGBoost, EBM,
regression with interactions, stacking) is refitted on all 1,451 patients with its development
settings, and its intercept is re-estimated on the pooled years 2023–2025 from cross-validated
predictions (temporal recalibration). Regression with multiple imputation is validated but not
offered in the calculator, because the browser cannot repeat its random imputation of a patient's
missing inputs. The calculator uses the chosen model by default; the others can be selected, each
with exact attributions computed in the browser. The model bundle carries a SHA-256 fingerprint
shown on the About tab.

## 8. Limitations

- Single centre; applicability elsewhere is unknown until externally validated (Multicentre tab).
- Small temporal test set; wide intervals, especially for death at day 7 (6 events).
- Recent follow-up gaps (24-h NIHSS, 90-day mRS) make the later stages rest on selected subsets.
- ASPECTS from 2016, anaesthesia from about 2012; collaterals and CT perfusion volumes recorded
  only for new patients (optional template columns, `config/future.yaml`; tested once ≥ 100 + 100
  patients have them), number of passes not recorded.
- Outcomes shifted over time; the published intercepts need yearly checking (Adaptive tab).

## 9. Privacy

Only models and aggregates are published: tree leaves ≥ 20 patients, groups under 10 hidden, no
identifiers, dates or hospital names. `tests/test_privacy.py` checks every public file.

## 10. Maintenance

Update policy (draft) on the Adaptive tab: quarterly observed-vs-expected monitoring, yearly
discrimination and slope checks, intercept re-estimation on recent complete years, full refit with
temporal validation when performance falls outside its validation interval.
