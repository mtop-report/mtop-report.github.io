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

## 5. Model development

- Learners per stage and outcome: logistic regression with restricted cubic splines (reference;
  3–5 knots), linear logistic regression, LightGBM (monotone constraints), XGBoost and an MLP,
  each with 50 candidate settings on the same 5 cross-validation folds of the development years.
- Every tree leaf summarises at least 20 patients. The MLP is validated but not published.
- Model choice, fixed before testing (from the 2026 analysis): the reference stays unless a
  candidate's paired 95 % CI of the AUROC difference on the temporal test lies above zero, or in
  nested cross-validation it gains ≥ 0.01 AUROC on average and is better in ≥ 80 % of folds, in
  both cases with calibration-in-the-large at most 0.05 worse. Chosen: regression with splines for
  every stage and outcome.
- Sample size: with 33 pre-procedural predictors (92 regression parameters) the development data
  fall short of the Riley criteria (Standards tab); estimation is penalised (L2) and the expected
  shrinkage is reported.

## 6. Validation

- Primary temporal split: development 2005–2022 (1,225), test 2023–2025 (226; 60 good outcomes,
  40 deaths). Sensitivity split: test 2021–2025 (379; 110 and 77).
- Nested cross-validation on all years (5 folds × 3 repeats, admission stage, mRS 0–2).
- Measures: AUROC, Brier, calibration intercept and slope, O/E, ICI, decision curves, thresholds,
  subgroups (with bootstrap CIs), recalibration on newer patients; 2,000 bootstrap resamples.
- Results: README and the Validation tab. Before thrombectomy, mRS 0–2: regression with splines
  AUROC 0.809 (0.743–0.867); development model calibration intercept −0.41, slope 1.47.

## 7. Deployed models

Every publishable learner (regression with splines, linear regression, LightGBM, XGBoost) is
refitted on all 1,451 patients with its development settings; its intercept is re-estimated on the
pooled years 2023–2025 from cross-validated predictions (temporal recalibration). The calculator uses
the chosen model by default; the others can be selected. The model bundle carries a SHA-256
fingerprint shown on the About tab.

## 8. Limitations

- Single centre; applicability elsewhere is unknown until externally validated (Multicentre tab).
- Small temporal test set; wide intervals, especially for death at day 7 (6 events).
- Recent follow-up gaps (24-h NIHSS, 90-day mRS) make the later stages rest on selected subsets.
- ASPECTS from 2016, anaesthesia from about 2012; collaterals, core volume and passes not recorded.
- Outcomes shifted over time; the published intercepts need yearly checking (Adaptive tab).

## 9. Privacy

Only models and aggregates are published: tree leaves ≥ 20 patients, groups under 10 hidden, no
identifiers, dates or hospital names. `tests/test_privacy.py` checks every public file.

## 10. Maintenance

Update policy (draft) on the Adaptive tab: quarterly observed-vs-expected monitoring, yearly
discrimination and slope checks, intercept re-estimation on recent complete years, full refit with
temporal validation when performance falls outside its validation interval.
