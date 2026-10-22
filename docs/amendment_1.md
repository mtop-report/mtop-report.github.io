# Amendment 1 to the MTOP analysis plan

Date: 19 October 2026. Committed before any of the new learners below was validated in this
pipeline; the git history of this file and of `config/settings.yaml` is the record.

## What changes

**1. More candidates for the model choice.** The calculator offers only models that can be
published without patient-level information (every tree leaf or bin summarises at least 20 patients;
no training rows) and evaluated in the browser with exact explanations. Four learners that meet this
join the candidates (`selection.deployable`):

| Learner | Specification |
|---|---|
| EBM | interpret `ExplainableBoostingClassifier`, default settings except at least 20 patients per leaf (`min_samples_leaf = 20`; the 2026 report used the default of 4). Only the cut points where a shape function changes are published. |
| LR + interactions | the reference specification (L2 logistic regression with restricted cubic splines; knots and penalty chosen by cross-validation) plus the two interactions of the 2026 report: age × NIHSS on admission and onset-to-puncture time × age, as products of centred values. |
| LR + multiple imputation | the reference specification with its tuned knots and penalty, fitted on 10 chained-equation imputations of the predictors (Bayesian ridge, posterior draws; the outcome is not used) stacked with weight 1/10, instead of median imputation with missing-value indicators. |
| Stacking | a logistic meta-model (L2, C = 1) on the out-of-fold log-odds (the 5 shared folds) of the reference, XGBoost (both with their tuned settings) and EBM (at least 20 patients per leaf; 4 outer bags, as in the 2026 report). |

Unchanged: the reference (regression with splines) and the other candidates (linear regression,
LightGBM, XGBoost).

**2. A stricter temporal-test criterion.** With 7 alternatives instead of 3, the chance that one of
them beats the reference on the temporal test by luck alone is larger. The temporal criterion now
uses Holm's step-down procedure over the candidates: a candidate meets it when its AUROC difference
against the reference is positive and its Holm-adjusted paired-bootstrap p-value is below 0.05
(before: an unadjusted 95% interval above zero). The nested cross-validation criterion (mean AUROC
gain at least 0.01 and better in at least 80% of the 15 outer folds), the calibration condition
(mean |calibration-in-the-large| at most 0.05 worse than the reference's) and the tie-break (highest
temporal AUROC) are unchanged. The rule can only become harder to meet.

**3. Learners validated for comparison only.** They are reported next to the candidates, with
the same split, folds and bootstrap, but are never chosen or published:

- MLP (as before): a neural network, without the 20-patient guarantee;
- CatBoost, default settings as in the 2026 report: its symmetric trees cannot guarantee 20 patients
  per leaf, and a version that could would be a different model;
- probability averaging of the reference, XGBoost and EBM: the average of probabilities has no exact
  additive explanation (stacking is the explainable form of the same idea);
- FT-Transformer, default and tuned (12 trials): neural networks, as the MLP;
- TabPFN v2 and TabICL v2: in-context learners that read the training patients at every prediction,
  so they can never be published; run only from local weight files, never through an online service.

**4. Every score the calculator shows is validated at that stage**, and four published models join
the benchmarks: MR PREDICTS (its pre-treatment predictor set, refitted without the CTA collateral
score, which the registry does not record), THRIVE-EVT (the published formula), CLEAR and BET.

**5. New predictors are collected before they are modelled.** CT perfusion (rCBF < 30% volume, Tmax
> 6 s volume, mismatch volume, mismatch ratio, relative mismatch; "not done" recorded as such), the
CTA collateral score (Tan, 0-3) and serum GFAP (with the time from onset to sampling and the assay)
enter the partner template, the registry checks and the calculator's preview now, but no model.
Their added value is tested once at least 100 patients with mRS 0-2 and 100 with mRS 3-6 at 90 days
have the variable recorded (for death: 100 deaths and 100 survivors): the current model's linear
predictor as an offset plus the new variable, on the most recent patients held out in time
(likelihood-ratio test, AUROC difference, calibration). Only a variable that adds information starts
a new model version.

## What was known when this was written

- The 2026 FNO report and the conference poster compared most of these learners on an earlier
  version of the data (1,422 patients; temporal test 221): none beat the logistic regression.
- On 19 October 2026, before this amendment, exploratory timing runs on the primary split's admission
  stage (mRS 0-2) gave AUROC 0.805 for EBM with default settings, 0.774 for CatBoost, 0.793 for a
  regression with all pairwise interactions of seven predictors and 0.796 for a linear regression on 5
  imputations, against 0.809 for the reference. The specifications in this amendment differ from
  those runs (20 patients per leaf, the report's two interactions, the reference's splines, 10
  imputations).

The amendment is therefore not blind to these results; the stricter temporal criterion makes sure that
knowing them cannot make a replacement easier.

## Why

- The 2026 analysis plan applied the same rule to these alternatives; the app had narrowed the
  candidates to the learners it could already publish.
- A model offered in the calculator must be publishable and explainable, and its validation must be
  that of the published version.

## Amendment 1a (19 October 2026, the same night, before the validation runs)

**The candidate EBM keeps 10 pair terms on 32 bins.** With interpret's default number of pair terms
(three per input, 126 at admission) on 64 bins, every published EBM is 1.5 to 2 MB; the calculator
would carry 16 of them (the EBM itself and the stacking's EBM, for 4 stages and 2 outcomes), about 30 MB,
on a page used on phones. The candidate EBM, also inside the stacking, is therefore: at least 20
patients per leaf, 10 pair terms (interpret's earlier default) on at most 32 bins, otherwise the
defaults (about 100-130 kB per model). The **default EBM** (interpret's defaults throughout, as in the
2026 report) joins the comparison-only learners, so its result is reported either way.

What was known when 1a was written: a code check of the amendment-1 specifications on the primary
split's admission stage (mRS 0-2) had shown temporal-test AUROCs of 0.807 for the EBM with 126 pairs and
at least 20 patients per leaf, 0.806 for LR + interactions and 0.802 for LR + multiple imputation
(reference 0.809). The change is made for the size of the published page; no alternative EBM was
evaluated on the test patients to choose it (only its size on the development patients was measured).
The other specifications are unchanged.

## Notes on carrying out amendment 1 (22 October 2026, after the runs)

Facts about how the runs were done. None of them changes the rule, the candidates or what was
compared on the test patients.

- **FT-Transformer, tuned** (12 Optuna trials) took about 27 minutes per stage and outcome on the
  analysis computer (2 CPU cores, no GPU), so it was run on the primary split's admission stage
  only, for both outcomes. The default FT-Transformer ran on every stage of both splits and in the
  nested cross-validation. Both are comparison-only learners and cannot affect the model choice.
- **TabPFN v2 and TabICL v2** have not been run: their weight files could not be transferred to the
  analysis computer (the upload failed). They stay on the list of comparison-only learners. Once the
  files are on a computer inside the hospital, `python -m mtop compare --tag tab --learner "TabPFN v2"
  --learner "TabICL v2"` runs them offline, without any online service, and `python -m mtop compare
  --merge` adds their results, dated by the run.
- **Nested cross-validation** of the amendment-1 and comparison-only learners re-tunes the reference
  and XGBoost inside each of the 15 outer folds (20 trials, the nested budget) and passes their
  settings to the stacking and the multiple imputation, as `src/mtop/compare.py` documents.
- **Chunks.** The runs were split by split, stage and fold and merged afterwards
  (`python -m mtop compare --merge`). All were run against the same saved training results of
  18 October 2026; chunks written from 22 October also carry a fingerprint of the reference's saved
  test predictions, so a chunk computed on other training results is not merged.
- **Reporting added alongside.** SHAP values of every validated model on the same test patients
  (`python -m mtop shap`; exact where the model allows it, sampled Shapley values for the MLP and the
  averaged probabilities, expected gradients for the FT-Transformers), DeLong's test next to the paired bootstrap, the updating analysis for every
  published score, and three analyses from the 2026 report: missing 90-day outcomes in the test
  years, why outcomes fell in 2023–2025, and time to groin puncture. They describe the models; none
  of them enters the model choice.
