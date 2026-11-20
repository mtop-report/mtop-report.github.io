# MTOP version 1.1: results of the registered evaluation

Written 2026-11-20 by `python -m mtop.v11 report` from the analysis outputs. Plan: [v1_1_plan.md](v1_1_plan.md) (17 November 2026) with amendment 1 (version 1.1 = changes 2.1–2.5, 3.1 and 4.2 on the data of v1.0.0) and amendment 2 (pipeline check to numerical tolerance, exact solver, how the plan is carried out), both committed before any change was evaluated. Development years 2005–2022 decide; the 2023–2025 test is reported but is not independent.

## Summary

- **Pipeline check passed** (to the tolerance of amendment 2): with every change switched off, the code reproduces v1.0.0; every patient's estimate within 4e-06.
- **Changes 2.1–2.4 are kept** (rule a): optimism-corrected AUROC −0.0003 (before thrombectomy), −0.0031 (24 hours); log-loss −0.0021, +0.0039. Coefficients of the deployed reference: 91 → 59 before thrombectomy, 135–148 → 82 on day 7.
- **Slope correction** (rule b) applies only to 24 hours, death (factor 1.102).
- **The update model is not adopted** (rule c): it gained +0.007 AUROC for mRS 0–2 at 24 hours and day 7, where +0.010 was required. It did give much narrower individual risk ranges (section 3).
- **No candidate learner replaces the reference** when the selection rule of amendment 1 is applied again with version 1.1 as the reference.
- **Deployed version 1.1**: fingerprint `841d810c28097e00…`; v1.0.0 archived and selectable in the app.

## 1. Pipeline check (section 5, step 2; amendment 2)

- Registry export SHA-256 `1a19f809ed688a3940c38ed4d0b1198f0a978b806f9ed7b50721777d980d50f7`.
- Data checks, cohorts and events at every stage, model columns: identical to v1.0.0.
- Reference and linear regression re-tuned on 2005–2022: the same knots and penalties at all eight stage–outcome pairs; 2023–2025 results identical to 5 decimals.
- Redeployed on all years: every exported number other than the coefficients and the recent-era intercepts is identical; largest coefficient difference 5.7e-06, largest recent-era intercept difference 1e-06, largest difference in any patient's estimate 3.9e-06 (criterion: below 1e-05).
- **Passed to tolerance: yes.** Bit for bit: no, because where L-BFGS stops depends on the machine (amendment 2).

## 2. Nested cross-validation, 2005–2022 (5 outer folds × 3 repeats)

Mean over the 15 outer folds. *1.0.0 specification*: v1.0.0's model refitted with the exact solver; *+ 2.1*, *+ 2.2*, *+ 2.3*: each change added in turn (for the record); *2.1-2.4*: the four simplifications, the step that rule (a) judges; *1.1 full structure*: plus change 2.5's tuning (5-fold CV repeated 5 times); *update model*: change 3.1 (at day 7 built on the 24-hour full structure, the model chosen there, or on the 24-hour update model); *NIHSS alone*: the published points-to-risk benchmark.

**Before thrombectomy, mRS 0–2** (n = 1225, events 493)

| Specification | AUROC | Log-loss | Brier | Calibration intercept | Calibration slope | Coefficients |
|---|---|---|---|---|---|---|
| 1.0.0 specification | 0.7489 | 0.5838 | 0.1981 | +0.005 | 1.044 | 91 |
| + 2.1 | 0.7489 | 0.5825 | 0.1978 | +0.005 | 1.057 | 78 |
| + 2.2 | 0.7531 | 0.5763 | 0.1958 | +0.006 | 1.051 | 63 |
| + 2.3 | 0.7525 | 0.5759 | 0.1958 | +0.004 | 1.033 | 55 |
| 2.1-2.4 | 0.7531 | 0.5755 | 0.1956 | +0.004 | 1.032 | 54 |
| 1.1 full structure | 0.7532 | 0.5757 | 0.1957 | +0.003 | 1.057 | 54 |

**Before thrombectomy, death** (n = 1225, events 269)

| Specification | AUROC | Log-loss | Brier | Calibration intercept | Calibration slope | Coefficients |
|---|---|---|---|---|---|---|
| 1.0.0 specification | 0.7731 | 0.4464 | 0.1434 | −0.001 | 1.061 | 91 |
| + 2.1 | 0.7732 | 0.4461 | 0.1434 | −0.002 | 1.063 | 78 |
| + 2.2 | 0.7776 | 0.4428 | 0.1424 | +0.005 | 1.066 | 62 |
| + 2.3 | 0.7761 | 0.4439 | 0.1427 | +0.006 | 1.073 | 55 |
| 2.1-2.4 | 0.7766 | 0.4436 | 0.1426 | +0.006 | 1.076 | 54 |
| 1.1 full structure | 0.7768 | 0.4437 | 0.1426 | +0.007 | 1.086 | 54 |

**End of procedure, mRS 0–2** (n = 1210, events 486)

| Specification | AUROC | Log-loss | Brier | Calibration intercept | Calibration slope | Coefficients |
|---|---|---|---|---|---|---|
| 1.0.0 specification | 0.7812 | 0.5549 | 0.1857 | +0.011 | 1.051 | 119 |
| + 2.1 | 0.7833 | 0.5531 | 0.1849 | +0.011 | 1.065 | 99 |
| + 2.2 | 0.7895 | 0.5431 | 0.1824 | +0.013 | 1.067 | 79 |
| + 2.3 | 0.7883 | 0.5437 | 0.1828 | +0.012 | 1.058 | 72 |
| 2.1-2.4 | 0.7890 | 0.5432 | 0.1825 | +0.012 | 1.061 | 71 |
| 1.1 full structure | 0.7891 | 0.5429 | 0.1824 | +0.011 | 1.060 | 71 |

**End of procedure, death** (n = 1210, events 265)

| Specification | AUROC | Log-loss | Brier | Calibration intercept | Calibration slope | Coefficients |
|---|---|---|---|---|---|---|
| 1.0.0 specification | 0.7879 | 0.4359 | 0.1401 | +0.001 | 1.136 | 119 |
| + 2.1 | 0.7886 | 0.4349 | 0.1399 | +0.001 | 1.123 | 99 |
| + 2.2 | 0.7965 | 0.4298 | 0.1384 | +0.002 | 1.129 | 79 |
| + 2.3 | 0.7960 | 0.4313 | 0.1388 | +0.001 | 1.150 | 72 |
| 2.1-2.4 | 0.7960 | 0.4311 | 0.1388 | +0.000 | 1.138 | 71 |
| 1.1 full structure | 0.7959 | 0.4308 | 0.1387 | +0.001 | 1.125 | 71 |

**24 hours, mRS 0–2** (n = 1115, events 470)

| Specification | AUROC | Log-loss | Brier | Calibration intercept | Calibration slope | Coefficients |
|---|---|---|---|---|---|---|
| 1.0.0 specification | 0.8942 | 0.4105 | 0.1309 | +0.000 | 1.136 | 126 |
| + 2.1 | 0.8949 | 0.4085 | 0.1303 | +0.000 | 1.142 | 106 |
| + 2.2 | 0.8968 | 0.4022 | 0.1284 | +0.006 | 1.121 | 80 |
| + 2.3 | 0.8929 | 0.4078 | 0.1306 | +0.006 | 1.070 | 72 |
| 2.1-2.4 | 0.8936 | 0.4076 | 0.1303 | +0.005 | 1.064 | 71 |
| 1.1 full structure | 0.8934 | 0.4069 | 0.1303 | +0.007 | 1.059 | 71 |
| update model | 0.9008 | 0.3984 | 0.1272 | +0.002 | 1.048 | 5 |
| NIHSS alone | 0.8905 | 0.4140 | 0.1315 | +0.003 | 1.043 | 2 |

**24 hours, death** (n = 1115, events 222)

| Specification | AUROC | Log-loss | Brier | Calibration intercept | Calibration slope | Coefficients |
|---|---|---|---|---|---|---|
| 1.0.0 specification | 0.9024 | 0.2975 | 0.0909 | +0.008 | 1.092 | 126 |
| + 2.1 | 0.9039 | 0.2963 | 0.0904 | +0.004 | 1.098 | 106 |
| + 2.2 | 0.9052 | 0.2938 | 0.0899 | +0.011 | 1.092 | 80 |
| + 2.3 | 0.9038 | 0.2959 | 0.0909 | +0.015 | 1.089 | 72 |
| 2.1-2.4 | 0.9040 | 0.2952 | 0.0905 | +0.015 | 1.092 | 71 |
| 1.1 full structure | 0.9037 | 0.2958 | 0.0906 | +0.016 | 1.087 | 71 |
| update model | 0.9053 | 0.2954 | 0.0900 | −0.001 | 1.061 | 5 |
| NIHSS alone | 0.8808 | 0.3196 | 0.0958 | −0.001 | 1.022 | 2 |

**Day 7, mRS 0–2** (n = 1000, events 467)

| Specification | AUROC | Log-loss | Brier | Calibration intercept | Calibration slope | Coefficients |
|---|---|---|---|---|---|---|
| 1.0.0 specification | 0.9015 | 0.3962 | 0.1271 | +0.010 | 1.038 | 135 |
| + 2.1 | 0.9024 | 0.3935 | 0.1265 | +0.010 | 1.054 | 114 |
| + 2.2 | 0.9105 | 0.3767 | 0.1210 | +0.003 | 1.034 | 84 |
| + 2.3 | 0.9140 | 0.3694 | 0.1186 | −0.008 | 1.042 | 76 |
| 2.1-2.4 | 0.9141 | 0.3702 | 0.1184 | −0.007 | 1.033 | 75 |
| 1.1 full structure | 0.9145 | 0.3700 | 0.1181 | −0.007 | 1.031 | 75 |
| update model (24 h full) | 0.9219 | 0.3581 | 0.1104 | −0.019 | 1.035 | 6 |
| update model (24 h update) | 0.9190 | 0.3672 | 0.1129 | −0.009 | 1.034 | 6 |
| NIHSS alone | 0.9184 | 0.3630 | 0.1112 | +0.005 | 1.044 | 2 |

**Day 7, death** (n = 1000, events 130)

| Specification | AUROC | Log-loss | Brier | Calibration intercept | Calibration slope | Coefficients |
|---|---|---|---|---|---|---|
| 1.0.0 specification | 0.9118 | 0.2222 | 0.0643 | −0.006 | 1.199 | 148 |
| + 2.1 | 0.9116 | 0.2216 | 0.0641 | −0.011 | 1.192 | 120 |
| + 2.2 | 0.9109 | 0.2199 | 0.0636 | −0.016 | 1.153 | 90 |
| + 2.3 | 0.9116 | 0.2200 | 0.0639 | −0.017 | 1.144 | 76 |
| 2.1-2.4 | 0.9123 | 0.2195 | 0.0638 | −0.015 | 1.152 | 76 |
| 1.1 full structure | 0.9130 | 0.2191 | 0.0637 | −0.016 | 1.158 | 75 |
| update model (24 h full) | 0.9121 | 0.2157 | 0.0624 | −0.009 | 1.071 | 6 |
| update model (24 h update) | 0.9077 | 0.2200 | 0.0634 | +0.001 | 1.061 | 6 |
| NIHSS alone | 0.8952 | 0.2291 | 0.0654 | −0.004 | 1.036 | 2 |

## 3. Full-pipeline bootstrap, 2005–2022 (500 resamples, seed 2026)

Optimism-corrected measures (Harrell). The resamples are those of the v1.0.0 bootstrap, drawn over the patients before thrombectomy; each later stage uses the resampled patients eligible for it. Risk range: median width of a patient's 95% range across the 500 bootstrap models. The 1.0.0 specification reproduces the published v1.0.0 bootstrap before thrombectomy (corrected AUROC 0.761 and 0.790, slope 1.09 and 1.10, risk ranges 21 and 15 points).

**Before thrombectomy, mRS 0–2**

| Specification | AUROC | Log-loss | Brier | Calibration intercept | Calibration slope | Risk range | Coefficients |
|---|---|---|---|---|---|---|---|
| 1.0.0 specification | 0.7611 | 0.5708 | 0.1935 | −0.001 | 1.093 | 20.7 points | 91 |
| 2.1-2.4 | 0.7607 | 0.5687 | 0.1931 | −0.002 | 1.092 | 21.3 points | 58 |
| 1.1 full structure | 0.7614 | 0.5660 | 0.1921 | −0.002 | 1.030 | 21.1 points | 58 |

**Before thrombectomy, death**

| Specification | AUROC | Log-loss | Brier | Calibration intercept | Calibration slope | Risk range | Coefficients |
|---|---|---|---|---|---|---|---|
| 1.0.0 specification | 0.7895 | 0.4347 | 0.1394 | +0.004 | 1.100 | 14.8 points | 91 |
| 2.1-2.4 | 0.7876 | 0.4353 | 0.1397 | +0.005 | 1.082 | 14.7 points | 58 |
| 1.1 full structure | 0.7876 | 0.4353 | 0.1397 | +0.005 | 1.082 | 14.7 points | 58 |

**End of procedure, mRS 0–2**

| Specification | AUROC | Log-loss | Brier | Calibration intercept | Calibration slope | Risk range | Coefficients |
|---|---|---|---|---|---|---|---|
| 1.0.0 specification | 0.7957 | 0.5364 | 0.1798 | +0.000 | 1.070 | 24.6 points | 119 |
| 2.1-2.4 | 0.7981 | 0.5298 | 0.1778 | −0.002 | 1.011 | 24.9 points | 76 |
| 1.1 full structure | 0.7970 | 0.5333 | 0.1790 | −0.002 | 1.071 | 24.7 points | 76 |

**End of procedure, death**

| Specification | AUROC | Log-loss | Brier | Calibration intercept | Calibration slope | Risk range | Coefficients |
|---|---|---|---|---|---|---|---|
| 1.0.0 specification | 0.8068 | 0.4210 | 0.1353 | +0.003 | 1.082 | 16.4 points | 119 |
| 2.1-2.4 | 0.8070 | 0.4210 | 0.1357 | +0.005 | 1.063 | 16.1 points | 76 |
| 1.1 full structure | 0.8071 | 0.4209 | 0.1357 | +0.006 | 1.063 | 16.1 points | 76 |

**24 hours, mRS 0–2**

| Specification | AUROC | Log-loss | Brier | Calibration intercept | Calibration slope | Risk range | Coefficients |
|---|---|---|---|---|---|---|---|
| 1.0.0 specification | 0.9026 | 0.3901 | 0.1244 | −0.001 | 1.034 | 24.0 points | 127 |
| 2.1-2.4 | 0.8995 | 0.3940 | 0.1260 | −0.004 | 1.000 | 27.9 points | 74 |
| 1.1 full structure | 0.8994 | 0.3940 | 0.1260 | −0.004 | 1.000 | 27.3 points | 74 |
| update model | 0.9013 | 0.3941 | 0.1264 | −0.009 | 1.010 | 12.5 points | 5 |

**24 hours, death**

| Specification | AUROC | Log-loss | Brier | Calibration intercept | Calibration slope | Risk range | Coefficients |
|---|---|---|---|---|---|---|---|
| 1.0.0 specification | 0.9174 | 0.2760 | 0.0839 | +0.008 | 1.099 | 12.0 points | 127 |
| 2.1-2.4 | 0.9185 | 0.2741 | 0.0840 | +0.006 | 1.051 | 11.8 points | 74 |
| 1.1 full structure | 0.9174 | 0.2781 | 0.0850 | +0.006 | 1.102 | 11.9 points | 74 |
| update model | 0.9086 | 0.2906 | 0.0885 | +0.002 | 1.037 | 5.5 points | 5 |

**Day 7, mRS 0–2**

| Specification | AUROC | Log-loss | Brier | Calibration intercept | Calibration slope | Risk range | Coefficients |
|---|---|---|---|---|---|---|---|
| 1.0.0 specification | 0.9189 | 0.3599 | 0.1157 | −0.002 | 1.034 | 25.6 points | 135 |
| 2.1-2.4 | 0.9221 | 0.3537 | 0.1131 | −0.007 | 1.042 | 26.4 points | 77 |
| 1.1 full structure | 0.9232 | 0.3487 | 0.1116 | −0.007 | 0.990 | 26.0 points | 77 |
| update model | 0.9260 | 0.3492 | 0.1080 | −0.009 | 1.026 | 13.1 points | 6 |

**Day 7, death**

| Specification | AUROC | Log-loss | Brier | Calibration intercept | Calibration slope | Risk range | Coefficients |
|---|---|---|---|---|---|---|---|
| 1.0.0 specification | 0.9285 | 0.2003 | 0.0579 | +0.014 | 1.038 | 7.1 points | 148 |
| 2.1-2.4 | 0.9269 | 0.2040 | 0.0595 | +0.014 | 1.039 | 7.2 points | 77 |
| 1.1 full structure | 0.9253 | 0.2078 | 0.0605 | +0.014 | 1.085 | 7.1 points | 77 |
| update model | 0.9169 | 0.2125 | 0.0616 | +0.003 | 1.043 | 3.3 points | 6 |

## 4. Decisions (section 6, step 3)

- **Rule (a), before thrombectomy, mRS 0–2:** optimism-corrected AUROC −0.0003, log-loss −0.0021 (2.1–2.4 against the 1.0.0 specification). Kept: True.
- **Rule (a), 24 hours, mRS 0–2:** optimism-corrected AUROC −0.0031, log-loss +0.0039 (2.1–2.4 against the 1.0.0 specification). Kept: True.
- **Changes 2.1–2.4 kept:** True.
- **Rule (b), before thrombectomy, mRS 0–2:** optimism-corrected slope 1.030; inside 0.9–1.1, no correction.
- **Rule (b), before thrombectomy, death:** optimism-corrected slope 1.082; inside 0.9–1.1, no correction.
- **Rule (b), end of procedure, mRS 0–2:** optimism-corrected slope 1.071; inside 0.9–1.1, no correction.
- **Rule (b), end of procedure, death:** optimism-corrected slope 1.063; inside 0.9–1.1, no correction.
- **Rule (b), 24 hours, mRS 0–2:** optimism-corrected slope 1.000; inside 0.9–1.1, no correction.
- **Rule (b), 24 hours, death:** optimism-corrected slope 1.102; slope correction × 1.102 applied.
- **Rule (b), day 7, mRS 0–2:** optimism-corrected slope 0.990; inside 0.9–1.1, no correction.
- **Rule (b), day 7, death:** optimism-corrected slope 1.085; inside 0.9–1.1, no correction.
- **Rule (c), 24 hours, mRS 0–2:** update model against the full structure: nested AUROC +0.0075, log-loss −0.0085. Replaces the full structure: False.
- **Rule (c), day 7, mRS 0–2:** update model (24 h full) against the full structure: nested AUROC +0.0074, log-loss −0.0119. Replaces the full structure: False.
- **Rule (c), 24 hours, death:** update model against the full structure: nested AUROC +0.0015, log-loss −0.0004. Replaces the full structure: False.
- **Rule (c), day 7, death:** update model (24 h full) against the full structure: nested AUROC −0.0009, log-loss −0.0035. Replaces the full structure: False.

## 5. Temporal test, 2023–2025 (not independent)

Version 1.1 was designed after these results were known, so they decide nothing (section 6, step 4). Models fitted on 2005–2022; AUROC with 95% bootstrap interval; the difference is paired (2,000 resamples).

| Stage | Outcome | n (events) | v1.0.0 AUROC | Version 1.1 AUROC | Difference (95% CI) | v1.0.0 intercept / slope | Version 1.1 intercept / slope |
|---|---|---|---|---|---|---|---|
| Before thrombectomy | mRS 0–2 | 226 (60) | 0.809 (0.743–0.867) | 0.816 (0.751–0.873) | +0.007 (−0.006 to +0.021) | −0.41 / 1.47 | −0.42 / 1.47 |
| Before thrombectomy | death | 226 (40) | 0.734 (0.635–0.823) | 0.742 (0.652–0.826) | +0.008 (−0.021 to +0.042) | −0.39 / 0.94 | −0.38 / 0.98 |
| End of procedure | mRS 0–2 | 221 (59) | 0.842 (0.785–0.895) | 0.845 (0.789–0.896) | +0.002 (−0.009 to +0.014) | −0.42 / 1.45 | −0.43 / 1.46 |
| End of procedure | death | 221 (38) | 0.746 (0.654–0.833) | 0.737 (0.641–0.828) | −0.008 (−0.045 to +0.032) | −0.40 / 0.96 | −0.39 / 0.88 |
| 24 hours | mRS 0–2 | 120 (36) | 0.883 (0.818–0.936) | 0.907 (0.849–0.954) | +0.024 (−0.005 to +0.056) | −0.69 / 1.05 | −0.55 / 1.43 |
| 24 hours | death | 120 (13) | 0.859 (0.726–0.952) | 0.836 (0.705–0.938) | −0.023 (−0.065 to +0.008) | −0.59 / 0.90 | −0.68 / 0.71 |
| Day 7 | mRS 0–2 | 139 (52) | 0.884 (0.826–0.934) | 0.920 (0.876–0.958) | +0.036 (+0.013 to +0.062) | −0.63 / 1.16 | −0.63 / 1.65 |
| Day 7 | death | 139 (6) | 0.805 (0.574–0.979) | 0.813 (0.546–0.973) | +0.008 (−0.064 to +0.100) | −1.69 / 0.56 | −1.93 / 0.55 |

## 6. The selection rule of amendment 1, applied again

Version 1.1 is the reference. Temporal criterion: Holm over the seven candidates (paired bootstrap of the AUROC difference, 2,000 resamples). Nested criterion (all years, before thrombectomy, mRS 0–2, v1.0.0's folds): version 1.1 mean AUROC 0.7682; v1.0.0 reproduced on those folds within 5.0e-05.

A candidate qualifies on the temporal test only with a higher AUROC and a Holm-adjusted p below 0.05. The candidates' 2005–2022 fits were regenerated with the v1.0.0 code; their 2023–2025 AUROCs equal the published ones within 0.0001, except stacking for death at 24 hours (0.852 against 0.854).

| Stage | Outcome | Chosen | Best candidate: AUROC difference, Holm-adjusted p |
|---|---|---|---|
| Before thrombectomy | mRS 0–2 | Regression with splines, version 1.1 | Stacking: −0.007, p = 1.000 |
| Before thrombectomy | death | Regression with splines, version 1.1 | XGBoost: +0.019, p = 1.000 |
| End of procedure | mRS 0–2 | Regression with splines, version 1.1 | LR + interactions: −0.004, p = 1.000 |
| End of procedure | death | Regression with splines, version 1.1 | Stacking: +0.041, p = 0.238 |
| 24 hours | mRS 0–2 | Regression with splines, version 1.1 | LR + multiple imputation: +0.000, p = 1.000 |
| 24 hours | death | Regression with splines, version 1.1 | LR + interactions: +0.028, p = 1.000 |
| Day 7 | mRS 0–2 | Regression with splines, version 1.1 | Stacking: +0.003, p = 1.000 |
| Day 7 | death | Regression with splines, version 1.1 | XGBoost: +0.038, p = 1.000 |

## 7. Deployed version 1.1

Fingerprint `841d810c28097e00d32ea2e0de6ea6f6b432f235a5a8e228ed7021b955556296`. Knots and penalty from the 2005–2022 tuning, refitted on all 1,451 patients, intercept re-estimated on 2023–2025 from out-of-fold estimates. The other learners are carried over from v1.0.0 unchanged; v1.0.0's reference stays in the bundle's archive and in the app.

| Stage | Outcome | Knots | Penalty C | Slope factor | Coefficients (v1.0.0 → 1.1) | Recent-era intercept |
|---|---|---|---|---|---|---|
| Before thrombectomy | mRS 0–2 | 3 | 0.0160 | 1.000 | 91 → 59 | −0.294 |
| Before thrombectomy | death | 3 | 0.0121 | 1.000 | 91 → 59 | −0.224 |
| End of procedure | mRS 0–2 | 3 | 0.0160 | 1.000 | 119 → 77 | −0.286 |
| End of procedure | death | 3 | 0.0121 | 1.000 | 119 → 77 | −0.261 |
| 24 hours | mRS 0–2 | 3 | 0.0655 | 1.000 | 127 → 76 | −0.348 |
| 24 hours | death | 3 | 0.0281 | 1.102 | 127 → 76 | −0.489 |
| Day 7 | mRS 0–2 | 3 | 0.0869 | 1.000 | 135 → 82 | −0.295 |
| Day 7 | death | 3 | 0.0281 | 1.000 | 148 → 82 | −1.063 |

