# MTOP ordinal model: results of the registered evaluation

Written 2026-11-22 by `python -m mtop.ordinal_run report` from the analysis outputs. Plan: [ordinal_plan.md](ordinal_plan.md) (22 November 2026) with amendment 1, both committed before any ordinal-model result was computed. The development years 2005–2022 decide (sections 1–6); the decisions were committed before any 2023–2025 or 2021–2025 result of the ordinal model or of the comparison-only learner was computed.

Model: a penalised proportional-odds regression on the seven 90-day mRS grades with the inputs of the v1.1 reference at each stage. Comparators within the development years: the v1.1 proportional-odds layer (which splits the binary models' bands into grades), the v1.1 bands split by the ordinal model's shares (rule O1's alternative), the v1.1 binary models (for the implied mRS 0–2 and death), and an ordinal LightGBM (comparison only).

## Decisions (section 7)

- **O1, the calculator's distribution: the ordinal model replaces the layer.** Nested cross-validated ranked probability score (RPS, lower is better) of the v1.1 bands split by the ordinal model's shares against the layer's: before thrombectomy 0.1661 against 0.1663; 24 hours 0.1073 against 0.1108. The rule needs a lower score at both primary stages.
- **O2, external validation: sent as a secondary model.** Pipeline check passed; nested cross-validated AUROC of the implied P(mRS 0–2) against the v1.1 binary model: before thrombectomy 0.760 against 0.753 (+0.007); 24 hours 0.906 against 0.893 (+0.012). The model is not sent if it is more than 0.02 lower at a primary stage.
- **O3, slope correction: applied at day 7 (× 0.887).** Optimism-corrected ordinal calibration slope: before thrombectomy 1.083; end of procedure 1.087; 24 hours 0.906; day 7 0.887 (applied only outside 0.9–1.1).
- **O4, proportional odds:** descriptive, section 4.

## 1. Pipeline check (section 5, step 3)

- Registry export SHA-256 `1a19f809ed688a3940c38ed4d0b1198f0a978b806f9ed7b50721777d980d50f7`: identical to v1.1.
- Development and test patients at each stage: before thrombectomy 1225 and 226; end of procedure 1210 and 221; 24 hours 1115 and 120; day 7 1000 and 139 (identical to v1.1).
- Unpenalised ordinal model (age and admission NIHSS) against statsmodels: largest difference 1.4e-09 (criterion 10⁻⁶).
- Sparse grades: every grade has at least 20 development patients at every stage; no grade is merged.
- **Passed: yes.**

## 2. Nested cross-validation, 2005–2022 (5 outer folds × 3 repeats)

Mean over the 15 outer folds (2.5th to 97.5th percentile of the folds in brackets, a spread, not a confidence interval). RPS: ranked probability score; log-loss: ordinal log-loss; C (utility): generalised concordance on the expected utility-weighted mRS; C (grade): on the expected grade; AUROC for the implied dichotomies; slope: calibration slope of the implied P(mRS 0–2); utility: mean predicted minus observed utility-weighted mRS.

### Before thrombectomy (1225 patients)

| Model | RPS | Log-loss | C (utility) | C (grade) | AUROC 0–1 | AUROC 0–2 | AUROC 0–3 | AUROC death | Slope 0–2 | Utility |
|---|---|---|---|---|---|---|---|---|---|---|
| Ordinal regression | 0.1651 | 1.764 | 0.711 | 0.711 | 0.759 | 0.760 | 0.765 | 0.768 | 1.088 | −0.001 |
| v1.1 layer | 0.1663 | 1.771 | 0.709 | 0.708 | 0.757 | 0.753 | 0.762 | 0.774 | 1.057 | −0.001 |
| v1.1 bands with ordinal shares | 0.1661 | 1.770 | 0.710 | 0.709 | 0.757 | 0.753 | 0.763 | 0.774 | 1.057 | −0.002 |
| v1.1 binary models | – | – | – | – | – | 0.753 | – | 0.774 | 1.057 | – |

Paired differences, fold by fold:

| Comparison | RPS | Log-loss | C (utility) | AUROC 0–2 | AUROC death |
|---|---|---|---|---|---|
| Ordinal regression − v1.1 layer | −0.0012 (−0.0030 to +0.0011) | −0.007 (−0.037 to +0.023) | +0.003 (−0.006 to +0.011) | +0.007 (−0.012 to +0.022) | −0.006 (−0.028 to +0.028) |
| v1.1 bands with ordinal shares − v1.1 layer | −0.0002 (−0.0007 to +0.0002) | −0.001 (−0.006 to +0.003) | +0.001 (−0.001 to +0.003) | +0.000 (+0.000 to +0.000) | +0.000 (+0.000 to +0.000) |
| Ordinal regression − v1.1 bands with ordinal shares | −0.0010 (−0.0027 to +0.0010) | −0.006 (−0.035 to +0.022) | +0.002 (−0.006 to +0.008) | +0.007 (−0.012 to +0.022) | −0.006 (−0.028 to +0.028) |
| Ordinal regression − v1.1 binary models | – | – | – | +0.007 (−0.012 to +0.023) | −0.006 (−0.028 to +0.029) |
| v1.1 bands with ordinal shares − v1.1 binary models | – | – | – | +0.000 (+0.000 to +0.000) | +0.000 (+0.000 to +0.000) |
| v1.1 layer − v1.1 binary models | – | – | – | +0.000 (+0.000 to +0.000) | +0.000 (+0.000 to +0.000) |

Ordinal model across folds: knots 3/4/5 in 100 %, 0 %, 0 % of folds; median log10 C −1.92; penalty at the end of the grid in 0 %; median 54 coefficients. Patients whose observed grade had a probability below 10⁻⁶: Ordinal regression 0, v1.1 layer 0, v1.1 bands with ordinal shares 0.

### End of procedure (1210 patients)

| Model | RPS | Log-loss | C (utility) | C (grade) | AUROC 0–1 | AUROC 0–2 | AUROC 0–3 | AUROC death | Slope 0–2 | Utility |
|---|---|---|---|---|---|---|---|---|---|---|
| Ordinal regression | 0.1550 | 1.722 | 0.741 | 0.741 | 0.788 | 0.793 | 0.813 | 0.793 | 1.098 | −0.001 |
| v1.1 layer | 0.1556 | 1.724 | 0.739 | 0.739 | 0.785 | 0.789 | 0.810 | 0.794 | 1.060 | −0.003 |
| v1.1 bands with ordinal shares | 0.1555 | 1.724 | 0.740 | 0.740 | 0.786 | 0.789 | 0.812 | 0.794 | 1.060 | −0.003 |
| Ordinal LightGBM (comparison only) | 0.1549 | 2.600 | 0.743 | 0.745 | 0.777 | 0.785 | 0.815 | 0.807 | 1.073 | −0.000 |
| v1.1 binary models | – | – | – | – | – | 0.789 | – | 0.794 | 1.060 | – |

Paired differences, fold by fold:

| Comparison | RPS | Log-loss | C (utility) | AUROC 0–2 | AUROC death |
|---|---|---|---|---|---|
| Ordinal regression − v1.1 layer | −0.0006 (−0.0040 to +0.0024) | −0.002 (−0.021 to +0.028) | +0.002 (−0.006 to +0.015) | +0.004 (−0.010 to +0.013) | −0.002 (−0.024 to +0.020) |
| v1.1 bands with ordinal shares − v1.1 layer | −0.0001 (−0.0004 to +0.0003) | −0.000 (−0.005 to +0.003) | +0.001 (−0.000 to +0.003) | +0.000 (+0.000 to +0.000) | +0.000 (+0.000 to +0.000) |
| Ordinal regression − v1.1 bands with ordinal shares | −0.0005 (−0.0036 to +0.0023) | −0.002 (−0.020 to +0.030) | +0.001 (−0.006 to +0.012) | +0.004 (−0.010 to +0.013) | −0.002 (−0.024 to +0.020) |
| Ordinal LightGBM (comparison only) − Ordinal regression | −0.0001 (−0.0050 to +0.0051) | +0.878 (+0.357 to +1.633) | +0.002 (−0.014 to +0.016) | −0.008 (−0.035 to +0.009) | +0.015 (−0.017 to +0.051) |
| Ordinal regression − v1.1 binary models | – | – | – | +0.004 (−0.010 to +0.013) | −0.002 (−0.024 to +0.021) |
| v1.1 bands with ordinal shares − v1.1 binary models | – | – | – | −0.000 (−0.000 to +0.000) | +0.000 (+0.000 to +0.000) |
| v1.1 layer − v1.1 binary models | – | – | – | −0.000 (−0.000 to +0.000) | +0.000 (+0.000 to +0.000) |

Ordinal model across folds: knots 3/4/5 in 100 %, 0 %, 0 % of folds; median log10 C −1.92; penalty at the end of the grid in 0 %; median 71 coefficients. Patients whose observed grade had a probability below 10⁻⁶: Ordinal regression 0, v1.1 layer 0, v1.1 bands with ordinal shares 0, Ordinal LightGBM (comparison only) 90.

### 24 hours (1115 patients)

| Model | RPS | Log-loss | C (utility) | C (grade) | AUROC 0–1 | AUROC 0–2 | AUROC 0–3 | AUROC death | Slope 0–2 | Utility |
|---|---|---|---|---|---|---|---|---|---|---|
| Ordinal regression | 0.1040 | 1.427 | 0.853 | 0.853 | 0.898 | 0.906 | 0.918 | 0.899 | 0.940 | +0.006 |
| v1.1 layer | 0.1108 | 1.467 | 0.839 | 0.839 | 0.877 | 0.893 | 0.914 | 0.905 | 1.061 | −0.001 |
| v1.1 bands with ordinal shares | 0.1073 | 1.434 | 0.846 | 0.846 | 0.889 | 0.893 | 0.918 | 0.905 | 1.061 | −0.000 |
| Ordinal LightGBM (comparison only) | 0.1024 | 1.646 | 0.854 | 0.857 | 0.901 | 0.904 | 0.920 | 0.914 | 1.073 | +0.001 |
| v1.1 binary models | – | – | – | – | – | 0.893 | – | 0.905 | 1.059 | – |

Paired differences, fold by fold:

| Comparison | RPS | Log-loss | C (utility) | AUROC 0–2 | AUROC death |
|---|---|---|---|---|---|
| Ordinal regression − v1.1 layer | −0.0069 (−0.0119 to −0.0029) | −0.040 (−0.079 to −0.007) | +0.014 (+0.005 to +0.022) | +0.012 (−0.002 to +0.028) | −0.005 (−0.022 to +0.009) |
| v1.1 bands with ordinal shares − v1.1 layer | −0.0036 (−0.0044 to −0.0023) | −0.033 (−0.047 to −0.020) | +0.007 (+0.004 to +0.009) | +0.000 (+0.000 to +0.000) | +0.000 (+0.000 to +0.000) |
| Ordinal regression − v1.1 bands with ordinal shares | −0.0033 (−0.0076 to −0.0002) | −0.008 (−0.049 to +0.040) | +0.007 (+0.000 to +0.013) | +0.012 (−0.002 to +0.028) | −0.005 (−0.022 to +0.009) |
| Ordinal LightGBM (comparison only) − Ordinal regression | −0.0016 (−0.0056 to +0.0020) | +0.219 (−0.027 to +0.420) | +0.001 (−0.008 to +0.009) | −0.002 (−0.014 to +0.008) | +0.015 (−0.005 to +0.033) |
| Ordinal regression − v1.1 binary models | – | – | – | +0.012 (−0.001 to +0.028) | −0.005 (−0.022 to +0.009) |
| v1.1 bands with ordinal shares − v1.1 binary models | – | – | – | +0.000 (−0.000 to +0.000) | +0.000 (−0.000 to +0.000) |
| v1.1 layer − v1.1 binary models | – | – | – | +0.000 (−0.000 to +0.000) | +0.000 (−0.000 to +0.000) |

Ordinal model across folds: knots 3/4/5 in 67 %, 13 %, 20 % of folds; median log10 C −0.45; penalty at the end of the grid in 0 %; median 72 coefficients. Patients whose observed grade had a probability below 10⁻⁶: Ordinal regression 0, v1.1 layer 0, v1.1 bands with ordinal shares 0, Ordinal LightGBM (comparison only) 25.

### Day 7 (1000 patients)

| Model | RPS | Log-loss | C (utility) | C (grade) | AUROC 0–1 | AUROC 0–2 | AUROC 0–3 | AUROC death | Slope 0–2 | Utility |
|---|---|---|---|---|---|---|---|---|---|---|
| Ordinal regression | 0.0897 | 1.340 | 0.876 | 0.876 | 0.928 | 0.928 | 0.935 | 0.909 | 0.918 | +0.009 |
| v1.1 layer | 0.0991 | 1.412 | 0.855 | 0.855 | 0.900 | 0.914 | 0.931 | 0.914 | 1.036 | −0.002 |
| v1.1 bands with ordinal shares | 0.0927 | 1.352 | 0.870 | 0.870 | 0.920 | 0.914 | 0.937 | 0.914 | 1.036 | −0.001 |
| Ordinal LightGBM (comparison only) | 0.0883 | 1.675 | 0.877 | 0.879 | 0.926 | 0.932 | 0.939 | 0.905 | 1.088 | +0.002 |
| v1.1 binary models | – | – | – | – | – | 0.914 | – | 0.914 | 1.031 | – |

Paired differences, fold by fold:

| Comparison | RPS | Log-loss | C (utility) | AUROC 0–2 | AUROC death |
|---|---|---|---|---|---|
| Ordinal regression − v1.1 layer | −0.0093 (−0.0136 to −0.0046) | −0.072 (−0.132 to −0.030) | +0.021 (+0.012 to +0.028) | +0.014 (+0.005 to +0.022) | −0.005 (−0.027 to +0.013) |
| v1.1 bands with ordinal shares − v1.1 layer | −0.0064 (−0.0083 to −0.0045) | −0.059 (−0.080 to −0.033) | +0.014 (+0.009 to +0.021) | +0.000 (+0.000 to +0.000) | +0.000 (+0.000 to +0.000) |
| Ordinal regression − v1.1 bands with ordinal shares | −0.0029 (−0.0059 to +0.0003) | −0.013 (−0.058 to +0.020) | +0.006 (+0.002 to +0.011) | +0.014 (+0.005 to +0.022) | −0.005 (−0.027 to +0.013) |
| Ordinal LightGBM (comparison only) − Ordinal regression | −0.0014 (−0.0066 to +0.0019) | +0.335 (−0.042 to +0.760) | +0.001 (−0.006 to +0.010) | +0.003 (−0.009 to +0.016) | −0.004 (−0.031 to +0.020) |
| Ordinal regression − v1.1 binary models | – | – | – | +0.014 (+0.005 to +0.022) | −0.005 (−0.027 to +0.013) |
| v1.1 bands with ordinal shares − v1.1 binary models | – | – | – | −0.000 (−0.000 to +0.000) | +0.000 (+0.000 to +0.000) |
| v1.1 layer − v1.1 binary models | – | – | – | −0.000 (−0.000 to +0.000) | +0.000 (+0.000 to +0.000) |

Ordinal model across folds: knots 3/4/5 in 100 %, 0 %, 0 % of folds; median log10 C 0.29; penalty at the end of the grid in 0 %; median 75 coefficients. Patients whose observed grade had a probability below 10⁻⁶: Ordinal regression 0, v1.1 layer 0, v1.1 bands with ordinal shares 0, Ordinal LightGBM (comparison only) 30.

### Consistency with the binary models

Implied P(mRS 0–2) and P(death) of the ordinal model against the v1.1 binary models on the same folds (mean over folds).

| Stage | Outcome | AUROC ordinal | AUROC binary | Log-loss ordinal | Log-loss binary | CITL ordinal | CITL binary | Slope ordinal | Slope binary |
|---|---|---|---|---|---|---|---|---|---|
| Before thrombectomy | mRS 0–2 | 0.760 | 0.753 | 0.5719 | 0.5757 | +0.017 | +0.003 | 1.088 | 1.057 |
| Before thrombectomy | death | 0.768 | 0.774 | 0.4439 | 0.4458 | +0.025 | −0.001 | 1.026 | 1.100 |
| End of procedure | mRS 0–2 | 0.793 | 0.789 | 0.5438 | 0.5429 | +0.033 | +0.011 | 1.098 | 1.060 |
| End of procedure | death | 0.793 | 0.794 | 0.4313 | 0.4328 | +0.047 | −0.012 | 0.955 | 1.100 |
| 24 hours | mRS 0–2 | 0.906 | 0.893 | 0.3852 | 0.4069 | −0.042 | +0.007 | 0.940 | 1.059 |
| 24 hours | death | 0.899 | 0.905 | 0.3157 | 0.2980 | +0.041 | −0.014 | 0.745 | 0.971 |
| Day 7 | mRS 0–2 | 0.928 | 0.914 | 0.3405 | 0.3700 | −0.106 | −0.007 | 0.918 | 1.031 |
| Day 7 | death | 0.909 | 0.914 | 0.2481 | 0.2242 | +0.050 | −0.001 | 0.667 | 1.087 |

## 3. Full-pipeline bootstrap, 2005–2022 (500 resamples, seed 2026)

Apparent, optimism and optimism-corrected (Harrell) measures. Slope (ordinal): ordinal calibration slope.

| Stage | Model | RPS | Log-loss | C (utility) | AUROC 0–2 | AUROC death | Slope (ordinal) | Slope 0–2 | Slope death |
|---|---|---|---|---|---|---|---|---|---|
| Before thrombectomy | Ordinal regression | 0.1635 | 1.756 | 0.716 | 0.765 | 0.772 | 1.083 | 1.087 | 1.018 |
| End of procedure | Ordinal regression | 0.1531 | 1.711 | 0.745 | 0.798 | 0.799 | 1.087 | 1.124 | 0.977 |
| End of procedure | Ordinal LightGBM (comparison only) | 0.1502 | 2.176 | 0.750 | 0.803 | 0.836 | – | 1.093 | 1.097 |
| 24 hours | Ordinal regression | 0.1020 | 1.401 | 0.855 | 0.905 | 0.902 | 0.906 | 0.853 | 0.739 |
| 24 hours | Ordinal LightGBM (comparison only) | 0.0989 | 1.490 | 0.858 | 0.908 | 0.925 | – | 1.020 | 1.082 |
| Day 7 | Ordinal regression | 0.0882 | 1.319 | 0.878 | 0.928 | 0.908 | 0.887 | 0.855 | 0.661 |
| Day 7 | Ordinal LightGBM (comparison only) | 0.0803 | 1.358 | 0.885 | 0.943 | 0.942 | – | 0.961 | 0.971 |

Stability across the resamples: median (90th percentile) width of each patient's 95 % range.

| Stage | Model | P(mRS 0–2) | P(death) | Expected utility | Knots 3/4/5 | Median log10 C |
|---|---|---|---|---|---|---|
| Before thrombectomy | Ordinal regression | 0.182 (0.264) | 0.122 (0.254) | 0.163 (0.242) | 98 % / 2 % / 0 % | −2.16 |
| End of procedure | Ordinal regression | 0.203 (0.290) | 0.134 (0.280) | 0.182 (0.262) | 99 % / 1 % / 0 % | −2.16 |
| End of procedure | Ordinal LightGBM (comparison only) | 0.298 (0.388) | 0.200 (0.430) | 0.213 (0.287) | – | – |
| 24 hours | Ordinal regression | 0.229 (0.388) | 0.057 (0.367) | 0.187 (0.291) | 75 % / 4 % / 21 % | −1.43 |
| 24 hours | Ordinal LightGBM (comparison only) | 0.184 (0.378) | 0.074 (0.414) | 0.128 (0.228) | – | – |
| Day 7 | Ordinal regression | 0.209 (0.452) | 0.017 (0.390) | 0.175 (0.306) | 56 % / 31 % / 13 % | −1.06 |
| Day 7 | Ordinal LightGBM (comparison only) | 0.129 (0.466) | 0.046 (0.462) | 0.125 (0.252) | – | – |

## 4. Proportional odds (rule O4, descriptive)

A penalised binary model of mRS > k at each cut-point, with the development model's layout and penalty. Slope and correlation of its linear predictor on the shared one: both 1 under proportional odds, apart from the penalty.

| Stage | > 0 | > 1 | > 2 | > 3 | > 4 | > 5 |
|---|---|---|---|---|---|---|
| Before thrombectomy | 0.75 (0.91) | 0.85 (0.97) | 0.88 (0.98) | 0.91 (0.98) | 0.93 (0.98) | 0.83 (0.94) |
| End of procedure | 0.73 (0.91) | 0.84 (0.97) | 0.89 (0.97) | 0.96 (0.98) | 0.96 (0.98) | 0.81 (0.94) |
| 24 hours | 1.07 (0.93) | 0.91 (0.97) | 1.03 (0.98) | 1.06 (0.97) | 1.05 (0.96) | 0.89 (0.86) |
| Day 7 | 1.19 (0.93) | 1.47 (0.96) | 1.23 (0.97) | 1.21 (0.95) | 1.06 (0.91) | 0.77 (0.73) |

Calibration of the implied dichotomies (nested cross-validation, mean slope over folds): before thrombectomy 0–1 1.08, 0–2 1.09, 0–3 1.09, death 1.03; end of procedure 0–1 1.09, 0–2 1.10, 0–3 1.20, death 0.96; 24 hours 0–1 0.94, 0–2 0.94, 0–3 0.96, death 0.74; day 7 0–1 1.01, 0–2 0.92, 0–3 0.90, death 0.67.

## 5. Sample size (Riley et al., descriptive)

Required patients at each cut-point for the development model's number of coefficients, with the conservative anticipated R² (15 % of the maximum); in brackets, with the R² implied by v1.1's nested AUROC (mRS 0–2 and death).

| Stage | Patients | Coefficients | > 0 | > 1 | > 2 | > 3 | > 4 | > 5 |
|---|---|---|---|---|---|---|---|---|
| Before thrombectomy | 1225 | 58 | 6028 | 4655 | 4405 (2539) | 4352 | 4508 | 5050 (2867) |
| End of procedure | 1210 | 76 | 7918 | 6101 | 5774 (2497) | 5702 | 5904 | 6623 (3244) |
| 24 hours | 1115 | 74 | 7524 | 5878 | 5592 (1484) | 5569 | 5813 | 6655 (1878) |
| Day 7 | 1000 | 77 | 7452 | 5989 | 5776 (1469) | 5882 | 6374 | 8193 (2397) |

Events per coefficient at the scarcest side of each cut-point: before thrombectomy 2.9 / 6.2 / 8.5 / 9.8 / 7.3 / 4.6; end of procedure 2.2 / 4.7 / 6.4 / 7.4 / 5.5 / 3.5; 24 hours 2.2 / 4.7 / 6.4 / 6.7 / 4.9 / 3.0; day 7 2.1 / 4.5 / 6.1 / 5.0 / 3.4 / 1.7 (cut-points > 0 to > 5).

## 6. Development-year models (2005–2022)

| Stage | Patients | Knots | C | Coefficients | Thresholds θ₀…θ₅ |
|---|---|---|---|---|---|
| Before thrombectomy | 1225 | 3 | 0.009103 | 58 | −2.23, −1.10, −0.53, 0.15, 0.77, 1.56 |
| End of procedure | 1210 | 3 | 0.009103 | 76 | −2.35, −1.18, −0.57, 0.14, 0.83, 1.67 |
| 24 hours | 1115 | 3 | 0.6251 | 74 | −3.63, −1.93, −0.92, 0.36, 1.56, 3.03 |
| Day 7 | 1000 | 3 | 2.56 | 77 | −3.85, −1.85, −0.58, 1.08, 2.65, 4.60 |

