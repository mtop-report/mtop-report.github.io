# MTOP ordinal model: results of the registered evaluation

Written 2026-11-22 by `python -m mtop.ordinal_run report` from the analysis outputs. Plan: [ordinal_plan.md](ordinal_plan.md) (22 November 2026) with amendment 1, both committed before any ordinal-model result was computed. The development years 2005–2022 decide (sections 1–6); the decisions were committed before any 2023–2025 or 2021–2025 result of the ordinal model or of the comparison-only learner was computed.

Model: a penalised proportional-odds regression on the seven 90-day mRS grades with the inputs of the v1.1 reference at each stage. Comparators within the development years: the v1.1 proportional-odds layer (which splits the binary models' bands into grades), the v1.1 bands split by the ordinal model's shares (rule O1's alternative), the v1.1 binary models (for the implied mRS 0–2 and death), and an ordinal LightGBM (comparison only).

## Decisions (section 7)

- **O1, the calculator's distribution: the ordinal model replaces the layer.** Nested cross-validated ranked probability score (RPS, lower is better) of the v1.1 bands split by the ordinal model's shares against the layer's: before thrombectomy 0.1661 against 0.1663; 24 hours 0.1073 against 0.1108. The rule needs a lower score at both primary stages.
- **O2, external validation: sent as a secondary model.** Pipeline check passed; nested cross-validated AUROC of the implied P(mRS 0–2) against the v1.1 binary model: before thrombectomy 0.760 against 0.753 (+0.007); 24 hours 0.906 against 0.893 (+0.012). The model is not sent if it is more than 0.02 lower at a primary stage.
- **O3, slope correction: applied at day 7 (× 0.887).** Optimism-corrected ordinal calibration slope: before thrombectomy 1.083; end of procedure 1.087; 24 hours 0.906; day 7 0.887 (applied only outside 0.9–1.1).
- **O4, proportional odds:** descriptive, section 4.

## Summary of the tests on later patients (reported, not deciding)

- **Bands with ordinal shares against the layer, 2023–2025** (RPS): before thrombectomy 0.1480 against 0.1479; end of procedure 0.1406 against 0.1405; 24 hours 0.1010 against 0.1037; day 7 0.0974 against 0.1033.
- **Ordinal regression on its own against the layer, 2023–2025** (RPS): before thrombectomy 0.1491 against 0.1479; end of procedure 0.1419 against 0.1405; 24 hours 0.1060 against 0.1037; day 7 0.1018 against 0.1033.
- **Calibration-in-the-large of mRS 0–2, 2023–2025** (before the recent-era shift; negative: independence over-predicted): ordinal regression −0.52, −0.55, −1.01, −1.28; v1.1 binary models −0.42, −0.43, −0.55, −0.63 (before thrombectomy, end of procedure, 24 hours, day 7). The deployed model's recent-era threshold shift reduces this for mRS 0–2 but over-predicts death (section 8).
- **Comparison-only LightGBM (section 9):** lower RPS than the ordinal regression in both the nested cross-validation and the 2023–2025 test at end of procedure. The plan notes this as a reason to pre-specify it as a candidate in a later version; it is not deployed and not sent to external validation.

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

## 7. Temporal and sensitivity tests (section 8, reported, not deciding)

The development-year models tested on later patients. **Not independent:** the specification is that of v1.1, which was designed with the 2023–2025 results known. 95 % intervals from 2,000 bootstrap resamples of the test patients; differences against the ordinal regression are paired and unadjusted. Rule O3's factor is used in both splits.

### Primary split: development before 2023, test 2023–2025

**Before thrombectomy** (1225 development, 226 test patients; test grades 0–6: 23, 23, 14, 29, 41, 56, 40)

| Model | RPS | Log-loss | C (utility) | AUROC 0–2 | AUROC death | CITL 0–2 | Slope 0–2 | Slope (ordinal) | Utility |
|---|---|---|---|---|---|---|---|---|---|
| Ordinal regression | 0.1491 (0.1373 to 0.1611) | 1.766 (1.701 to 1.827) | 0.727 (0.684 to 0.769) | 0.815 (0.748 to 0.872) | 0.724 (0.635 to 0.811) | −0.518 (−0.817 to −0.230) | 1.592 (1.132 to 2.212) | 1.218 (0.938 to 1.569) | +0.070 |
| v1.1 layer | 0.1479 (0.1355 to 0.1605) | 1.728 (1.661 to 1.792) | 0.720 (0.676 to 0.765) | 0.816 (0.751 to 0.873) | 0.742 (0.652 to 0.826) | −0.423 (−0.736 to −0.135) | 1.473 (1.083 to 2.033) | – | +0.056 |
| v1.1 bands with ordinal shares | 0.1480 (0.1357 to 0.1604) | 1.730 (1.664 to 1.793) | 0.721 (0.677 to 0.765) | 0.816 (0.751 to 0.873) | 0.742 (0.652 to 0.826) | −0.423 (−0.736 to −0.135) | 1.473 (1.083 to 2.033) | – | +0.059 |
| MR PREDICTS predictors, proportional odds (no collaterals) | 0.1515 (0.1396 to 0.1646) | 1.779 (1.709 to 1.846) | 0.725 (0.679 to 0.766) | 0.792 (0.720 to 0.855) | 0.732 (0.638 to 0.817) | −0.630 (−0.940 to −0.339) | 1.310 (0.923 to 1.818) | 1.133 (0.873 to 1.445) | +0.084 |
| v1.1 binary models | – | – | – | 0.816 (0.751 to 0.873) | 0.742 (0.652 to 0.826) | −0.423 (−0.736 to −0.135) | 1.473 (1.083 to 2.033) | – | – |

Differences against the ordinal regression (other model minus ordinal; RPS and log-loss: negative favours the other model):

| Comparison | RPS | C (utility) | AUROC 0–2 | AUROC death |
|---|---|---|---|---|
| v1.1 layer − Ordinal regression | −0.0012 (−0.0038 to +0.0014) | −0.006 (−0.015 to +0.002) | +0.001 (−0.019 to +0.023) | +0.017 (−0.014 to +0.050) |
| v1.1 bands with ordinal shares − Ordinal regression | −0.0012 (−0.0036 to +0.0012) | −0.006 (−0.013 to +0.001) | +0.001 (−0.019 to +0.023) | +0.017 (−0.014 to +0.050) |
| MR PREDICTS predictors, proportional odds (no collaterals) − Ordinal regression | +0.0024 (−0.0030 to +0.0082) | −0.002 (−0.025 to +0.020) | −0.023 (−0.058 to +0.009) | +0.007 (−0.036 to +0.047) |
| v1.1 binary models − Ordinal regression | – | – | +0.001 (−0.019 to +0.023) | +0.017 (−0.014 to +0.050) |

Subgroups (point estimates; RPS / C (utility) of the ordinal regression and the layer): men (n 108): 0.1452 / 0.778 and 0.1438 / 0.779; women (n 118): 0.1528 / 0.686 and 0.1516 / 0.677; age under 80 (n 174): 0.1536 / 0.730 and 0.1525 / 0.728.
Not shown (fewer than 10 patients in mRS 0–2, 3–5 or 6): age 80 or older.

**End of procedure** (1210 development, 221 test patients; test grades 0–6: 22, 23, 14, 28, 40, 56, 38)

| Model | RPS | Log-loss | C (utility) | AUROC 0–2 | AUROC death | CITL 0–2 | Slope 0–2 | Slope (ordinal) | Utility |
|---|---|---|---|---|---|---|---|---|---|
| Ordinal regression | 0.1419 (0.1300 to 0.1552) | 1.731 (1.663 to 1.800) | 0.747 (0.703 to 0.790) | 0.840 (0.779 to 0.895) | 0.737 (0.645 to 0.824) | −0.549 (−0.873 to −0.258) | 1.507 (1.100 to 2.109) | 1.161 (0.893 to 1.494) | +0.076 |
| v1.1 layer | 0.1405 (0.1280 to 0.1538) | 1.701 (1.627 to 1.774) | 0.744 (0.700 to 0.788) | 0.845 (0.789 to 0.896) | 0.737 (0.641 to 0.828) | −0.433 (−0.773 to −0.138) | 1.463 (1.083 to 2.014) | – | +0.058 |
| v1.1 bands with ordinal shares | 0.1406 (0.1282 to 0.1537) | 1.702 (1.629 to 1.774) | 0.745 (0.700 to 0.789) | 0.845 (0.789 to 0.896) | 0.737 (0.641 to 0.828) | −0.433 (−0.773 to −0.138) | 1.463 (1.083 to 2.014) | – | +0.061 |
| MR PREDICTS predictors, proportional odds (no collaterals) | 0.1499 (0.1378 to 0.1630) | 1.775 (1.705 to 1.845) | 0.728 (0.681 to 0.771) | 0.793 (0.716 to 0.862) | 0.733 (0.631 to 0.824) | −0.620 (−0.944 to −0.324) | 1.309 (0.914 to 1.864) | 1.150 (0.882 to 1.466) | +0.084 |
| Ordinal LightGBM (comparison only) | 0.1383 (0.1257 to 0.1514) | 2.773 (2.042 to 3.664) | 0.757 (0.714 to 0.800) | 0.830 (0.771 to 0.885) | 0.773 (0.682 to 0.857) | −0.530 (−0.865 to −0.226) | 1.381 (1.030 to 1.859) | – | +0.082 |
| v1.1 binary models | – | – | – | 0.845 (0.789 to 0.896) | 0.737 (0.641 to 0.828) | −0.433 (−0.773 to −0.138) | 1.463 (1.083 to 2.014) | – | – |

Differences against the ordinal regression (other model minus ordinal; RPS and log-loss: negative favours the other model):

| Comparison | RPS | C (utility) | AUROC 0–2 | AUROC death |
|---|---|---|---|---|
| v1.1 layer − Ordinal regression | −0.0014 (−0.0042 to +0.0012) | −0.004 (−0.012 to +0.005) | +0.004 (−0.016 to +0.025) | −0.000 (−0.032 to +0.031) |
| v1.1 bands with ordinal shares − Ordinal regression | −0.0014 (−0.0039 to +0.0012) | −0.003 (−0.010 to +0.005) | +0.004 (−0.016 to +0.025) | −0.000 (−0.032 to +0.031) |
| MR PREDICTS predictors, proportional odds (no collaterals) − Ordinal regression | +0.0080 (+0.0005 to +0.0155) | −0.020 (−0.050 to +0.009) | −0.047 (−0.094 to +0.000) | −0.005 (−0.059 to +0.049) |
| Ordinal LightGBM (comparison only) − Ordinal regression | −0.0036 (−0.0097 to +0.0019) | +0.010 (−0.011 to +0.031) | −0.010 (−0.042 to +0.023) | +0.036 (−0.017 to +0.090) |
| v1.1 binary models − Ordinal regression | – | – | +0.004 (−0.016 to +0.025) | −0.000 (−0.032 to +0.031) |

Subgroups (point estimates; RPS / C (utility) of the ordinal regression and the layer): men (n 106): 0.1384 / 0.787 and 0.1378 / 0.786; women (n 115): 0.1452 / 0.709 and 0.1430 / 0.703; age under 80 (n 170): 0.1463 / 0.753 and 0.1446 / 0.754.
Not shown (fewer than 10 patients in mRS 0–2, 3–5 or 6): age 80 or older.

**24 hours** (1115 development, 120 test patients; test grades 0–6: 11, 14, 11, 17, 21, 33, 13)

| Model | RPS | Log-loss | C (utility) | AUROC 0–2 | AUROC death | CITL 0–2 | Slope 0–2 | Slope (ordinal) | Utility |
|---|---|---|---|---|---|---|---|---|---|
| Ordinal regression | 0.1060 (0.0910 to 0.1224) | 1.542 (1.396 to 1.699) | 0.861 (0.825 to 0.894) | 0.912 (0.858 to 0.957) | 0.839 (0.723 to 0.926) | −1.009 (−1.500 to −0.514) | 1.073 (0.796 to 1.619) | 0.901 (0.728 to 1.170) | +0.092 |
| v1.1 layer | 0.1037 (0.0909 to 0.1176) | 1.483 (1.370 to 1.602) | 0.850 (0.812 to 0.885) | 0.907 (0.849 to 0.954) | 0.836 (0.705 to 0.938) | −0.549 (−0.987 to −0.127) | 1.428 (1.028 to 2.197) | – | +0.068 |
| v1.1 bands with ordinal shares | 0.1010 (0.0879 to 0.1153) | 1.457 (1.329 to 1.597) | 0.859 (0.823 to 0.891) | 0.907 (0.849 to 0.954) | 0.836 (0.705 to 0.938) | −0.549 (−0.987 to −0.127) | 1.428 (1.028 to 2.197) | – | +0.077 |
| MR PREDICTS@24H predictors, proportional odds | 0.1134 (0.0979 to 0.1299) | 1.582 (1.446 to 1.729) | 0.858 (0.823 to 0.891) | 0.907 (0.849 to 0.955) | 0.837 (0.716 to 0.928) | −1.195 (−1.665 to −0.694) | 1.101 (0.783 to 1.810) | 0.995 (0.796 to 1.281) | +0.118 |
| NIHSS alone, proportional odds | 0.1209 (0.1021 to 0.1417) | 1.632 (1.492 to 1.783) | 0.839 (0.792 to 0.882) | 0.889 (0.822 to 0.944) | 0.756 (0.577 to 0.900) | −1.282 (−1.766 to −0.773) | 1.003 (0.719 to 1.632) | 1.019 (0.790 to 1.347) | +0.125 |
| Ordinal LightGBM (comparison only) | 0.1122 (0.0956 to 0.1296) | 1.835 (1.436 to 2.490) | 0.845 (0.804 to 0.883) | 0.897 (0.835 to 0.945) | 0.841 (0.691 to 0.943) | −0.884 (−1.333 to −0.415) | 1.252 (0.902 to 2.024) | – | +0.097 |
| v1.1 binary models | – | – | – | 0.907 (0.849 to 0.954) | 0.836 (0.705 to 0.938) | −0.549 (−0.987 to −0.127) | 1.427 (1.026 to 2.195) | – | – |

Differences against the ordinal regression (other model minus ordinal; RPS and log-loss: negative favours the other model):

| Comparison | RPS | C (utility) | AUROC 0–2 | AUROC death |
|---|---|---|---|---|
| v1.1 layer − Ordinal regression | −0.0024 (−0.0110 to +0.0066) | −0.011 (−0.031 to +0.007) | −0.005 (−0.031 to +0.021) | −0.003 (−0.071 to +0.065) |
| v1.1 bands with ordinal shares − Ordinal regression | −0.0051 (−0.0117 to +0.0018) | −0.002 (−0.014 to +0.010) | −0.005 (−0.031 to +0.021) | −0.003 (−0.071 to +0.065) |
| MR PREDICTS@24H predictors, proportional odds − Ordinal regression | +0.0074 (+0.0010 to +0.0140) | −0.003 (−0.016 to +0.012) | −0.005 (−0.022 to +0.011) | −0.002 (−0.033 to +0.034) |
| NIHSS alone, proportional odds − Ordinal regression | +0.0149 (+0.0003 to +0.0314) | −0.021 (−0.055 to +0.009) | −0.023 (−0.059 to +0.009) | −0.083 (−0.206 to +0.026) |
| Ordinal LightGBM (comparison only) − Ordinal regression | +0.0062 (−0.0040 to +0.0175) | −0.015 (−0.040 to +0.006) | −0.015 (−0.046 to +0.011) | +0.002 (−0.059 to +0.062) |
| v1.1 binary models − Ordinal regression | – | – | −0.005 (−0.031 to +0.021) | −0.003 (−0.071 to +0.065) |

Not shown (fewer than 10 patients in mRS 0–2, 3–5 or 6): men, women, age under 80, age 80 or older.

**Day 7** (1000 development, 139 test patients; test grades 0–6: 20, 20, 12, 24, 25, 32, <10)

| Model | RPS | Log-loss | C (utility) | AUROC 0–2 | AUROC death | CITL 0–2 | Slope 0–2 | Slope (ordinal) | Utility |
|---|---|---|---|---|---|---|---|---|---|
| Ordinal regression | 0.1018 (0.0885 to 0.1160) | 1.483 (1.354 to 1.615) | 0.891 (0.861 to 0.917) | 0.939 (0.900 to 0.971) | 0.865 (0.730 to 0.976) | −1.278 (−1.688 to −0.866) | 1.590 (1.189 to 2.463) | 1.077 (0.891 to 1.411) | +0.094 |
| v1.1 layer | 0.1033 (0.0909 to 0.1165) | 1.511 (1.400 to 1.626) | 0.851 (0.812 to 0.884) | 0.920 (0.876 to 0.958) | 0.813 (0.546 to 0.973) | −0.628 (−1.033 to −0.230) | 1.651 (1.289 to 2.339) | – | +0.047 |
| v1.1 bands with ordinal shares | 0.0974 (0.0851 to 0.1107) | 1.451 (1.329 to 1.575) | 0.867 (0.834 to 0.899) | 0.920 (0.876 to 0.958) | 0.813 (0.546 to 0.973) | −0.628 (−1.033 to −0.230) | 1.651 (1.289 to 2.339) | – | +0.059 |
| MR PREDICTS@24H predictors, proportional odds | 0.1453 (0.1262 to 0.1652) | 1.810 (1.672 to 1.952) | 0.730 (0.674 to 0.783) | 0.774 (0.688 to 0.850) | 0.579 (0.399 to 0.769) | −0.355 (−0.854 to 0.129) | 0.677 (0.479 to 0.954) | 0.653 (0.508 to 0.857) | +0.050 |
| NIHSS alone, proportional odds | 0.1201 (0.1021 to 0.1395) | 1.634 (1.463 to 1.810) | 0.887 (0.858 to 0.915) | 0.924 (0.879 to 0.961) | 0.856 (0.707 to 0.979) | −1.916 (−2.357 to −1.490) | 1.300 (0.972 to 1.936) | 1.051 (0.849 to 1.385) | +0.140 |
| Ordinal LightGBM (comparison only) | 0.1056 (0.0894 to 0.1221) | 1.749 (1.375 to 2.337) | 0.880 (0.845 to 0.910) | 0.929 (0.889 to 0.964) | 0.856 (0.683 to 0.982) | −1.225 (−1.639 to −0.832) | 1.643 (1.251 to 2.388) | – | +0.109 |
| v1.1 binary models | – | – | – | 0.920 (0.876 to 0.958) | 0.813 (0.546 to 0.973) | −0.632 (−1.039 to −0.233) | 1.646 (1.283 to 2.335) | – | – |

Differences against the ordinal regression (other model minus ordinal; RPS and log-loss: negative favours the other model):

| Comparison | RPS | C (utility) | AUROC 0–2 | AUROC death |
|---|---|---|---|---|
| v1.1 layer − Ordinal regression | +0.0014 (−0.0076 to +0.0114) | −0.040 (−0.069 to −0.017) | −0.019 (−0.051 to +0.008) | −0.051 (−0.211 to +0.044) |
| v1.1 bands with ordinal shares − Ordinal regression | −0.0044 (−0.0109 to +0.0025) | −0.024 (−0.044 to −0.007) | −0.019 (−0.051 to +0.008) | −0.051 (−0.211 to +0.044) |
| MR PREDICTS@24H predictors, proportional odds − Ordinal regression | +0.0435 (+0.0237 to +0.0636) | −0.161 (−0.211 to −0.112) | −0.166 (−0.245 to −0.095) | −0.286 (−0.395 to −0.154) |
| NIHSS alone, proportional odds − Ordinal regression | +0.0182 (+0.0075 to +0.0295) | −0.004 (−0.023 to +0.015) | −0.015 (−0.040 to +0.008) | −0.009 (−0.054 to +0.025) |
| Ordinal LightGBM (comparison only) − Ordinal regression | +0.0038 (−0.0043 to +0.0119) | −0.011 (−0.031 to +0.007) | −0.010 (−0.041 to +0.020) | −0.009 (−0.075 to +0.033) |
| v1.1 binary models − Ordinal regression | – | – | −0.019 (−0.051 to +0.008) | −0.051 (−0.211 to +0.044) |

Not shown (fewer than 10 patients in mRS 0–2, 3–5 or 6): men, women, age under 80, age 80 or older.

### Sensitivity split: development before 2021, test 2021–2025

**Before thrombectomy** (1072 development, 379 test patients; test grades 0–6: 39, 48, 23, 46, 70, 76, 77)

| Model | RPS | Log-loss | C (utility) | AUROC 0–2 | AUROC death | CITL 0–2 | Slope 0–2 | Slope (ordinal) | Utility |
|---|---|---|---|---|---|---|---|---|---|
| Ordinal regression | 0.1613 (0.1512 to 0.1717) | 1.795 (1.740 to 1.851) | 0.696 (0.660 to 0.730) | 0.755 (0.698 to 0.807) | 0.722 (0.658 to 0.785) | −0.404 (−0.643 to −0.180) | 1.132 (0.817 to 1.500) | 0.995 (0.800 to 1.221) | +0.054 |
| v1.1 layer | 0.1601 (0.1500 to 0.1708) | 1.771 (1.715 to 1.827) | 0.691 (0.655 to 0.726) | 0.752 (0.697 to 0.803) | 0.752 (0.692 to 0.812) | −0.315 (−0.557 to −0.089) | 1.088 (0.809 to 1.425) | – | +0.040 |
| v1.1 bands with ordinal shares | 0.1601 (0.1500 to 0.1706) | 1.771 (1.717 to 1.827) | 0.692 (0.657 to 0.727) | 0.752 (0.697 to 0.803) | 0.752 (0.692 to 0.812) | −0.315 (−0.557 to −0.089) | 1.088 (0.809 to 1.425) | – | +0.043 |
| MR PREDICTS predictors, proportional odds (no collaterals) | 0.1656 (0.1550 to 0.1760) | 1.818 (1.758 to 1.875) | 0.689 (0.653 to 0.724) | 0.752 (0.696 to 0.803) | 0.700 (0.632 to 0.764) | −0.587 (−0.830 to −0.352) | 1.017 (0.753 to 1.332) | 0.903 (0.723 to 1.112) | +0.081 |
| v1.1 binary models | – | – | – | 0.752 (0.697 to 0.803) | 0.752 (0.692 to 0.812) | −0.315 (−0.557 to −0.089) | 1.088 (0.809 to 1.425) | – | – |

Differences against the ordinal regression (other model minus ordinal; RPS and log-loss: negative favours the other model):

| Comparison | RPS | C (utility) | AUROC 0–2 | AUROC death |
|---|---|---|---|---|
| v1.1 layer − Ordinal regression | −0.0012 (−0.0034 to +0.0011) | −0.004 (−0.010 to +0.002) | −0.003 (−0.017 to +0.010) | +0.030 (−0.000 to +0.058) |
| v1.1 bands with ordinal shares − Ordinal regression | −0.0012 (−0.0032 to +0.0009) | −0.003 (−0.008 to +0.002) | −0.003 (−0.017 to +0.010) | +0.030 (−0.000 to +0.058) |
| MR PREDICTS predictors, proportional odds (no collaterals) − Ordinal regression | +0.0044 (−0.0014 to +0.0100) | −0.007 (−0.026 to +0.013) | −0.003 (−0.032 to +0.023) | −0.023 (−0.061 to +0.015) |
| v1.1 binary models − Ordinal regression | – | – | −0.003 (−0.017 to +0.010) | +0.030 (−0.000 to +0.058) |

Subgroups (point estimates; RPS / C (utility) of the ordinal regression and the layer): men (n 192): 0.1600 / 0.719 and 0.1589 / 0.715; women (n 187): 0.1625 / 0.674 and 0.1614 / 0.670; age under 80 (n 299): 0.1662 / 0.686 and 0.1649 / 0.684; age 80 or older (n 80): 0.1429 / 0.694 and 0.1422 / 0.680.

**End of procedure** (1060 development, 371 test patients; test grades 0–6: 37, 48, 23, 45, 68, 76, 74)

| Model | RPS | Log-loss | C (utility) | AUROC 0–2 | AUROC death | CITL 0–2 | Slope 0–2 | Slope (ordinal) | Utility |
|---|---|---|---|---|---|---|---|---|---|
| Ordinal regression | 0.1514 (0.1417 to 0.1617) | 1.750 (1.693 to 1.805) | 0.730 (0.698 to 0.762) | 0.788 (0.737 to 0.835) | 0.759 (0.697 to 0.814) | −0.408 (−0.641 to −0.171) | 1.112 (0.839 to 1.471) | 1.031 (0.852 to 1.253) | +0.057 |
| v1.1 layer | 0.1515 (0.1417 to 0.1619) | 1.735 (1.673 to 1.793) | 0.722 (0.690 to 0.756) | 0.779 (0.731 to 0.827) | 0.774 (0.713 to 0.831) | −0.294 (−0.535 to −0.047) | 0.988 (0.762 to 1.279) | – | +0.036 |
| v1.1 bands with ordinal shares | 0.1514 (0.1416 to 0.1617) | 1.736 (1.675 to 1.793) | 0.725 (0.693 to 0.758) | 0.779 (0.731 to 0.827) | 0.774 (0.713 to 0.831) | −0.294 (−0.535 to −0.047) | 0.988 (0.762 to 1.279) | – | +0.040 |
| MR PREDICTS predictors, proportional odds (no collaterals) | 0.1645 (0.1542 to 0.1756) | 1.815 (1.759 to 1.872) | 0.689 (0.652 to 0.725) | 0.749 (0.691 to 0.803) | 0.698 (0.628 to 0.763) | −0.574 (−0.816 to −0.335) | 1.022 (0.755 to 1.345) | 0.921 (0.738 to 1.130) | +0.080 |
| Ordinal LightGBM (comparison only) | 0.1446 (0.1351 to 0.1548) | 2.816 (2.240 to 3.423) | 0.755 (0.724 to 0.785) | 0.797 (0.746 to 0.844) | 0.785 (0.725 to 0.840) | −0.464 (−0.706 to −0.223) | 1.141 (0.884 to 1.456) | – | +0.069 |
| v1.1 binary models | – | – | – | 0.779 (0.731 to 0.827) | 0.774 (0.713 to 0.831) | −0.294 (−0.535 to −0.048) | 0.988 (0.762 to 1.279) | – | – |

Differences against the ordinal regression (other model minus ordinal; RPS and log-loss: negative favours the other model):

| Comparison | RPS | C (utility) | AUROC 0–2 | AUROC death |
|---|---|---|---|---|
| v1.1 layer − Ordinal regression | +0.0000 (−0.0024 to +0.0026) | −0.008 (−0.014 to −0.002) | −0.008 (−0.025 to +0.010) | +0.015 (−0.014 to +0.044) |
| v1.1 bands with ordinal shares − Ordinal regression | −0.0001 (−0.0023 to +0.0022) | −0.006 (−0.011 to −0.000) | −0.008 (−0.025 to +0.010) | +0.015 (−0.014 to +0.044) |
| MR PREDICTS predictors, proportional odds (no collaterals) − Ordinal regression | +0.0130 (+0.0062 to +0.0202) | −0.041 (−0.066 to −0.018) | −0.038 (−0.077 to −0.003) | −0.060 (−0.112 to −0.014) |
| Ordinal LightGBM (comparison only) − Ordinal regression | −0.0068 (−0.0121 to −0.0016) | +0.025 (+0.008 to +0.043) | +0.010 (−0.019 to +0.037) | +0.027 (−0.019 to +0.069) |
| v1.1 binary models − Ordinal regression | – | – | −0.008 (−0.025 to +0.010) | +0.015 (−0.014 to +0.044) |

Subgroups (point estimates; RPS / C (utility) of the ordinal regression and the layer): men (n 189): 0.1513 / 0.752 and 0.1523 / 0.743; women (n 182): 0.1516 / 0.709 and 0.1506 / 0.702; age under 80 (n 293): 0.1562 / 0.723 and 0.1565 / 0.716; age 80 or older (n 78): 0.1334 / 0.736 and 0.1325 / 0.714.

**24 hours** (1029 development, 206 test patients; test grades 0–6: 22, 34, 17, 26, 35, 45, 27)

| Model | RPS | Log-loss | C (utility) | AUROC 0–2 | AUROC death | CITL 0–2 | Slope 0–2 | Slope (ordinal) | Utility |
|---|---|---|---|---|---|---|---|---|---|
| Ordinal regression | 0.1200 (0.1049 to 0.1363) | 1.606 (1.475 to 1.743) | 0.826 (0.791 to 0.860) | 0.856 (0.802 to 0.905) | 0.898 (0.841 to 0.943) | −0.665 (−1.110 to −0.197) | 0.699 (0.529 to 0.943) | 0.772 (0.648 to 0.935) | +0.063 |
| v1.1 layer | 0.1223 (0.1082 to 0.1371) | 1.619 (1.507 to 1.739) | 0.811 (0.775 to 0.846) | 0.846 (0.792 to 0.895) | 0.879 (0.804 to 0.940) | −0.306 (−0.694 to 0.104) | 0.799 (0.595 to 1.100) | – | +0.032 |
| v1.1 bands with ordinal shares | 0.1203 (0.1057 to 0.1355) | 1.606 (1.485 to 1.734) | 0.818 (0.782 to 0.852) | 0.846 (0.792 to 0.895) | 0.879 (0.804 to 0.940) | −0.306 (−0.694 to 0.104) | 0.799 (0.595 to 1.100) | – | +0.041 |
| MR PREDICTS@24H predictors, proportional odds | 0.1282 (0.1119 to 0.1459) | 1.663 (1.527 to 1.809) | 0.821 (0.783 to 0.855) | 0.847 (0.792 to 0.897) | 0.892 (0.834 to 0.939) | −0.982 (−1.420 to −0.510) | 0.657 (0.487 to 0.905) | 0.799 (0.657 to 0.990) | +0.094 |
| NIHSS alone, proportional odds | 0.1304 (0.1143 to 0.1489) | 1.667 (1.545 to 1.810) | 0.818 (0.780 to 0.852) | 0.844 (0.790 to 0.894) | 0.851 (0.763 to 0.922) | −1.074 (−1.489 to −0.598) | 0.698 (0.532 to 0.926) | 0.869 (0.708 to 1.060) | +0.101 |
| Ordinal LightGBM (comparison only) | 0.1231 (0.1075 to 0.1402) | 2.532 (1.825 to 3.368) | 0.826 (0.789 to 0.859) | 0.859 (0.807 to 0.905) | 0.882 (0.807 to 0.941) | −0.710 (−1.104 to −0.285) | 0.810 (0.616 to 1.101) | – | +0.075 |
| v1.1 binary models | – | – | – | 0.846 (0.792 to 0.895) | 0.879 (0.804 to 0.941) | −0.307 (−0.695 to 0.104) | 0.799 (0.595 to 1.100) | – | – |

Differences against the ordinal regression (other model minus ordinal; RPS and log-loss: negative favours the other model):

| Comparison | RPS | C (utility) | AUROC 0–2 | AUROC death |
|---|---|---|---|---|
| v1.1 layer − Ordinal regression | +0.0023 (−0.0044 to +0.0089) | −0.015 (−0.029 to −0.002) | −0.010 (−0.030 to +0.008) | −0.019 (−0.055 to +0.013) |
| v1.1 bands with ordinal shares − Ordinal regression | +0.0003 (−0.0049 to +0.0053) | −0.008 (−0.018 to +0.001) | −0.010 (−0.030 to +0.008) | −0.019 (−0.055 to +0.013) |
| MR PREDICTS@24H predictors, proportional odds − Ordinal regression | +0.0082 (+0.0012 to +0.0157) | −0.006 (−0.020 to +0.009) | −0.009 (−0.031 to +0.012) | −0.006 (−0.028 to +0.014) |
| NIHSS alone, proportional odds − Ordinal regression | +0.0104 (−0.0021 to +0.0236) | −0.008 (−0.034 to +0.016) | −0.012 (−0.044 to +0.023) | −0.047 (−0.114 to +0.006) |
| Ordinal LightGBM (comparison only) − Ordinal regression | +0.0031 (−0.0070 to +0.0132) | −0.000 (−0.022 to +0.019) | +0.003 (−0.027 to +0.033) | −0.016 (−0.060 to +0.024) |
| v1.1 binary models − Ordinal regression | – | – | −0.010 (−0.030 to +0.008) | −0.019 (−0.054 to +0.013) |

Subgroups (point estimates; RPS / C (utility) of the ordinal regression and the layer): men (n 103): 0.1179 / 0.830 and 0.1266 / 0.805; women (n 103): 0.1221 / 0.823 and 0.1179 / 0.819; age under 80 (n 169): 0.1235 / 0.820 and 0.1272 / 0.804.
Not shown (fewer than 10 patients in mRS 0–2, 3–5 or 6): age 80 or older.

**Day 7** (934 development, 205 test patients; test grades 0–6: 33, 38, 17, 31, 36, 39, 11)

| Model | RPS | Log-loss | C (utility) | AUROC 0–2 | AUROC death | CITL 0–2 | Slope 0–2 | Slope (ordinal) | Utility |
|---|---|---|---|---|---|---|---|---|---|
| Ordinal regression | 0.1034 (0.0915 to 0.1160) | 1.482 (1.370 to 1.598) | 0.860 (0.829 to 0.888) | 0.913 (0.873 to 0.948) | 0.904 (0.821 to 0.968) | −0.878 (−1.247 to −0.512) | 1.220 (0.958 to 1.632) | 0.978 (0.839 to 1.192) | +0.070 |
| v1.1 layer | 0.1111 (0.0997 to 0.1228) | 1.555 (1.458 to 1.658) | 0.823 (0.785 to 0.857) | 0.886 (0.840 to 0.928) | 0.881 (0.730 to 0.970) | −0.279 (−0.630 to 0.053) | 1.165 (0.912 to 1.546) | – | +0.015 |
| v1.1 bands with ordinal shares | 0.1062 (0.0948 to 0.1181) | 1.508 (1.404 to 1.623) | 0.838 (0.801 to 0.870) | 0.886 (0.840 to 0.928) | 0.881 (0.730 to 0.970) | −0.279 (−0.630 to 0.053) | 1.165 (0.912 to 1.546) | – | +0.031 |
| MR PREDICTS@24H predictors, proportional odds | 0.1877 (0.1629 to 0.2163) | 2.135 (1.949 to 2.352) | 0.676 (0.622 to 0.724) | 0.730 (0.654 to 0.795) | 0.703 (0.568 to 0.833) | 0.454 (−0.259 to 1.162) | 0.343 (0.224 to 0.470) | 0.349 (0.238 to 0.464) | −0.073 |
| NIHSS alone, proportional odds | 0.1161 (0.1009 to 0.1328) | 1.586 (1.447 to 1.744) | 0.874 (0.845 to 0.900) | 0.914 (0.873 to 0.948) | 0.901 (0.816 to 0.972) | −1.659 (−2.040 to −1.276) | 1.167 (0.913 to 1.581) | 1.005 (0.850 to 1.253) | +0.123 |
| Ordinal LightGBM (comparison only) | 0.1070 (0.0933 to 0.1223) | 2.172 (1.557 to 2.887) | 0.867 (0.835 to 0.895) | 0.929 (0.891 to 0.958) | 0.908 (0.816 to 0.974) | −1.025 (−1.355 to −0.724) | 1.677 (1.306 to 2.266) | – | +0.096 |
| v1.1 binary models | – | – | – | 0.886 (0.840 to 0.928) | 0.881 (0.730 to 0.970) | −0.282 (−0.633 to 0.050) | 1.162 (0.909 to 1.542) | – | – |

Differences against the ordinal regression (other model minus ordinal; RPS and log-loss: negative favours the other model):

| Comparison | RPS | C (utility) | AUROC 0–2 | AUROC death |
|---|---|---|---|---|
| v1.1 layer − Ordinal regression | +0.0077 (−0.0007 to +0.0154) | −0.037 (−0.057 to −0.018) | −0.027 (−0.052 to −0.004) | −0.023 (−0.104 to +0.026) |
| v1.1 bands with ordinal shares − Ordinal regression | +0.0028 (−0.0034 to +0.0087) | −0.022 (−0.037 to −0.009) | −0.027 (−0.052 to −0.004) | −0.023 (−0.104 to +0.026) |
| MR PREDICTS@24H predictors, proportional odds − Ordinal regression | +0.0843 (+0.0606 to +0.1116) | −0.184 (−0.233 to −0.139) | −0.184 (−0.252 to −0.123) | −0.201 (−0.286 to −0.115) |
| NIHSS alone, proportional odds − Ordinal regression | +0.0127 (+0.0023 to +0.0225) | +0.014 (−0.005 to +0.035) | +0.001 (−0.024 to +0.026) | −0.003 (−0.038 to +0.033) |
| Ordinal LightGBM (comparison only) − Ordinal regression | +0.0036 (−0.0036 to +0.0112) | +0.007 (−0.009 to +0.023) | +0.016 (−0.011 to +0.044) | +0.004 (−0.026 to +0.031) |
| v1.1 binary models − Ordinal regression | – | – | −0.027 (−0.052 to −0.004) | −0.023 (−0.104 to +0.026) |

Not shown (fewer than 10 patients in mRS 0–2, 3–5 or 6): men, women, age under 80, age 80 or older.

## 8. Deployed model (section 11)

Refitted on every patient of 2005–2025 with the development-year knots and penalty and rule O3's factor; the recent-era threshold shift estimated on 2023–2025 from out-of-fold estimates; the slope changed only under the update policy (slope outside 0.8–1.2 with a 95 % interval that excludes 1).

| Stage | Patients | Coefficients | Knots | C | O3 factor | Recent shift | Recent slope (95 % interval) | Slope updated |
|---|---|---|---|---|---|---|---|---|
| Before thrombectomy | 1451 | 59 | 3 | 0.009103 | 1.000 | −0.101 | 1.17 (0.88–1.45) | no |
| End of procedure | 1431 | 77 | 3 | 0.009103 | 1.000 | −0.110 | 1.11 (0.85–1.36) | no |
| 24 hours | 1235 | 76 | 3 | 0.6251 | 1.000 | −0.396 | 0.89 (0.69–1.09) | no |
| Day 7 | 1139 | 82 | 3 | 2.56 | 0.887 | −0.284 | 0.97 (0.77–1.18) | no |

Out-of-fold estimates in 2023–2025 before and after the threshold shift:

| Stage | Recent patients | RPS before → after | CITL mRS 0–2 before → after | CITL death before → after | Utility before → after | C (utility) |
|---|---|---|---|---|---|---|
| Before thrombectomy | 226 | 0.1481 → 0.1472 | −0.43 → −0.33 | −0.37 → −0.47 | +0.056 → +0.037 | 0.725 |
| End of procedure | 221 | 0.1403 → 0.1394 | −0.43 → −0.32 | −0.39 → −0.50 | +0.057 → +0.037 | 0.747 |
| 24 hours | 120 | 0.1065 → 0.1011 | −0.88 → −0.48 | −1.06 → −1.46 | +0.088 → +0.045 | 0.854 |
| Day 7 | 139 | 0.1007 → 0.0978 | −1.00 → −0.72 | −2.01 → −2.30 | +0.067 → +0.041 | 0.876 |

CITL: calibration-in-the-large (negative: the outcome is predicted more often than it happened). One common shift of all thresholds moves every patient towards worse grades: it reduces the over-prediction of independence but adds to the over-prediction of death, because in 2023–2025 the outcomes moved towards mRS 4–5 rather than towards death. This is why the calculator takes independence and death from the binary models (with their own recent-era intercepts) and uses the ordinal model only to split the bands into grades.

**MTOP ordinal 1.0.0**, fingerprint `78436c388af37724da897a7d29e95f103b12b2a3a8c5de47d30b4286b393ea46`; calculator distribution: the v1.1 bands split by the ordinal model. Built on the binary models of version 1.1.0 (`841d810c28097e00…`), which are unchanged.

Browser parity (web/model.js against the Python model, every patient of each stage, 5256 stage-patient rows, and the synthetic demonstration cohort): largest difference in the linear predictor, any grade probability or any explanation term 1.5e-14 (exported numbers 5.3e-15); criterion 10⁻⁷: **passed**.

## 9. Reporting (TRIPOD+AI)

| TRIPOD+AI topic | Where |
|---|---|
| Source of data, participants, outcome | plan, section 3; [model card](../model_card_ordinal.md), section 2; section 1 here |
| Predictors and their handling, missing data | plan, section 4; [diagnostics](../model_diagnostics_ordinal.md) |
| Sample size | section 5 |
| Model type, tuning, internal validation | plan, sections 4–7, and amendment 1; sections 2–4 |
| Comparators and the comparison-only learner | sections 2, 3 and 7 |
| Recalibration and updating | rule O3 (decisions) and section 8 |
| Performance with intervals, subgroups | sections 2, 3 and 7 |
| Full model specification | artifacts/public/ordinal_bundle.json and its fingerprint |
| Limitations and intended use | [model card](../model_card_ordinal.md), sections 1 and 6 |

