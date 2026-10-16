# reports — local outputs (not in git)

`python -m mtop report` writes here:

- `data_quality.xlsx` — every value the cleaner changed or removed, with patient IDs, so the
  source file can be corrected. Contains patient IDs: keep it inside FNO.
- `table1.xlsx` — baseline characteristics (development vs temporal validation, outcome groups)
- `validation_summary.md` — all performance tables for the manuscript
- `figures/` — ROC + calibration per stage, AUROC by stage, SHAP summaries (300-dpi PNG)
