# data/raw — patient data (not in git)

Put the registry export here as `mt_patients_master.xlsx` (one sheet, one row per patient,
columns as in the FNO master file). Everything in this folder except this README is ignored
by git. Never commit, upload or e-mail the raw file together with the code.

To try the pipeline without real data:

```bash
python -m mtop all --input data/synthetic/registry_synthetic.xlsx --quick
```
