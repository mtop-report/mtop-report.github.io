# MTOP data studio (local, FNO only)

The data studio shows the registry at every stage of the cleaning, marks every value the cleaning
changed or flagged, and lets the registry team correct the export. Corrections are recorded, not
written over the Excel file. It runs only on the analysis computer at FNO. It is not part of the
public app, and nothing it shows or stores is published.

## Start

```bash
pip install -e ".[studio]"                 # Streamlit
python -m mtop studio --set-password       # once; stored as a scrypt hash in data/studio/password.json
python -m mtop studio                      # opens http://127.0.0.1:8765 on this computer
python -m mtop studio --input other.xlsx   # another registry export
```

The server listens on 127.0.0.1 only, so other computers on the network cannot reach it. Streamlit's
usage statistics are switched off. The studio locks after 30 minutes without activity.

## What it shows

| Version | Content |
|---|---|
| Registry export | the Excel file as recorded (48 columns), with the active corrections applied |
| Cleaned registry | every record after the cleaning rules: standard names and codes, NIHSS status, derived columns, why a record is not analysed |
| Analysed patients | adults with a 90-day mRS, with their temporal split and the stage cohorts they belong to |
| Model inputs (4 stages) | each stage cohort with exactly the values the models use |

Column names switch between English and Czech. Cells are marked:

- **red**: the cleaning set the value to missing (implausible or unreadable);
- **amber**: the value is kept but listed for review (two fields disagree, for example a
  thrombectomy label with the extraction system "žádný");
- **blue**: repaired automatically (a one-year typo in the onset date);
- **purple**: the record is kept in the registry but not analysed (under 18);
- **green**: changed by a correction;
- **grey**: checked, the value is correct.

Each table can be filtered to the rows with a given mark and searched by patient ID or any value.
Selecting a row opens the record: what the cleaning found, where the record stands in every version
(for example "set to missing in the cleaned registry, not in the 24-hour cohort"), and the
correction tools. The *To review* tab lists every finding with its status (open, corrected,
checked).

## Correcting

For a finding: **apply the suggestion** (for example 25.78 for a BMI of 257.8, a likely misplaced
decimal point), **type the right value**, or confirm that **the value is correct**. Any other value
of the record can be corrected as well. Every entry needs your name and a reason.

- Entries go to `data/studio/corrections.csv`. Each one records the patient ID, the registry
  column, the value it was made against, the new value, the reason, who made it and when. The Excel
  file is never overwritten.
- `python -m mtop clean` (and *Re-run the cleaning and save* in the studio) applies the active
  corrections to a copy of the export before cleaning. A correction applies only while the export
  still holds the value it was made against. If a newer export already contains the fix, it is
  reported as "already in the source". If the export holds a different value, it is a conflict and
  is not applied. `python -m mtop clean --no-corrections` cleans the export exactly as recorded.
- Reverting keeps the entry, marked as reverted, with who reverted it and when. A new correction of
  the same cell replaces the earlier one ("replaced").
- `data/studio/audit.csv` logs every action (correct, confirm, revert, re-run, export).
- The *Corrections* tab downloads the corrected export (Excel, with the corrections and the findings
  as extra sheets) to send to the registry team, so the source is fixed too.

The models and the public app change only after `python -m mtop all` and a commit. The public app
shows the number of corrections applied, and only when it is at least 10.

## Privacy

`data/studio/` holds patient IDs and values. Git ignores it, and the CI check
(`scripts/check_no_patient_data.sh`) refuses any tracked file there. Keep it on the FNO computer.
Do not copy it into the repository, e-mail it outside FNO or run the studio on a server.
