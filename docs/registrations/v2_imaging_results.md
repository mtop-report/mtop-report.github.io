# MTOP version 2.0 imaging module: results of Aim A and Aim B (ISLES'24)

Results under [v2_imaging_plan.md](v2_imaging_plan.md), run on 28 November 2026 on a MetaCentrum
(CESNET) computer. Aggregates only; no row-level data leave that computer.

## Data

- ISLES'24 training set, version v7 (Zenodo record 17652035, 20 November 2025). Archive
  `train.7z`, MD5 `4959a5dd2438d53e3c86d6858484e781`, checked after download.
- 149 cases with perfusion maps and a follow-up infarct mask; 148 with clinical tables.
- Tmax values were in seconds (step 1: medians 2.5–3.0 s, 95th percentiles 7.5–21 s in the first
  five cases), so no unit factor was needed.
- The column mapping was fixed from column names alone and committed (d08fc5e) before step 2.

## Aim A: perfusion-volume extractor

| | Primary definition | Clusters < 0.5 mL removed (sensitivity) |
|---|---|---|
| Cases with both volumes | 149 | 149 |
| Errors | 0 | 0 |
| Spearman correlation, core vs follow-up infarct (95 % CI) | **0.272** (0.092–0.444) | 0.272 (0.095–0.444) |
| Median core (mL) | 11.9 | 9.2 |
| Median follow-up infarct (mL) | 12.4 | 12.4 |
| Bland–Altman mean difference, follow-up minus core (mL) | 13.8 | 16.2 |
| 95 % limits of agreement (mL) | −61.3 to 88.8 | −58.9 to 91.4 |
| Follow-up infarct larger than core | 54.4 % | 63.1 % |

**Decision (pre-specified, section 4.2): the extractor is not usable.** The correlation is below
the registered minimum of 0.5, and the upper limit of its 95 % interval (0.44) is also below it.
Following amendment 1 (section A1.4), the tool is not offered in the app, and the data page says so.

The median volumes agree, but agreement in individual patients is poor. The following are possible
reasons. They are hypotheses, not tested here:

- **No CSF exclusion.** The method has no mask for cerebrospinal fluid or bone. Cerebrospinal
  fluid has low CBF and unreliable Tmax, so it can be counted as core.
- **A pooled reference.** The reference CBF is the median of all normally perfused tissue, grey
  and white matter together. Commercial programs compare with the opposite hemisphere.
- **The maps' own processing.** The maps were computed by the dataset's own pipeline, and its
  relative CBF scaling may not suit a fixed 30 % threshold.

Any revised method would need a new registered plan. ISLES'24 has now been used for this test, so
a revision should also be checked on data that played no part in designing it.

## Aim B: technical demonstration node

The ISLES'24 cases were written in the MTOP upload template and read by the network's own template
reader:

- **Rows:** 148; 128 with a 90-day mRS.
- **Columns mapped:** 11. They are age, pre-stroke mRS, NIHSS on admission, atrial fibrillation,
  hypertension, diabetes, hyperlipidaemia, statin (lipid-lowering drugs), final TICI, NIHSS at
  24 hours and 90-day mRS.
- **Unrecognised values:** 1 (hyperlipidaemia; treated as unknown).
- **Perfusion columns recorded:** 128 patients with a known outcome, from the extractor of Aim A.
  They are not model inputs.
- **Patients each stage could use:** before thrombectomy 128, end of procedure 89, 24 hours 74,
  day 7 0 (no day-7 NIHSS in the dataset).

The node works as intended. It can join the four-centre simulation of amendment 1, which waits for
the MR CLEAN team's permission to use MRS CLEAN.
