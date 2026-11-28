# MTOP version 2.0, imaging module: perfusion-volume extractor and ISLES'24 demonstration node

Plan, 28 November 2026. Committed to the repository before the ISLES'24 data are downloaded; the
commit's time on GitHub is its time stamp. The code it describes (`src/mtop/isles.py`,
settings in `config/isles24.yaml`) was written and tested on synthetic maps only
(`tests/test_isles.py`). Any later change is added at the end as a dated amendment with its reason.

## 1. Purpose

MTOP (Mechanical Thrombectomy Outcome Predictor) uses no imaging before thrombectomy apart from
ASPECTS (Alberta Stroke Program Early CT Score). Amendment 1 added optional columns to the partner
template for CT perfusion volumes, the CT angiography collateral score and serum GFAP (glial
fibrillary acidic protein). They are collected but never used by a model until their added value
is tested on at least 100 patients with mRS (modified Rankin Scale) 0–2 and 100 with mRS 3–6.

This plan adds two things, both preparatory:

- **Aim A. A perfusion-volume extractor.** Partner centres usually report volumes from their own
  perfusion software (for example RAPID, Olea, syngo.via). Centres without such software, or
  without exported volumes, could compute them from their perfusion maps with one fixed, openly
  defined method. Aim A tests that method on a public dataset before it is offered to anyone.
- **Aim B. A technical demonstration node.** The ISLES'24 cases, written in the MTOP upload
  template with the extracted volumes as the optional perfusion columns, check that the network
  software reads data from outside FNO (University Hospital Ostrava) correctly.

## 2. What this plan does not do

- **No model changes.** MTOP v1.0.0 (fingerprint 77bb37a3…), version 1.1 and MTOP ordinal 1.0.0
  stay as they are. No imaging variable enters any model under this plan.
- **No outcome analysis on ISLES'24.** No MTOP model is fitted, evaluated or recalibrated on
  ISLES'24, and the added value of perfusion predictors is not tested on it. ISLES'24 is too small
  for that (149 public training cases, below the 100 + 100 rule of amendment 1). It is also
  selected: every patient had successful reperfusion, so patients with failed thrombectomy, who
  carry much of the information about poor outcome, are absent.
- **No claim of external validation.** Results from Aim B are reported as a software test only.
- **Any outcome analysis on ISLES'24 needs a new amendment**, committed before it is run.

## 3. Data

- **ISLES'24 training set** (Ischemic Stroke Lesion Segmentation Challenge 2024), Technical
  University of Munich, University of Zurich and partners. Zenodo, latest version via
  https://zenodo.org/records/16748089/latest. The version number and the MD5 checksum of the
  downloaded archive are recorded at download, in the first amendment or in the run summary.
- **Contents used:** acute perfusion maps (Tmax, time to maximum of the residue function; CBF,
  cerebral blood flow), the infarct mask on follow-up MRI (ses-0002), and the clinical tables
  (demographics, admission NIHSS (National Institutes of Health Stroke Scale), 3-month mRS).
- **Licence:** CC BY-NC-SA 4.0 (attribution, non-commercial, share-alike). The work is
  non-commercial research. Row-level derived data (per-case volumes, the template file) are not
  redistributed and stay in a git-ignored folder (`data/raw/isles24/`). Only aggregates are
  published (`artifacts/public/isles24_extractor.json`), and the dataset is cited.
- **Where it is processed:** the archive (about 99 GB) is downloaded and processed on an
  IT4Innovations computer or another research computer, not on a hospital system.

## 4. Aim A: perfusion-volume extractor

### 4.1 Definitions (fixed in `config/isles24.yaml`)

All volumes in mL, from voxel counts × voxel volume (taken from the image header). The maps
provided with the dataset are used as they are, in their original space; when both an original
and a co-registered copy exist, the original is used.

| Quantity | Definition |
|---|---|
| Valid tissue | Tmax finite and within 0–30 s, CBF finite and above 0 |
| Hypoperfusion (`ctp_tmax6_ml`) | Valid voxels with Tmax > 6 s |
| Reference CBF | Median CBF of valid voxels with Tmax < 4 s; at least 1,000 such voxels, otherwise the core is missing |
| Ischaemic core (`ctp_core_ml`) | Valid voxels with CBF < 30 % of the reference and Tmax > 6 s |
| Mismatch volume (`ctp_mismatch_ml`) | Hypoperfusion minus core |
| Mismatch ratio (`ctp_mismatch_ratio`) | Hypoperfusion ÷ core; missing when the core is 0 |
| Relative mismatch (`ctp_mismatch_rel`) | Mismatch ÷ hypoperfusion × 100; missing when hypoperfusion is 0 |
| Follow-up infarct | Voxels of the follow-up MRI infarct mask (value > 0.5) |

These follow the thresholds most perfusion software uses (relative CBF < 30 %, Tmax > 6 s). The
reference CBF uses normally perfused tissue (Tmax < 4 s) instead of a mirrored hemisphere, so
that no registration to a template is needed. No morphological cleaning is applied in the primary
definition.

**Units check (step 1, section 5).** Before the volumes are computed, the 5th, 50th and 95th
percentiles of the Tmax map in the first five cases are read to confirm that Tmax is in seconds.
If it is not, the only allowed change is the unit factor (`tmax_unit_factor`), recorded in the
configuration and an amendment before step 2. No threshold is changed after the data are seen.

### 4.2 Analysis

Patients with successful reperfusion are the group in which the acute core should best predict the
final infarct, which makes ISLES'24 suitable for this check even though it is unsuitable for
outcome analysis.

- **Primary:** Spearman correlation between the automated core and the follow-up infarct volume,
  with a 95 % percentile bootstrap interval (2,000 resamples, seed 2026).
- **Secondary:**
  - Bland–Altman mean difference (follow-up minus core) and 95 % limits of agreement, in mL;
  - median volumes;
  - the share of cases in which the follow-up infarct is larger than the core (expected to be most,
    because infarcts grow and DWI also shows tissue lost after reperfusion).
- **Sensitivity:** the same after removing connected clusters smaller than 0.5 mL (26-connectivity)
  from the hypoperfusion and core masks.
- **Failures:** cases whose maps are missing or unreadable are counted and listed by error type;
  they are not imputed.
- **Pre-specified decision:** the extractor is offered to partner centres only if the primary
  Spearman correlation is at least 0.5. Otherwise centres report volumes from their own software
  only, and the extractor is described as not suitable.

The correlation threshold reflects the moderate agreement usually reported between CT perfusion
core and final infarct in reperfused patients. It is a minimum for usefulness, not a target.

## 5. Aim B: technical demonstration node

1. **Step 1, column names only.** `python -m mtop isles --root ISLES24_DIR --headers-only`
   prints the clinical column names, the number of missing values per column and the Tmax value
   ranges of section 4.1. It prints no clinical values and no outcome distribution.
2. **Mapping fixed.** From the names alone, the mapping in `config/isles24.yaml` (ISLES'24 column
   → MTOP template column) is completed. `mapping_status` is set to `fixed` with the date and
   committed before step 3. A column whose meaning is unclear from its name and the dataset
   documentation (for example whether an "admission" mRS is the pre-stroke mRS) is left unmapped.
3. **Step 2, full run.** `python -m mtop isles --root ISLES24_DIR` computes the volumes (Aim A),
   writes the cases in the upload template with the volumes as the optional perfusion columns,
   and reads that file with the network's own template reader (`fed/schema.py`). The node is built
   only when the mapping is fixed.

**Reported (aggregates only):**

- the template reader's mapping report: columns found and missing, unrecognised values, patients
  with a known outcome;
- the number of patients each MTOP stage and outcome could use;
- the number of patients with each perfusion column recorded.

No model is trained or evaluated, and no performance is reported.

## 6. Outputs

- `artifacts/public/isles24_extractor.json`: counts, the agreement results of Aim A and the
  node's mapping report (aggregates only).
- `data/raw/isles24/` (git-ignored): per-case volumes and the template file. Not published,
  because of the dataset licence.
- A short results file, `docs/registrations/v2_imaging_results.md`, with the decision of
  section 4.2.

## 7. Afterwards

If the extractor passes, it is offered to partner centres as an optional local tool that fills
the perfusion columns of the template. Whether perfusion volumes improve MTOP is tested later, under
amendment 1, on centres' own data with at least 100 patients in each outcome group and before any
model includes them. That analysis needs its own registered plan.
