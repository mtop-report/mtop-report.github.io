# MTOP data dictionary

Every predictor MTOP uses, the moment it is first used, the registry column it comes from, and
how the registry's own values were coded. The cleaning code is `src/mtop/clean.py`; definitions,
ranges and calculator options are in `config/features.yaml`. The same table is in the app (About
tab).

**How options enter the model.** Predictors with several unordered options (occlusion site,
technique, anaesthesia) become one yes/no column per option (one-hot coding). If the value is
missing, all of that predictor's columns are missing and the trees learn where such patients
belong. Ordered predictors (pre-stroke mRS, mTICI) stay single numbers, so the model can use their
order. Several registry values were grouped into one option where they mean the same for
prognosis or are too rare to stand alone. Values outside a plausible range become missing and are
listed with the patient ID in `reports/data_quality.xlsx` (never published).

## Predictors

| Predictor | Used from | Registry column | In the model ← registry values |
|---|---|---|---|
| Age | Admission | `Age` | years; patients under 18 excluded |
| Sex | Admission | `Sex` | male ← `Muž`; female ← `Žena` |
| Pre-stroke mRS | Admission | `mRS before` | 0–5 |
| NIHSS on admission | Admission | `Admission NIHSS` | 0–42 |
| ASPECTS | Admission | `ASPECTS` | 0–10, middle cerebral artery territory; recorded from 2016. Posterior-circulation strokes mostly have 10 (105 of 125 recorded); the registry has no pc-ASPECTS, and values below 10 in posterior strokes are listed for review |
| Occlusion site | Admission | `Occlusion site` | intracranial ICA / carotid T ← `ACI-Intracranial`, `ACI+ACM`<br>MCA M1 ← `M1-ACM`<br>MCA M2–M3 / ACA ← `M2-ACM`, `M3-ACM`, `ACA`<br>cervical ICA/CCA (± tandem) ← `ACI-odstup`, `ACI-odstup +T typ`, `ACC`, `ACC+ACI+ACM`<br>basilar artery ← `AB`, `AB+AV`<br>vertebral / PCA ← `AV`, `ACP`<br>missing ← `Rekanalizace` (already open at angiography), empty |
| Affected hemisphere | Admission | `Hemisphere` | left ← `levá hemisféra`; right ← `pravá hemisféra`; posterior circulation (`zadní povodí`) is coded 0 like right, the occlusion site carries it |
| Onset to groin puncture | Admission | `Stroke onset to puncture (min)` | minutes from onset or last known well |
| IV thrombolysis | Admission | `IVT` | yes/no ← `ANO`/`NE` |
| Transferred from another hospital | Admission | `Transfer from different hospital` | drip-and-ship ← name of a referring hospital; direct ← `FNO`, `žádné` |
| Glucose on admission | Admission | `Glycemia` | mmol/L |
| Systolic blood pressure | Admission | `Systolic BP` | mmHg; systolic ≤ diastolic → both missing |
| Hypertension | Admission | `Hypertension` | yes/no |
| Diabetes mellitus | Admission | `DM` | yes/no |
| Atrial fibrillation / arrhythmia | Admission | `Arrhythmia` | yes/no |
| Previous stroke | Admission | `CMP before` | yes/no |
| Anticoagulated before stroke | Admission | `Anti-thrombotics before` | yes ← `VKA-Warfarin`, `NOAC-rivaroxaban`, `NOAC-apixaban`, `NOAC-dabigatran`, `LMWH`<br>no ← `žádná` or antiplatelets only (`AT-ASA`, `AC-acylpyrin`, `AP-apotic`, `AT-Clopidogrel`, `AT-dual ASA+Clop`) |
| Final reperfusion (mTICI) | End of MT | `TICI` | 0, 1, 2a, 2b, 3 as recorded; `2` without subgrade is coded between 2a and 2b and is not offered as a calculator input |
| Procedure duration | End of MT | `Procedure duration` | minutes from puncture to recanalisation or end of the procedure |
| First-line technique | End of MT | `Intervention procedure` + `Extraction system` | aspiration ← `ADAPT`, or another MT procedure with device `PENUMBRA`<br>stent retriever ← `MT-stent retriever`, or another MT procedure with `Catch`, `Preset`, `Solitaire`, `TIGERTRIEVER`, `Embotrap`, `Bonnet`, `Merci`<br>combined ← `Solumbra`, or another MT procedure with `PENUMBRA + Catch / Preset / TIGERTRIEVER`<br>no mechanical thrombectomy ← `cPTA`, `PTA`, `Stent`, `Stent+PTA`, `cPTA+stent`, `EKOS+stent`, `LIAT`, `EKOS+LIAT`, `cPTA+LIAT`<br>not analysed ← `PAG` (diagnostic angiography only) |
| Acute stenting | End of MT | `Stent combo` + `Intervention procedure` | yes ← any stent in `Stent combo`, or `MT+stent`, `Stent`, `Stent+PTA`, `cPTA+stent`, `EKOS+stent`; no ← `žádný` |
| Anaesthesia | End of MT | `Anesthesia` | general ← `Celková anestezie`; conscious sedation ← `Analgosedace`; local only ← `bez ARO` |
| NIHSS at 24 h | 24 hours | `NIHSS 24h` | 0–42; `43` (died) and `99` (not assessable) are never used as scores, and such patients leave the later stages |
| Symptomatic intracranial haemorrhage | 24 hours | `SICH` | yes/no |
| NIHSS change in 24 h | 24 hours | computed | NIHSS at 24 h minus NIHSS on admission |
| NIHSS on day 7 | Day 7 | `NIHSS 7d` | 0–42; `43` and `99` as above |
| Cerebral oedema | Day 7 | `Cerebral edema` | yes/no |
| **Outcome** | 90 days | `90-day mRS` | mRS 0–2 = functional independence; mRS 6 = death |

## Our readings, to be confirmed with the registry team

The export does not document these codes; the model uses the reading below.

| Registry value | Read as |
|---|---|
| `NIHSS 24h` / `NIHSS 7d` = 43 | died (70 of the 71 patients coded 43 on day 7 had died by 90 days) |
| `NIHSS 24h` / `NIHSS 7d` = 99 | not assessable |
| `Occlusion site` = `Rekanalizace` | artery already open at angiography; occlusion site unknown |
| `Intervention procedure` = `PAG` | diagnostic angiography only (excluded) |
| `Anesthesia` = `bez ARO` | local anaesthesia only |
| `TICI` = `2` | mTICI 2 recorded without a subgrade |
| `Anti-thrombotics before` = `LMWH` | anticoagulated (the dose is not recorded) |
| `Transfer from different hospital` = `FNO`, `žádné` | direct admission to FNO |
| `Extraction system` | technique by device when the procedure label does not state it (e.g. `PENUMBRA` = aspiration) |

## How the data were cleaned and checked

`python -m mtop clean` applies these rules to all 1,577 registry records. Every change and
every flag is listed with the patient ID in `reports/data_quality.xlsx`, which stays at FNO so the
source can be corrected; flagged values are kept as recorded. The same counts are in the app
(About tab); counts under 10 are shown as <10.

| Check | Records | Action |
|---|---|---|
| Onset-date year typos repaired from the neighbouring patient IDs | 10 | repaired |
| Values outside a plausible range set to missing | 42 | set to missing |
| Recanalisation time before puncture set to missing | 24 | set to missing |
| Systolic pressure not above diastolic: both set to missing | <10 | set to missing |
| Unreadable onset time set to missing | <10 | set to missing |
| Patients under 18 (kept in the registry, not in the models) | <10 | kept, not modelled |
| IVT = no, but a place of IVT is recorded | 19 | kept, listed for review |
| IVT = yes, but the place of IVT is 'none' | <10 | kept, listed for review |
| IVT = no, but an onset-to-IVT time is recorded | 10 | kept, listed for review |
| IVT = no and no intra-arterial lysis, but a thrombolytic is recorded | 12 | kept, listed for review |
| IVT recorded more than 30 min after groin puncture | <10 | kept, listed for review |
| Groin puncture more than 24 h after onset | <10 | kept, listed for review |
| Basilar occlusion recorded with a lateral hemisphere | <10 | kept, listed for review |
| Anterior-circulation occlusion recorded as posterior circulation | <10 | kept, listed for review |
| ASPECTS below 10 in a posterior-circulation stroke | 20 | kept, listed for review |
| Recanalisation recorded more than 30 min after the end of the procedure | 14 | kept, listed for review |
| Coded as died by day 7 (NIHSS 43), but 90-day mRS is not 6 | <10 | kept, listed for review |
| 90-day mRS at least 2 points better than pre-stroke mRS | <10 | kept, listed for review |

## Registry columns MTOP does not use

| Registry column | Why |
|---|---|
| `Date of stroke onset`, `Time of stroke onset` | The year only splits development from validation and sets the recalibration; time of day is not a predictor. |
| `Thrombolytics` | IV thrombolysis yes/no is used; the agent (alteplase, tenecteplase) changed over the years. `Integrilin` is an antiplatelet drug, not a thrombolytic. |
| `IVT in different hospital` | Covered by the admission route (transfer). |
| `IAT` | Intra-arterial lysis is rare and part of the procedure details. |
| `Stroke onset to IVT (min)`, `Stroke onset to recanalization (min)` | Onset to puncture and procedure duration carry the same timing. |
| `non-SICH` | Only symptomatic haemorrhage is used. |
| `Hyperlipidemia`, `Smoking`, `Alcohol abusus`, `Statins before`, `TIA before`, `BMI`, `Heart condition`, `Diastolic BP`, antiplatelets | No added value in this registry: adding twelve such items (with onset hour and posterior circulation) changed the 2021–2025 AUROC by +0.001 for the MTOP model and −0.022 for spline regression, and on the earlier report's 2023–2025 test years by +0.006 and +0.018 (every model re-tuned with the same budget; `python -m mtop reconcile`); several are often missing. |
| `Cholesterol` | Measured later in the stay; whether it is missing tracks early death, which would leak the outcome. |
| `Etiology` | Known only after the work-up, not at the prediction moments. |
| `Decompression surgery` | Happens after the prediction moments. |
