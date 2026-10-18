/* MTOP Registry → "Data and preprocessing": how the registry export becomes the model input.
 * Everything here is computed by the pipeline (src/mtop/preprocessing.py, clean.py, explorer.py):
 * cohort flow, data checks, registry columns and their coding, model inputs and their missing
 * data, outcome availability, distributions and Table 1. Counts under min_cell show as "<min_cell". */
(function () {
  "use strict";
  const A = window.MTOPApp;
  if (!A) return;
  const { h, C, tx, num, L, STAGES, STAGE, FEAT } = A;
  const P = A.DATA.preprocessing;
  if (!P) return;
  const DC = A.DATA.data_checks || { checks: [] };
  const E = A.E;
  const S = A.S;
  S.ppStage = S.ppStage || "admission";
  const MIN = P.min_cell;
  const pct = (v, d = 1) => A.pct100(v, d);
  const cnt = (n, small) => (n === null || n === undefined || small ? `<${MIN}` : num(n));
  const t = () => A.t();

  // ------------------------------------------------------------------ labels
  const REASON = {
    age_under_18: ["aged under 18", "mladší 18 let"],
    mrs90_missing: ["90-day mRS not recorded", "mRS za 90 dní nezaznamenáno"],
    diagnostic_only: ["diagnostic angiography only", "pouze diagnostická angiografie"],
    impossible_time: ["impossible time order", "nemožné pořadí časů"],
  };
  const KEPT = {
    diagnostic_only: ["had a diagnostic angiography only (no endovascular treatment followed)", "měli jen diagnostickou angiografii (bez následné endovaskulární léčby)"],
    impossible_time: ["had an impossible time order; that time was set to missing", "měli nemožné pořadí časů; daný čas byl nastaven jako chybějící"],
    age_unknown: ["had no age recorded", "neměli zaznamenaný věk"],
  };
  const STAGE_REASON = {
    "tici:missing": ["final mTICI not recorded", "výsledné mTICI nezaznamenáno"],
    "nihss_24h:died": ["died before the 24-h assessment (NIHSS 43)", "zemřeli před hodnocením za 24 h (NIHSS 43)"],
    "nihss_24h:not_assessable": ["not assessable at 24 h (NIHSS 99)", "nehodnotitelní za 24 h (NIHSS 99)"],
    "nihss_24h:missing": ["24-h NIHSS not recorded", "NIHSS za 24 h nezaznamenáno"],
    "nihss_7d:died": ["died before day 7 (NIHSS 43)", "zemřeli před 7. dnem (NIHSS 43)"],
    "nihss_7d:not_assessable": ["not assessable on day 7 (NIHSS 99)", "nehodnotitelní 7. den (NIHSS 99)"],
    "nihss_7d:missing": ["day-7 NIHSS not recorded", "NIHSS 7. den nezaznamenáno"],
  };
  const ROLE = {
    predictor: ["Predictor", "Prediktor", "st-live"],
    source: ["Builds a predictor", "Zdroj prediktoru", "st-beta"],
    outcome: ["Outcome", "Výsledek", "st-progress"],
    split: ["Year and weekday", "Rok a den v týdnu", "st-beta"],
    id: ["Identifier, stays at FNO", "Identifikátor, zůstává ve FNO", "st-planned"],
    unused: ["Not used", "Nepoužito", "st-warn"],
  };
  const COLUMN_NOTE = {
    "Patient ID": ["Used only to repair date typos (neighbouring IDs) and to list issues for review at FNO. Never published.", "Slouží jen k opravě překlepů v datech (sousední ID) a k seznamu nálezů ke kontrole ve FNO. Nikdy se nezveřejňuje."],
    "Date of stroke onset": ["Gives the treatment year (temporal split, eras, drift analyses) and weekend onset. The date itself never leaves FNO.", "Dává rok léčby (časové rozdělení, období, analýzy driftu) a vznik o víkendu. Samotné datum FNO neopouští."],
    "Time of stroke onset": ["Hour of onset (or last known well) and onset at night (22–6 h).", "Hodina vzniku (nebo kdy byl naposledy v pořádku) a vznik v noci (22–6 h)."],
    "Stent combo": ["Stent device; with the procedure label it gives “stent placed”.", "Typ stentu; spolu s označením výkonu dává „zaveden stent“."],
    "Extraction system": ["Extraction device; gives the first-line technique where the procedure label does not.", "Extrakční systém; určuje techniku tam, kde ji neurčuje označení výkonu."],
    "IAT": ["Intra-arterial lysis; with the procedure label it gives “intra-arterial thrombolysis”.", "Intraarteriální lýza; spolu s označením výkonu dává „intraarteriální trombolýza“."],
    "Thrombolytics": ["Which thrombolytic was given. IV thrombolysis (yes/no), its place and timing are the predictors; the agent is used only in a consistency check.", "Které trombolytikum bylo podáno. Prediktory jsou IV trombolýza (ano/ne), místo a čas; látka slouží jen ke kontrole konzistence."],
    "Cholesterol": ["Missing in about half of the patients, and missing values go with a much worse outcome at a similar admission status (table below): a model would learn who had blood drawn, not what cholesterol does. Left out, as in the 2026 analysis.", "Chybí asi u poloviny pacientů a chybějící hodnota jde ruku v ruce s mnohem horším výsledkem při podobném stavu při přijetí (tabulka níže): model by se naučil, komu byla odebrána krev, ne vliv cholesterolu. Vynecháno, jako v analýze z roku 2026."],
    "Etiology": ["Established during the hospital stay, after the decisions the calculator supports; not known at admission.", "Stanovuje se během hospitalizace, až po rozhodnutích, která kalkulačka podporuje; při přijetí není známa."],
    "90-day mRS": ["mRS 0–2 (functional independence) and mRS 6 (death) are the two outcomes; the full 0–6 distribution is modelled on top of them.", "mRS 0–2 (funkční nezávislost) a mRS 6 (úmrtí) jsou dva výsledky; celé rozložení 0–6 se modeluje nad nimi."],
    "Admission NIHSS": ["Also the reference for the 24-h change and early neurological improvement.", "Zároveň výchozí hodnota pro změnu za 24 h a časné neurologické zlepšení."],
    "NIHSS 24h": ["43 = died and 99 = not assessable are read as status codes, never as scores.", "43 = úmrtí a 99 = nehodnotitelné se čtou jako stavové kódy, nikdy jako skóre."],
    "NIHSS 7d": ["43 = died and 99 = not assessable are read as status codes, never as scores.", "43 = úmrtí a 99 = nehodnotitelné se čtou jako stavové kódy, nikdy jako skóre."],
    "Systolic BP": ["Also gives the pulse pressure (systolic − diastolic).", "Dává i tlakovou amplitudu (systolický − diastolický)."],
    "Glycemia": ["Also gives hyperglycaemia (> 7.8 mmol/L).", "Dává i hyperglykémii (> 7,8 mmol/l)."],
    "Stroke onset to IVT (min)": ["Also gives the IVT-to-puncture interval; not applicable without IVT.", "Dává i interval od IVT do punkce; bez IVT se neuplatní."],
    "IVT in different hospital": ["Where IV thrombolysis was given: another hospital's name = given elsewhere, before the transfer; “FNO” = given at FNO; “žádné” = not given. With the IVT column it gives three states: none, at FNO, elsewhere.", "Kde byla podána IV trombolýza: název jiné nemocnice = podána jinde, před překladem; „FNO“ = podána ve FNO; „žádné“ = nepodána. Se sloupcem IVT dává tři stavy: žádná, ve FNO, jinde."],
    "Occlusion site": ["Tandem = a cervical ICA/CCA lesion plus an intracranial occlusion (ACI+ACM, ACI-odstup +T typ, ACC+ACI+ACM); it is a category of its own, separate from cervical ICA/CCA and from intracranial ICA. Every recorded value keeps its own detailed category; for the models, codes too rare to estimate join the group they belong to clinically, and the ACA is merged with none (see the coding).",
      "Tandem = léze krční ACI/ACC spolu s intrakraniálním uzávěrem (ACI+ACM, ACI-odstup +T typ, ACC+ACI+ACM); je to samostatná kategorie, oddělená od krční ACI/ACC i od intrakraniální ACI. Každá zaznamenaná hodnota má vlastní podrobnou kategorii; pro modely se kódy příliš vzácné na odhad přiřazují ke skupině, kam klinicky patří, a ACA se neslučuje s žádnou (viz kódování)."],
    "Intervention procedure": ["“kombinovaný výkon” = two or more techniques in one procedure (its own category); Solumbra = aspiration and a stent retriever together; PAG = diagnostic angiography only.", "„kombinovaný výkon“ = dvě a více technik v jednom výkonu (vlastní kategorie); Solumbra = aspirace a stentriever současně; PAG = pouze diagnostická angiografie."],
    "Extraction system": ["The device decides the technique only where the procedure label does not: PENUMBRA = aspiration; a stent retriever alone = stent retriever; PENUMBRA + a stent retriever = Solumbra.", "Zařízení určuje techniku jen tam, kde ji neurčuje označení výkonu: PENUMBRA = aspirace; samotný stentriever = stentriever; PENUMBRA + stentriever = Solumbra."],
  };
  const CLEAN_NAME = {
    onset_date: ["Onset date", "Datum vzniku"], onset_time: ["Onset time", "Čas vzniku"], "sbp/dbp": ["Blood pressure", "Krevní tlak"],
    heart: ["Heart condition", "Srdeční onemocnění"], agent: ["Thrombolytic agent", "Trombolytikum"], device: ["Extraction device", "Extrakční systém"],
    procedure: ["Procedure", "Výkon"], side: ["Hemisphere", "Hemisféra"], cholesterol: ["Cholesterol", "Cholesterol"], mrs90: ["90-day mRS", "mRS za 90 dní"],
    sex: ["Sex", "Pohlaví"], etiology: ["Etiology", "Etiologie"], iat: ["Intra-arterial lysis", "Intraarteriální lýza"],
    occlusion_detail: ["Occlusion site, detailed", "Místo okluze, podrobně"],
  };
  // detailed occlusion codes (every registry value apart) and how the models currently group them
  const OCC_DETAIL = {
    M1: ["MCA M1", "ACM M1"], M2: ["MCA M2", "ACM M2"], M3: ["MCA M3", "ACM M3"], ACA: ["ACA", "ACA"], ICA: ["Intracranial ICA", "Intrakraniální ACI"],
    tandem: ["Tandem (cervical ICA + intracranial)", "Tandem (krční ACI + intrakraniální)"],
    tandem_cca: ["CCA + ICA + MCA (complete carotid occlusion with intracranial extension)", "ACC + ACI + ACM (úplný uzávěr karotidy s intrakraniálním pokračováním)"],
    cervical_ICA: ["Cervical ICA", "Krční ACI"], CCA: ["Common carotid (CCA)", "A. carotis communis (ACC)"],
    basilar: ["Basilar", "Basilaris"], basilar_vertebral: ["Basilar + vertebral", "Basilaris + vertebrální"], vertebral: ["Vertebral", "Vertebrální"], PCA: ["PCA", "ACP"],
    missing: ["Not classified (artery already open)", "Neklasifikováno (tepna již průchodná)"],
  };
  const KIND = {
    repaired: ["Repaired", "Opraveno"],
    missing: ["Set to missing, patient kept", "Nastaveno jako chybějící, pacient ponechán"],
    kept: ["Kept in the registry, not analysed", "Ponecháno v registru, neanalyzováno"],
    flag: ["Flagged for review at the source, value kept", "Označeno ke kontrole u zdroje, hodnota ponechána"],
  };
  const TECH = {
    aspiration: ["Aspiration (ADAPT)", "Aspirace (ADAPT)"], stent_retriever: ["Stent retriever", "Stentriever"],
    solumbra: ["Solumbra (aspiration + stent retriever)", "Solumbra (aspirace + stentriever)"],
    combined: ["Combined procedure (two or more techniques)", "Kombinovaný výkon (dvě a více technik)"],
    none: ["No thrombectomy (stent, angioplasty or IA lysis only)", "Bez trombektomie (jen stent, angioplastika nebo IA lýza)"],
    diagnostic: ["Diagnostic angiography only", "Pouze diagnostická angiografie"], missing: ["Not recorded", "Nezaznamenáno"],
  };
  // the procedure label's coding: its technique token plus any additions (stent, angioplasty, IA lysis)
  const PROC = {
    aspiration: ["Thrombectomy: aspiration (ADAPT)", "Trombektomie: aspirace (ADAPT)"],
    stent_retriever: ["Thrombectomy: stent retriever", "Trombektomie: stentriever"],
    solumbra: ["Thrombectomy: Solumbra (aspiration + stent retriever)", "Trombektomie: Solumbra (aspirace + stentriever)"],
    combined: ["Thrombectomy: combined procedure (two or more techniques)", "Trombektomie: kombinovaný výkon (dvě a více technik)"],
    device: ["Thrombectomy: technique from the extraction system", "Trombektomie: technika podle extrakčního systému"],
    none: ["No thrombectomy in the label (technique from the extraction system if a device is recorded)", "Bez trombektomie v označení (technika podle extrakčního systému, je-li uveden)"],
    diagnostic: ["Diagnostic angiography only", "Pouze diagnostická angiografie"],
    stent: ["+ stent", "+ stent"], angioplasty: ["+ angioplasty", "+ angioplastika"], ia_lysis: ["+ intra-arterial lysis", "+ intraarteriální lýza"],
  };
  const TYPE = { numeric: ["number", "číslo"], binary: ["yes / no", "ano / ne"], ordinal: ["ordered", "pořadové"], categorical: ["categories", "kategorie"], derived: ["derived", "odvozené"] };
  const lab2 = (pair) => (pair ? tx(pair[0], pair[1]) : "");
  const featLabel = (k) => (FEAT[k] ? L(FEAT[k].label) : CLEAN_NAME[k] ? lab2(CLEAN_NAME[k]) : k);
  const stageName = (s) => (STAGE[s] ? L(STAGE[s].title) : s);

  // ------------------------------------------------------------------ small pieces
  function head(title, sub, id) {
    return h("div", { class: "chart-head", id: id || null }, h("div", null, h("div", { class: "chart-title", text: title }), sub ? h("div", { class: "chart-sub", text: sub }) : null));
  }
  function table(cols, rows, cls) {
    return h("div", { class: "table-scroll" }, h("table", { class: `data ${cls || ""}` },
      h("thead", null, h("tr", null, cols.map((c) => h("th", { scope: "col", text: c })))),
      h("tbody", null, rows)));
  }
  function kpi(label, value, sub) {
    return h("div", { class: "kpi" }, h("div", { class: "l", text: label }), h("div", { class: "v", text: value }), sub ? h("div", { class: "s", text: sub }) : null);
  }
  function roleChip(role) {
    const r = ROLE[role] || [role, role, "st-planned"];
    return h("span", { class: `status ${r[2]}` }, h("i", { "aria-hidden": "true" }), tx(r[0], r[1]));
  }

  // ------------------------------------------------------------------ 1. cohort flow
  function flowCard() {
    const F = P.flow;
    const card = h("div", { class: "card card-pad block", id: "pp-flow" });
    card.appendChild(head(tx("From registry to analysis", "Od registru k analýze"),
      tx("Patient flow, in the style of a CONSORT diagram", "Tok pacientů ve stylu diagramu CONSORT")));
    const box = (n, text, cls) => h("div", { class: `cf-box ${cls || ""}` }, h("b", { text: num(n) }), h("span", { text }));
    const fl = h("div", { class: "cflow" });
    const steps = F.steps;
    fl.appendChild(h("div", { class: "cf-main" }, box(steps[0].n, tx(`records in the registry export, ${E.period[0]}–${E.period[1]}`, `záznamů v exportu registru, ${E.period[0]}–${E.period[1]}`))));
    for (let i = 1; i < steps.length; i++) {
      const st = steps[i];
      fl.appendChild(h("div", { class: "cf-side" }, h("div", { class: "cf-box out" }, h("b", { text: num(st.removed) }),
        h("span", { text: tx(`excluded: ${lab2(REASON[st.reason]) || st.reason}`, `vyřazeno: ${lab2(REASON[st.reason]) || st.reason}`) }))));
      const last = i === steps.length - 1;
      const main = box(st.n, last ? tx("analysed: adults with a recorded 90-day mRS", "analyzováno: dospělí se zaznamenaným mRS za 90 dní") : tx("adults", "dospělí"), last ? "cf-key" : "");
      if (last && st.kept) {
        const ul = h("ul", { class: "cf-kept" });
        for (const [k, v] of Object.entries(st.kept)) if (v) ul.appendChild(h("li", { text: `${num(v)} ${lab2(KEPT[k]) || k}` }));
        main.appendChild(h("small", { text: tx("Kept, among them:", "Ponecháni, mezi nimi:") }));
        main.appendChild(ul);
      }
      fl.appendChild(h("div", { class: "cf-main" }, main));
    }
    const sp = F.splits.primary;
    const part = (g, title) => h("div", { class: "cf-box" }, h("span", { class: "cf-title", text: title }), h("b", { text: num(g.n) }),
      h("span", { text: tx(`mRS 0–2: ${num(g.good)} · deaths: ${num(g.death)}`, `mRS 0–2: ${num(g.good)} · úmrtí: ${num(g.death)}`) }));
    fl.appendChild(h("div", { class: "cf-split" },
      part(sp.dev, tx(`Development ${sp.dev.years[0]}–${sp.dev.years[1]}`, `Vývoj ${sp.dev.years[0]}–${sp.dev.years[1]}`)),
      part(sp.test, tx(`Temporal validation ${sp.test.years[0]}–${sp.test.years[1]}`, `Časová validace ${sp.test.years[0]}–${sp.test.years[1]}`))));
    card.appendChild(fl);
    const se = F.splits.sensitivity;
    if (se) card.appendChild(h("p", { class: "note", text: tx(
      `Primary split as in the 2026 analysis: the models are developed on patients treated up to ${sp.dev.years[1]} and tested on the ${num(sp.test.n)} treated from ${sp.test.years[0]}. Sensitivity analysis: development ${se.dev.years[0]}–${se.dev.years[1]} (n = ${num(se.dev.n)}) and validation ${se.test.years[0]}–${se.test.years[1]} (n = ${num(se.test.n)}; mRS 0–2 ${num(se.test.good)}, deaths ${num(se.test.death)}), which roughly doubles the test events. The published calculator is refitted on all ${num(steps[steps.length - 1].n)} patients.`,
      `Hlavní rozdělení jako v analýze z roku 2026: modely se vyvíjejí na pacientech léčených do roku ${sp.dev.years[1]} a testují na ${num(sp.test.n)} pacientech léčených od roku ${sp.test.years[0]}. Citlivostní analýza: vývoj ${se.dev.years[0]}–${se.dev.years[1]} (n = ${num(se.dev.n)}) a validace ${se.test.years[0]}–${se.test.years[1]} (n = ${num(se.test.n)}; mRS 0–2 ${num(se.test.good)}, úmrtí ${num(se.test.death)}), což zhruba zdvojnásobí testovací události. Zveřejněná kalkulačka je přetrénována na všech ${num(steps[steps.length - 1].n)} pacientech.`) }));
    // flow table (as in the earlier pipeline's cohort flow table)
    const rows = steps.map((st, i) => h("tr", null, h("th", { scope: "row", text: String(i + 1) }),
      h("td", { text: i === 0 ? tx("Registry export", "Export registru") : st.step === "adults" ? tx("Adults (age ≥ 18 or not recorded)", "Dospělí (věk ≥ 18 nebo nezaznamenán)") : tx("90-day mRS recorded", "Zaznamenáno mRS za 90 dní") }),
      h("td", { text: st.removed ? `−${num(st.removed)}` : "" }), h("td", { text: num(st.n) })));
    rows.push(h("tr", null, h("th", { scope: "row", text: String(steps.length + 1) }), h("td", { text: tx(`Development, treated ${sp.dev.years[0]}–${sp.dev.years[1]}`, `Vývoj, léčeni ${sp.dev.years[0]}–${sp.dev.years[1]}`) }), h("td", { text: "" }), h("td", { text: num(sp.dev.n) })));
    rows.push(h("tr", null, h("th", { scope: "row", text: String(steps.length + 2) }), h("td", { text: tx(`Temporal validation, treated ${sp.test.years[0]}–${sp.test.years[1]}`, `Časová validace, léčeni ${sp.test.years[0]}–${sp.test.years[1]}`) }), h("td", { text: "" }), h("td", { text: num(sp.test.n) })));
    const det = h("details", { class: "template" }, h("summary", { text: tx("Flow as a table", "Tok jako tabulka") }),
      table([tx("Step", "Krok"), tx("Description", "Popis"), tx("Removed", "Odebráno"), tx("Patients", "Pacienti")], rows, "compact"));
    card.appendChild(det);
    return card;
  }

  function stageCard() {
    const F = P.flow;
    const card = h("div", { class: "card card-pad block", id: "pp-stages" });
    card.appendChild(head(tx("Who enters each stage's model", "Kdo vstupuje do modelu každé fáze"),
      tx("A stage needs the information that defines it; nothing else is required", "Fáze potřebuje jen údaje, které ji definují; nic dalšího se nevyžaduje")));
    const rows = F.stages.map((r) => {
      const reasons = Object.entries(r.reasons || {}).map(([k, v]) => `${lab2(STAGE_REASON[k]) || k} ${v === null ? `<${MIN}` : num(v)}`);
      return h("tr", null, h("th", { scope: "row", text: `${A.idx(r.stage) + 1} · ${stageName(r.stage)}` }),
        h("td", { class: "num", text: num(r.n) }), h("td", { class: "num", text: num(r.dev) }), h("td", { class: "num", text: num(r.test) }),
        h("td", { class: "num", text: num(r.good) }), h("td", { class: "num", text: num(r.death) }),
        h("td", { text: r.excluded ? `${num(r.excluded)}: ${reasons.join("; ")}` : tx("none", "nikdo") }));
    });
    card.appendChild(table([tx("Stage", "Fáze"), tx("Patients", "Pacienti"), tx("Development", "Vývoj"), tx("Validation", "Validace"), "mRS 0–2", tx("Deaths", "Úmrtí"), tx("Not in this stage (a patient can have several reasons)", "Mimo tuto fázi (pacient může mít více důvodů)")], rows, "txt"));
    card.appendChild(h("p", { class: "note", text: tx(
      "The procedure stage needs the final mTICI; the 24-h stage the mTICI and a 24-h NIHSS score; the day-7 stage the mTICI and a day-7 NIHSS score (not the 24-h one). Patients who died before an assessment cannot be scored at that stage; their estimate is the one from the previous stage. Since 2021 many follow-up scores are missing, so the later stages are validated on fewer patients.",
      "Fáze výkonu vyžaduje výsledné mTICI; fáze 24 h mTICI a skóre NIHSS za 24 h; fáze 7. den mTICI a NIHSS 7. dne (ne NIHSS za 24 h). Pacienty, kteří zemřeli před hodnocením, nelze v dané fázi hodnotit; platí pro ně odhad z předchozí fáze. Od roku 2021 mnoho skóre ze sledování chybí, a pozdější fáze jsou proto validovány na menším počtu pacientů.") }));
    return card;
  }

  // ------------------------------------------------------------------ 2. cleaning
  function checksCard() {
    const card = h("div", { class: "card card-pad block", id: "pp-checks" });
    card.appendChild(head(tx("What the cleaning changed", "Co čištění změnilo"),
      tx(`Every rule applied to the ${num(DC.records || P.records)} registry records; patients affected`, `Každé pravidlo použité na ${num(DC.records || P.records)} záznamů registru; počet dotčených pacientů`)));
    const kinds = ["repaired", "missing", "kept", "flag"];
    const rows = [];
    for (const k of kinds) {
      const list = DC.checks.filter((c) => c.kind === k);
      if (!list.length) continue;
      rows.push(h("tr", { class: "sep" }, h("th", { colspan: 3, text: lab2(KIND[k]) })));
      for (const c of list) {
        const cols = (c.by_column || []).filter((b) => b.column).map((b) => `${featLabel(b.column)} ${cnt(b.n, b.small)}`);
        rows.push(h("tr", null, h("th", { scope: "row", text: L(c.label) }), h("td", { class: "num", text: cnt(c.n, c.small) }),
          h("td", { class: "muted", text: cols.length > 1 ? cols.join(" · ") : "" })));
      }
    }
    card.appendChild(table([tx("Check", "Kontrola"), tx("Patients", "Pacienti"), tx("By variable", "Podle proměnné")], rows, "txt checks-t"));
    const R = P.ranges || {};
    const rng = Object.entries(R).map(([k, [a, b]]) => `${featLabel(k)} ${num(a, a % 1 ? 1 : 0)}–${num(b, b % 1 ? 1 : 0)}`);
    card.appendChild(h("p", { class: "note", text: tx(
      `Rules: nothing is dropped silently and nothing is guessed. Implausible values become missing (a likely correction, such as a misplaced decimal point, is only suggested for the source file). The only automatic repair is a one-year typo in the onset date, when the date fits exactly between the neighbouring patient IDs. Cross-field checks keep the value and list the record for review at FNO; the list with patient IDs stays at FNO. Plausible ranges: ${rng.join("; ")}; NIHSS 0–42. Patient IDs are unique (checked).`,
      `Pravidla: nic se neodstraňuje potichu a nic se nehádá. Nevěrohodné hodnoty se stávají chybějícími (pravděpodobná oprava, např. posunutá desetinná čárka, se jen navrhne pro zdrojový soubor). Jedinou automatickou opravou je překlep o jeden rok v datu vzniku, pokud datum přesně zapadne mezi sousední ID pacientů. Křížové kontroly hodnotu ponechají a záznam zařadí ke kontrole ve FNO; seznam s ID pacientů zůstává ve FNO. Věrohodné rozsahy: ${rng.join("; ")}; NIHSS 0–42. ID pacientů jsou jedinečná (ověřeno).`) }));
    return card;
  }

  const CODES = [
    ["NIHSS 24h / 7d = 43", ["died before the assessment: a status, never a score", "úmrtí před hodnocením: stav, nikdy skóre"], ["70 of the 71 patients coded 43 on day 7 had died by 90 days. Used as a score, 43 would hand the outcome to the model.", "Ze 71 pacientů s kódem 43 v 7. den jich 70 do 90 dní zemřelo. Jako skóre by 43 prozradilo modelu výsledek."], true],
    ["NIHSS 24h / 7d = 99", ["not assessable (for example sedated or intubated)", "nehodnotitelné (např. sedace nebo intubace)"], ["Status, never a score.", "Stav, nikdy skóre."], true],
    ["TICI = 2", ["reperfusion grade 2 without the a/b subgrade", "reperfuze stupně 2 bez podstupně a/b"], ["Ordered between 2a and 2b; not counted as 2b–3. Not offered as a calculator input.", "Řazeno mezi 2a a 2b; nepočítá se do 2b–3. V kalkulačce se nenabízí."], false],
    ["Occlusion site = Rekanalizace", ["artery already open at angiography", "tepna při angiografii již průchodná"], ["No occlusion to classify: occlusion site missing, patient kept.", "Není co klasifikovat: místo uzávěru chybí, pacient ponechán."], true],
    ["Procedure = PAG", ["diagnostic angiography only", "pouze diagnostická angiografie"], ["Kept, with its own technique category: before the procedure nobody knows that no treatment will follow.", "Ponecháno, s vlastní kategorií techniky: před výkonem nikdo neví, že léčba nebude následovat."], false],
    ["Procedure = Stent, PTA, cPTA, LIAT, EKOS …", ["endovascular treatment without thrombectomy", "endovaskulární léčba bez trombektomie"], ["Technique “none” plus the stent, angioplasty and intra-arterial lysis flags.", "Technika „žádná“ plus příznaky stentu, angioplastiky a intraarteriální lýzy."], false],
    ["Procedure = kombinovaný výkon", ["combined procedure: two or more of the techniques", "kombinovaný výkon: dvě a více technik"], ["Its own category, not merged with Solumbra or any single technique. Almost all have a thrombectomy device recorded, and most a stent. Stenting is read from the stent column and intra-arterial lysis from the IAT column; angioplasty has no column of its own, so for these patients it counts as not recorded (no).", "Samostatná kategorie, nesloučená se Solumbrou ani s žádnou jednotlivou technikou. Téměř všichni mají uveden extrakční systém a většina stent. Stent se čte ze sloupce stentu a intraarteriální lýza ze sloupce IAT; angioplastika vlastní sloupec nemá, proto se u těchto pacientů počítá jako nezaznamenaná (ne)."], false],
    ["Procedure = Solumbra", ["aspiration and a stent retriever together", "aspirace a stentriever současně"], ["Its own technique category (not the combined procedure).", "Vlastní kategorie techniky (nejde o kombinovaný výkon)."], false],
    ["Extraction system = PENUMBRA + Catch / Preset / TIGERTRIEVER", ["Solumbra technique", "technika Solumbra"], ["Used where the procedure label does not name the technique (for example MT+stent).", "Použito tam, kde označení výkonu techniku neuvádí (např. MT+stent)."], false],
    ["Occlusion site = ACI+ACM, ACI-odstup +T typ", ["tandem occlusion", "tandemový uzávěr"], ["Cervical ICA lesion plus intracranial occlusion. Tandem is its own category, separate from cervical ICA / CCA and from intracranial ICA.", "Léze krční ACI s intrakraniálním uzávěrem. Tandem je samostatná kategorie, oddělená od krční ACI / ACC i od intrakraniální ACI."], false],
    ["Occlusion site = ACC+ACI+ACM", ["tandem: complete carotid occlusion reaching the MCA", "tandem: úplný uzávěr karotidy až do ACM"], ["Counted as tandem; its own detailed code in the data, so it can be analysed apart later.", "Počítá se jako tandem; v datech má vlastní podrobný kód, aby šel později analyzovat zvlášť."], false],
    ["Occlusion site = ACI-odstup, ACC", ["cervical ICA / CCA occlusion without an intracranial occlusion", "uzávěr krční ACI / ACC bez intrakraniálního uzávěru"], ["One category; ACC keeps its own detailed code in the data.", "Jedna kategorie; ACC má v datech vlastní podrobný kód."], false],
    ["Occlusion site = AB+AV", ["basilar plus vertebral artery", "a. basilaris spolu s a. vertebralis"], ["Its own category, also in the models (not merged with basilar).", "Samostatná kategorie, i v modelech (nesloučena s a. basilaris)."], false],
    ["Occlusion site = ACA", ["anterior cerebral artery", "a. cerebri anterior"], ["Its own category, merged with no other site. Too few patients for a model estimate yet: the models do not use the site for these patients, and the calculator gives no estimate for an ACA occlusion.", "Samostatná kategorie, nesloučená s žádným jiným místem. Na odhad v modelu zatím příliš málo pacientů: modely u těchto pacientů místo okluze nepoužívají a kalkulačka pro uzávěr ACA odhad nedává."], false],
    ["Anaesthesia = bez ARO", ["local anaesthesia, no anaesthesiologist", "lokální anestezie, bez anesteziologa"], ["“Without the anaesthesiology department”.", "„Bez ARO“."], true],
    ["Transfer = FNO or žádné", ["direct admission to FNO", "přímý příjem do FNO"], ["A referring hospital's name means a transfer; hospital names are never published.", "Název odesílající nemocnice znamená překlad; názvy nemocnic se nezveřejňují."], true],
    ["IVT in different hospital = a hospital's name", ["IV thrombolysis given elsewhere, not at FNO", "IV trombolýza podaná jinde, ne ve FNO"], ["“FNO” = given at FNO; “žádné” = not given.", "„FNO“ = podána ve FNO; „žádné“ = nepodána."], false],
    ["Heart condition = KS. Kardio.", ["pacemaker", "kardiostimulátor"], ["Grouped with the other structural conditions (valve disease, cardiomyopathy, PFO, left-atrial thrombus).", "Sloučeno s ostatními strukturálními onemocněními (chlopenní vada, kardiomyopatie, PFO, trombus v levém oušku)."], true],
    ["Thrombolytics = Integrilin", ["eptifibatide, an antiplatelet drug", "eptifibatid, antiagregans"], ["The agent column is not a predictor.", "Sloupec látky není prediktorem."], false],
    ["Time of stroke onset", ["onset, or the time last known well", "vznik, nebo kdy byl pacient naposledy v pořádku"], ["Times are minutes from this moment.", "Časy jsou minuty od tohoto okamžiku."], true],
    ["Date out of sequence by one year", ["a year typo", "překlep v roce"], ["Repaired only if the date then fits between the neighbouring patient IDs (IDs are assigned in order of admission).", "Opraveno jen tehdy, zapadne-li datum pak mezi sousední ID pacientů (ID se přidělují v pořadí přijetí)."], true],
  ];
  function codesCard() {
    const card = h("div", { class: "card card-pad block", id: "pp-codes" });
    card.appendChild(head(tx("How registry codes were read", "Jak byly čteny kódy registru"),
      tx("* our reading, to be confirmed with the registry team", "* náš výklad, k potvrzení s týmem registru")));
    const rows = CODES.map(([code, read, why, ours]) => h("tr", null, h("th", { scope: "row", class: "mono", text: code }),
      h("td", { text: `${tx(read[0], read[1])}${ours ? " *" : ""}` }), h("td", { class: "muted", text: tx(why[0], why[1]) })));
    card.appendChild(table([tx("In the registry", "V registru"), tx("Read as", "Čteno jako"), tx("Why, and what follows", "Proč a co z toho plyne")], rows, "txt codes-t"));
    return card;
  }

  // ------------------------------------------------------------------ 3. variables
  function variablesCard() {
    const card = h("div", { class: "card card-pad block", id: "pp-columns" });
    const V = P.variables;
    const nPred = V.filter((v) => v.role === "predictor").length;
    card.appendChild(head(tx("Registry columns and what became of them", "Sloupce registru a co se z nich stalo"),
      tx(`${V.length} columns in the export · ${nPred} are predictors, ${V.filter((v) => v.role === "source").length} build predictors, ${V.filter((v) => v.role === "unused").length} are not used`,
        `${V.length} sloupců v exportu · ${nPred} jsou prediktory, ${V.filter((v) => v.role === "source").length} tvoří prediktory, ${V.filter((v) => v.role === "unused").length} se nepoužívají`)));
    const list = h("div", { class: "varmap" });
    for (const v of V) {
      const vars = v.variables.map((k) => featLabel(k)).filter((x, i, a) => a.indexOf(x) === i);
      const note = COLUMN_NOTE[v.column];
      const top = h("div", { class: "vm-top" },
        h("b", { class: "mono", text: v.column }),
        roleChip(v.role),
        v.stage ? h("span", { class: "vm-stage", text: stageName(v.stage) }) : null,
        h("span", { class: "vm-miss", text: v.missing_pct === null || v.missing_pct === undefined ? "" : tx(`${pct(v.missing_pct)} empty`, `${pct(v.missing_pct)} prázdných`) }));
      const body = h("div", { class: "vm-body" });
      if (vars.length) body.appendChild(h("div", { class: "vm-vars" }, h("span", { class: "muted", text: "→ " }), vars.join(" · ")));
      if (note) body.appendChild(h("p", { class: "note", text: tx(note[0], note[1]) }));
      const item = h("div", { class: "vm-item" }, top, body);
      if (v.coding && v.coding.length) {
        const det = h("details", { class: "vm-coding" }, h("summary", { text: tx(`Coding of ${v.coding.length} recorded values`, `Kódování ${v.coding.length} zaznamenaných hodnot`) }));
        const toLabel = (k, to) => {
          if (k === "Intervention procedure") return String(to).split("+").map((p) => lab2(PROC[p]) || p).join("; ");
          const f = FEAT[{ "Occlusion site": "occlusion", "Anti-thrombotics before": "antithrombotic", "Heart condition": "heart3", Anesthesia: "anesthesia", "Extraction system": "technique" }[k]];
          const o = f && f.options ? f.options.find((o2) => String(o2.value) === String(to)) : null;
          return o ? L(o.label).replace(/ · .*/, "") : TECH[to] ? lab2(TECH[to]) : String(to);
        };
        if (v.column === "Occlusion site") {
          const occF = FEAT.occlusion;
          const minCat = P.min_category || 10;
          const modelLabel = (m) => {
            if (!m) return tx(`not used by the models yet (fewer than ${minCat} patients)`, `modely zatím nepoužívají (méně než ${minCat} pacientů)`);
            const o = occF && occF.options.find((o2) => o2.value === m);
            return o ? L(o.label).replace(/ · .*/, "") : String(m);
          };
          det.appendChild(table([tx("Recorded value", "Zaznamenaná hodnota"), tx("Detailed category", "Podrobná kategorie"), tx("In the models", "V modelech"), "n"],
            v.coding.map((c) => h("tr", null, h("th", { scope: "row", class: "mono", text: c.value }), h("td", { text: lab2(OCC_DETAIL[c.to]) || c.to }),
              h("td", { text: c.to === "missing" ? tx("missing", "chybí") : modelLabel(c.model) }),
              h("td", { class: "num", text: c.present ? cnt(c.n, c.small) : "0" }))), "compact txt"));
          det.appendChild(h("p", { class: "note", text: tx(
            `Every recorded value keeps its own detailed category. A site needs at least ${minCat} analysed patients before the models can estimate it (and before an estimate built on it may be published). The categories below that size are given to the group they belong to clinically: M3 to MCA M2 / M3 (the same artery), CCA + ICA + MCA to tandem, CCA to cervical ICA / CCA, vertebral and PCA together. The ACA belongs to none of the modelled groups, so it is not merged with any: the models do not use the site for ACA patients, and the calculator gives no estimate for an ACA occlusion until enough patients are recorded.`,
            `Každá zaznamenaná hodnota má vlastní podrobnou kategorii. Místo okluze musí mít alespoň ${minCat} analyzovaných pacientů, aby je modely mohly odhadnout (a aby se odhad na něm založený směl zveřejnit). Menší kategorie jsou přiřazeny ke skupině, kam klinicky patří: M3 k ACM M2 / M3 (stejná tepna), ACC + ACI + ACM k tandemu, ACC ke krční ACI / ACC, vertebrální a ACP spolu. ACA nepatří do žádné z modelovaných skupin, proto se s žádnou neslučuje: modely u pacientů s uzávěrem ACA místo okluze nepoužívají a kalkulačka pro uzávěr ACA nedává odhad, dokud nebude zaznamenáno dost pacientů.`) }));
        } else {
          det.appendChild(table([tx("Recorded value", "Zaznamenaná hodnota"), tx("Model value", "Hodnota v modelu"), "n"],
            v.coding.map((c) => h("tr", null, h("th", { scope: "row", class: "mono", text: c.value }), h("td", { text: toLabel(v.column, c.to) }),
              h("td", { class: "num", text: c.present ? cnt(c.n, c.small) : "0" }))), "compact"));
        }
        item.appendChild(det);
      }
      list.appendChild(item);
    }
    card.appendChild(list);
    return card;
  }

  // ------------------------------------------------------------------ 4. model inputs
  function inputsCard(root) {
    const card = h("div", { class: "card card-pad block", id: "pp-inputs" });
    const st = S.ppStage;
    const preds = P.predictors.filter((p) => p.stage === st);
    const counts = STAGES.map((s2) => P.predictors.filter((p) => p.stage === s2).length);
    card.appendChild(head(tx("Model inputs and their missing data", "Vstupy modelů a jejich chybějící data"),
      tx(`${P.predictors.length} inputs: ${counts.map((c, i) => `${c} ${tx(["before the procedure", "at its end", "at 24 h", "on day 7"][i], ["před výkonem", "na jeho konci", "za 24 h", "7. den"][i])}`).join(", ")}; every later stage also uses all earlier inputs`,
        `${P.predictors.length} vstupů: ${counts.map((c, i) => `${c} ${["před výkonem", "na jeho konci", "za 24 h", "7. den"][i]}`).join(", ")}; každá další fáze používá i všechny dřívější vstupy`)));
    const seg = h("div", { class: "mini-seg wrap", role: "group", "aria-label": tx("Stage", "Fáze") },
      STAGES.map((s2) => h("button", { type: "button", "aria-pressed": String(st === s2), onclick: () => { S.ppStage = s2; rerender(root, "pp-inputs"); } }, A.stageLabel(s2))));
    card.appendChild(h("div", { class: "controls" }, h("span", { text: tx("Added at", "Přidáno ve fázi") }), seg));
    const handling = (p) => (p.handling === "structural" ? tx("Not applicable without IVT", "Bez IVT se neuplatní")
      : p.handling === "derived" ? tx("Computed from its inputs", "Počítáno ze vstupů")
        : tx("May be missing", "Může chybět"));
    const mono = (m) => (m > 0 ? "+" : m < 0 ? "−" : "");
    const sp = P.flow.splits.primary;
    const rows = preds.map((p) => h("tr", null,
      h("th", { scope: "row" }, featLabel(p.key), h("span", { class: "mono sub", text: ` ${p.key}` })),
      h("td", { text: lab2(TYPE[p.type]) || p.type }),
      h("td", { class: "num", text: mono(p.monotone) }),
      h("td", { class: "num", text: pct(p.missing_pct) }), h("td", { class: "num", text: pct(p.dev_pct) }), h("td", { class: "num", text: pct(p.test_pct) }),
      h("td", { class: "num", text: p.good_if_missing === undefined ? "–" : `${pct(p.good_if_missing, 0)} / ${pct(p.good_if_known, 0)}` }),
      h("td", { text: handling(p) + (p.optional ? tx(" · optional in the calculator", " · v kalkulačce nepovinné") : "") })));
    card.appendChild(table([tx("Input", "Vstup"), tx("Type", "Typ"), tx("Monotone", "Monotónní"), tx("Missing", "Chybí"),
      tx(`Dev ${sp.dev.years[0]}–${sp.dev.years[1]}`, `Vývoj ${sp.dev.years[0]}–${sp.dev.years[1]}`), tx(`Test ${sp.test.years[0]}–${sp.test.years[1]}`, `Test ${sp.test.years[0]}–${sp.test.years[1]}`),
      tx("mRS 0–2 if missing / recorded", "mRS 0–2 při chybění / záznamu"), tx("Handling", "Zpracování")], rows));
    card.appendChild(h("p", { class: "note", text: tx(
      `Missing: share of the patients of this stage's cohort without the value. Missing values are never filled in before modelling: the regressions use the development median plus a missing-value indicator, the tree models learn which branch missing values take, and the calculator averages its estimate over registry values for an input you leave unknown. Monotone: in LightGBM a higher value can only raise (+) or only lower (−) the chance of mRS 0–2, as clinically expected. “mRS 0–2 if missing / recorded” shows that missing data are not random: patients without a value often fared differently. Shown when both groups have at least ${MIN} patients.`,
      `Chybí: podíl pacientů kohorty této fáze bez hodnoty. Chybějící hodnoty se před modelováním nikdy nedoplňují: regrese používají medián vývojových dat a indikátor chybění, stromové modely se učí, kterou větví chybějící hodnoty půjdou, a kalkulačka pro vstup ponechaný jako neznámý zprůměruje odhad přes hodnoty z registru. Monotónní: v LightGBM může vyšší hodnota šanci na mRS 0–2 jen zvýšit (+) nebo jen snížit (−), jak se klinicky očekává. „mRS 0–2 při chybění / záznamu“ ukazuje, že data nechybí náhodně: pacienti bez hodnoty často dopadli jinak. Zobrazeno, má-li každá skupina alespoň ${MIN} pacientů.`) }));
    // missingness by era
    const eraRows = preds.filter((p) => p.type !== "derived" || p.handling === "structural");
    const fh = C.figure(card, {
      title: tx("Missing data by treatment era", "Chybějící data podle období léčby"),
      sub: tx("Share of patients without the value (%); darker = more missing. What the registry recorded changed over time.", "Podíl pacientů bez hodnoty (%); tmavší = více chybí. Co registr zaznamenával, se v čase měnilo."),
      table: { columns: [tx("Input", "Vstup")].concat(P.eras), rows: eraRows.map((p) => [featLabel(p.key)].concat(P.eras.map((e) => (p.by_era[e] === null ? "–" : pct(p.by_era[e], 0))))) },
      labels: { table: t().showTable, chart: t().showChart } });
    C.heat(fh.body, { rows: eraRows.map((p) => ({ label: featLabel(p.key), values: P.eras.map((e) => p.by_era[e]) })), cols: P.eras, fmt: (v) => pct(v, 1), valueLabel: tx("missing", "chybí"), aria: "missing data by era" });
    if (eraRows.some((p) => p.by_group)) {
      const G = [["women", tx("Women", "Ženy")], ["men", tx("Men", "Muži")], ["<80", tx("Under 80", "Pod 80")], ["80+", tx("80 or older", "80 a více")]];
      const fg = C.figure(card, {
        title: tx("Missing data by sex and age", "Chybějící data podle pohlaví a věku"),
        sub: tx("Share of patients without the value (%). The same cleaning rules applied to everyone; this shows whether the data were equally complete.", "Podíl pacientů bez hodnoty (%). Pro všechny platila stejná pravidla čištění; zde je vidět, zda byla data stejně úplná."),
        table: { columns: [tx("Input", "Vstup")].concat(G.map((g) => g[1])), rows: eraRows.map((p) => [featLabel(p.key)].concat(G.map(([k]) => (p.by_group && p.by_group[k] !== null && p.by_group[k] !== undefined ? pct(p.by_group[k], 1) : "–")))) },
        labels: { table: t().showTable, chart: t().showChart } });
      C.heat(fg.body, { rows: eraRows.map((p) => ({ label: featLabel(p.key), values: G.map(([k]) => (p.by_group ? p.by_group[k] : null)) })), cols: G.map((g) => g[1]), fmt: (v) => pct(v, 1), valueLabel: tx("missing", "chybí"), aria: "missing data by sex and age" });
    }
    return card;
  }

  // ------------------------------------------------------------------ 5. outcome
  function outcomeCard() {
    const O = P.outcome;
    const sp = P.flow.splits.primary;
    const card = h("div", { class: "card card-pad block", id: "pp-outcome" });
    card.appendChild(head(tx("The outcome: is the 90-day mRS recorded?", "Výsledek: je mRS za 90 dní zaznamenáno?"),
      tx("Adults in the registry, before the outcome exclusion", "Dospělí v registru, před vyřazením pro chybějící výsledek")));
    const row = (lbl, g) => h("tr", null, h("th", { scope: "row", text: lbl }), h("td", { class: "num", text: num(g.n) }),
      h("td", { class: "num", text: num(g.n - (g.missing || 0)) }), h("td", { class: "num", text: cnt(g.missing, g.missing === null) }), h("td", { class: "num", text: pct(g.missing_pct) }));
    card.appendChild(table([tx("Patients", "Pacienti"), "n", tx("mRS recorded", "mRS zaznamenáno"), tx("Missing", "Chybí"), "%"], [
      row(tx("All adults", "Všichni dospělí"), O.overall),
      row(tx(`Development years ${sp.dev.years[0]}–${sp.dev.years[1]}`, `Vývojové roky ${sp.dev.years[0]}–${sp.dev.years[1]}`), O.dev),
      row(tx(`Validation years ${sp.test.years[0]}–${sp.test.years[1]}`, `Validační roky ${sp.test.years[0]}–${sp.test.years[1]}`), O.test)]));
    const two = h("div", { class: "grid-2" });
    const yc = h("div");
    const fy = C.figure(yc, { title: tx("Missing 90-day mRS by year", "Chybějící mRS za 90 dní podle roku"), sub: tx("% of adults treated that year", "% dospělých léčených v daném roce"),
      table: { columns: [tx("Year", "Rok"), "n", tx("Missing", "Chybí")], rows: O.years.map((r) => [r.label, r.n, pct(r.missing_pct)]) },
      labels: { table: t().showTable, chart: t().showChart } });
    C.lineChart(fy.body, { height: 190, x: { type: "cat", cats: O.years.map((r) => r.label) }, y: { domain: [0, 30], ticks: [0, 10, 20, 30] },
      series: [{ label: tx("Missing", "Chybí"), color: "var(--c2)", area: true, dots: true, points: O.years.map((r, i) => ({ x: i, y: r.missing_pct, n: r.n })) }],
      tipFmt: (v, p) => `${pct(v)} · n ${p.n}`, aria: "missing outcome by year" });
    two.appendChild(yc);
    const wc = h("div");
    const wm = O.who_missing_n || {};
    wc.appendChild(head(tx("Who lacks the 90-day mRS?", "Komu chybí mRS za 90 dní?"),
      tx(`Treated 2019 onwards: ${num(wm.missing)} without vs ${num(wm.observed)} with the outcome`, `Léčení od roku 2019: ${num(wm.missing)} bez výsledku vs ${num(wm.observed)} s výsledkem`)));
    wc.appendChild(table([tx("Characteristic", "Charakteristika"), tx("Missing", "Chybí"), tx("Recorded", "Zaznamenáno"), "SMD"],
      O.who_missing.map((r) => h("tr", null, h("th", { scope: "row", text: L(r.label) }),
        h("td", { class: "num", text: r.binary ? pct(r.missing) : num(r.missing) }), h("td", { class: "num", text: r.binary ? pct(r.observed) : num(r.observed) }),
        h("td", { class: Math.abs(r.smd || 0) > 0.1 ? "num smd hi" : "num", text: A.signed(r.smd) }))), "compact"));
    two.appendChild(wc);
    card.appendChild(h("div", { class: "block", style: "margin-top:22px" }, two));
    card.appendChild(h("p", { class: "note", text: tx(
      "Medians for age and NIHSS, % for the rest; SMD beyond ±0.1 marks a relevant difference. Patients without a 90-day mRS were similar in age and severity but were transferred more often and had fewer follow-up records, so they were probably followed elsewhere rather than sicker. They are excluded (a complete-case outcome), as in the 2026 analysis; outcome imputation would add assumptions without new information.",
      "Mediány u věku a NIHSS, jinak %; SMD za ±0,1 značí podstatný rozdíl. Pacienti bez mRS za 90 dní byli podobného věku a tíže, ale častěji byli přeloženi a měli méně záznamů ze sledování, takže byli spíše sledováni jinde, než že by byli nemocnější. Jsou vyřazeni (analýza úplných výsledků), jako v analýze z roku 2026; imputace výsledku by přidala předpoklady bez nové informace.") }));
    return card;
  }

  // ------------------------------------------------------------------ 6. distributions
  function distCard() {
    const D = P.distributions;
    const card = h("div", { class: "card card-pad block", id: "pp-dist" });
    card.appendChild(head(tx("Distributions in the analysed patients", "Rozložení u analyzovaných pacientů"),
      tx(`n = ${num(P.flow.steps[P.flow.steps.length - 1].n)}; bins under ${MIN} patients are hidden`, `n = ${num(P.flow.steps[P.flow.steps.length - 1].n)}; skupiny pod ${MIN} pacientů jsou skryty`)));
    const grid = h("div", { class: "grid-3" });
    const one = (title, key, labelFn, color) => {
      const bins = D[key];
      if (!bins) return;
      const box = h("div");
      const cats = bins.map((b) => (labelFn ? labelFn(b.bin) : b.bin));
      const f = C.figure(box, { title, table: { columns: [tx("Group", "Skupina"), "n"], rows: bins.map((b, i) => [cats[i], cnt(b.n, b.small)]) }, labels: { table: t().showTable, chart: t().showChart } });
      C.columns(f.body, { cats, values: bins.map((b) => b.n), small: bins.map((b) => b.small), color: color || "var(--accent)", fmt: (v) => num(v), smallLabel: `<${MIN}`, valueLabel: tx("patients", "pacienti"), height: 170, aria: title });
      grid.appendChild(box);
    };
    one(tx("90-day mRS", "mRS za 90 dní"), "mrs90", null, "var(--c1)");
    one(tx("NIHSS on admission", "NIHSS při přijetí"), "nihss_adm");
    one(tx("Age, years", "Věk, roky"), "age");
    one(tx("Onset to groin puncture", "Od vzniku do punkce"), "onset_to_puncture");
    one(tx("Final mTICI", "Výsledné mTICI"), "tici", (b) => (b === "missing" ? tx("n/a", "chybí") : b === "2" ? "2*" : b));
    one(tx("NIHSS at 24 h (assessed)", "NIHSS za 24 h (hodnoceno)"), "nihss_24h");
    one(tx("NIHSS on day 7 (assessed)", "NIHSS 7. den (hodnoceno)"), "nihss_7d");
    const stat = (b) => ({ assessed: tx("scored", "skóre"), died: tx("died (43)", "úmrtí (43)"), not_assessable: tx("n/a (99)", "nehodn. (99)"), missing: tx("empty", "prázdné") }[b] || b);
    one(tx("24-h NIHSS: how recorded", "NIHSS za 24 h: jak zapsáno"), "nihss_24h_status", stat, "var(--c3)");
    one(tx("Day-7 NIHSS: how recorded", "NIHSS 7. den: jak zapsáno"), "nihss_7d_status", stat, "var(--c3)");
    card.appendChild(grid);
    // technique as horizontal bars (long labels)
    const tb = h("div", { class: "block" });
    const ft = C.figure(tb, { title: tx("Procedure: first-line technique", "Výkon: technika první volby"),
      table: { columns: [tx("Technique", "Technika"), "n"], rows: D.technique.map((b) => [lab2(TECH[b.bin]) || b.bin, cnt(b.n, b.small)]) }, labels: { table: t().showTable, chart: t().showChart } });
    C.hbars(ft.body, { rows: D.technique.filter((b) => b.n !== null).map((b) => ({ label: lab2(TECH[b.bin]) || b.bin, value: b.n, display: num(b.n) })), color: "var(--c2)", valueLabel: tx("patients", "pacienti"), aria: "technique" });
    card.appendChild(tb);
    card.appendChild(h("p", { class: "note", text: tx(
      "2*: recorded as TICI 2 without a subgrade. Times are minutes from onset or last known well. Follow-up NIHSS: “died” and “n/a” are the registry codes 43 and 99, kept as a status and never used as scores.",
      "2*: zapsáno jako TICI 2 bez podstupně. Časy jsou minuty od vzniku nebo od doby, kdy byl pacient naposledy v pořádku. NIHSS ze sledování: „úmrtí“ a „nehodn.“ jsou kódy registru 43 a 99, vedené jako stav a nikdy nepoužité jako skóre.") }));
    return card;
  }

  function cholesterolCard() {
    const rows = P.cholesterol || [];
    if (rows.length < 2) return null;
    const card = h("div", { class: "card card-pad block", id: "pp-chol" });
    card.appendChild(head(tx("Why cholesterol is not a predictor", "Proč cholesterol není prediktorem"),
      tx("Patients treated 2016 onwards, when it was routinely recorded", "Pacienti léčení od roku 2016, kdy se běžně zaznamenával")));
    const g = { measured: tx("Cholesterol recorded", "Cholesterol zaznamenán"), missing: tx("Cholesterol missing", "Cholesterol chybí") };
    card.appendChild(table([tx("Group", "Skupina"), "n", tx("Age (median)", "Věk (medián)"), tx("NIHSS (median)", "NIHSS (medián)"), "mRS 0–2", tx("Death", "Úmrtí")],
      rows.map((r) => h("tr", null, h("th", { scope: "row", text: g[r.group] || r.group }), h("td", { class: "num", text: num(r.n) }), h("td", { class: "num", text: num(r.age) }),
        h("td", { class: "num", text: num(r.nihss) }), h("td", { class: "num", text: pct(r.good) }), h("td", { class: "num", text: pct(r.death) })))));
    const [a, b] = [rows.find((r) => r.group === "measured"), rows.find((r) => r.group === "missing")];
    if (a && b) card.appendChild(h("p", { class: "note", text: tx(
      `At a similar age and NIHSS, patients without a cholesterol value died ${num(b.death / a.death, 1)} times as often (${pct(b.death)} vs ${pct(a.death)}). The missing value marks patients too unwell for, or not reached by, the routine blood draw; a model would learn that, not the effect of cholesterol.`,
      `Při podobném věku a NIHSS umírali pacienti bez hodnoty cholesterolu ${num(b.death / a.death, 1)}× častěji (${pct(b.death)} vs ${pct(a.death)}). Chybějící hodnota označuje pacienty, u nichž rutinní odběr neproběhl; model by se naučil právě to, ne vliv cholesterolu.`) }));
    return card;
  }

  // ------------------------------------------------------------------ 7. Table 1
  function table1Card() {
    const t1 = E.table1;
    if (!t1) return null;
    const card = h("div", { class: "card card-pad block", id: "pp-table1" });
    const sp = P.flow.splits.primary;
    card.appendChild(head(tx("Table 1. Patient characteristics", "Tabulka 1. Charakteristika pacientů"),
      tx(`Median (IQR) or % of patients with the value recorded; counts under ${MIN} not shown`, `Medián (IQR) nebo % pacientů se zaznamenanou hodnotou; počty pod ${MIN} nejsou uvedeny`)));
    const cols = ["all", "dev", "test", "smd", "good", "poor", "smd_outcome"];
    const headCols = [tx("Characteristic", "Charakteristika"), `${tx("All", "Všichni")} (n ${num(t1.n.all)})`,
      `${tx("Development", "Vývoj")} ${sp.dev.years[0]}–${sp.dev.years[1]} (n ${num(t1.n.dev)})`, `${tx("Validation", "Validace")} ${sp.test.years[0]}–${sp.test.years[1]} (n ${num(t1.n.test)})`, "SMD",
      `mRS 0–2 (n ${num(t1.n.good)})`, `mRS 3–6 (n ${num(t1.n.poor)})`, "SMD"];
    const cell = (r, c) => {
      if (c === "smd" || c === "smd_outcome") return r[c] === null || r[c] === undefined ? "–" : A.signed(r[c], 2);
      if ((r.var === "good" || r.var === "death") && (c === "good" || c === "poor")) return "–";
      const v = r[c];
      if (!v) return "–";
      if (r.kind === "median") return v.median === null ? "–" : `${num(v.median)} (${num(v.iqr[0])}–${num(v.iqr[1])})`;
      if (v.small) return `<${MIN}`;
      return v.pct === null ? "–" : pct(v.pct);
    };
    const body = [];
    for (const r of t1.rows) {
      if (r.section) { body.push(h("tr", { class: "sep" }, h("th", { colspan: headCols.length, text: L(r.section) }))); continue; }
      body.push(h("tr", null, h("th", { scope: "row", text: L(r.label) }),
        cols.map((c) => h("td", { class: (c === "smd" || c === "smd_outcome") ? (r[c] !== null && r[c] !== undefined && Math.abs(r[c]) > 0.1 ? "num smd hi" : "num smd") : "num", text: cell(r, c) }))));
    }
    card.appendChild(table(headCols, body, "t1"));
    card.appendChild(h("p", { class: "note", text: tx(
      "SMD: standardised mean difference, validation minus development (left) and mRS 0–2 minus mRS 3–6 (right); beyond ±0.1 (bold) marks a relevant difference. The left column shows how case mix and care changed between the development and validation years; the right one which characteristics go with the outcome before any adjustment. Missing values are left out of each percentage; their share is in the model-input table above.",
      "SMD: standardizovaný rozdíl průměrů, validace minus vývoj (vlevo) a mRS 0–2 minus mRS 3–6 (vpravo); hodnoty za ±0,1 (tučně) značí podstatný rozdíl. Levý sloupec ukazuje, jak se mezi vývojovými a validačními roky změnila skladba pacientů a péče; pravý, které charakteristiky souvisejí s výsledkem před jakoukoli adjustací. Chybějící hodnoty nejsou v procentech zahrnuty; jejich podíl je v tabulce vstupů výše.") }));
    return card;
  }

  // ------------------------------------------------------------------ page
  const SECTIONS = [
    ["pp-flow", ["Patient flow", "Tok pacientů"]], ["pp-stages", ["Stage cohorts", "Kohorty fází"]], ["pp-checks", ["Cleaning", "Čištění"]],
    ["pp-codes", ["Registry codes", "Kódy registru"]], ["pp-columns", ["Registry columns", "Sloupce registru"]], ["pp-inputs", ["Model inputs", "Vstupy modelů"]],
    ["pp-outcome", ["Outcome availability", "Dostupnost výsledku"]], ["pp-dist", ["Distributions", "Rozložení"]], ["pp-chol", ["Cholesterol", "Cholesterol"]],
    ["pp-table1", ["Table 1", "Tabulka 1"]],
  ];
  let host = null;
  function rerender(root, anchor) {
    const y = window.scrollY;
    render(root);
    if (anchor) window.scrollTo(0, y);
  }
  function render(root) {
    host = root;
    // keep the Registry header and sub-tabs (the first child), rebuild the rest
    while (root.children.length > 1) root.removeChild(root.lastChild);
    const k = P.flow;
    const last = k.steps[k.steps.length - 1];
    const V = P.variables;
    root.appendChild(h("div", { class: "kpis" },
      kpi(tx("Registry records", "Záznamy registru"), num(P.records), `${E.period[0]}–${E.period[1]}`),
      kpi(tx("Analysed", "Analyzováno"), num(last.n), tx("adults with a 90-day mRS", "dospělí s mRS za 90 dní")),
      kpi(tx("Registry columns", "Sloupce registru"), num(V.length), tx(`${V.filter((v) => v.role === "predictor" || v.role === "source").length} feed the models`, `${V.filter((v) => v.role === "predictor" || v.role === "source").length} vstupuje do modelů`)),
      kpi(tx("Model inputs", "Vstupy modelů"), num(P.predictors.length), tx(`${P.predictors.filter((p) => p.stage === "admission").length} before the procedure`, `${P.predictors.filter((p) => p.stage === "admission").length} před výkonem`)),
      kpi(tx("Development / validation", "Vývoj / validace"), `${num(k.splits.primary.dev.n)} / ${num(k.splits.primary.test.n)}`, `${k.splits.primary.dev.years[0]}–${k.splits.primary.dev.years[1]} / ${k.splits.primary.test.years[0]}–${k.splits.primary.test.years[1]}`)));
    const toc = h("nav", { class: "toc block", "aria-label": tx("On this page", "Na této stránce") }, h("span", { class: "muted", text: tx("On this page:", "Na této stránce:") }),
      SECTIONS.map(([id, lbl]) => h("a", { href: `#${id}`, onclick: (e) => { e.preventDefault(); const el = document.getElementById(id); if (el) el.scrollIntoView({ behavior: "smooth", block: "start" }); }, text: lab2(lbl) })));
    root.appendChild(toc);
    root.appendChild(flowCard());
    root.appendChild(stageCard());
    root.appendChild(checksCard());
    root.appendChild(codesCard());
    root.appendChild(variablesCard());
    root.appendChild(inputsCard(root));
    root.appendChild(outcomeCard());
    root.appendChild(distCard());
    const ch = cholesterolCard();
    if (ch) root.appendChild(ch);
    const t1 = table1Card();
    if (t1) root.appendChild(t1);
    root.appendChild(h("p", { class: "note block", text: tx(
      "Reporting follows the earlier MTOP pipeline's data report (cohort flow table and diagram, cleaning and exclusion logs, data-quality checks, outcome availability, missingness table, predictor dictionary, baseline tables by split and by outcome with SMD, distributions) and TRIPOD+AI items 5–9, 11 and 20; see the Standards tab.",
      "Výkaznictví navazuje na datovou zprávu dřívější pipeline MTOP (tabulka a diagram toku pacientů, záznamy čištění a vyřazení, kontroly kvality dat, dostupnost výsledku, tabulka chybějících dat, slovník prediktorů, výchozí tabulky podle rozdělení a podle výsledku se SMD, rozložení) a na položky TRIPOD+AI 5–9, 11 a 20; viz záložka Standardy.") }));
  }
  A.preprocessingView = render;
})();
