/* MTOP data dictionary (About tab): every predictor with its options and how the registry's own
 * values were coded into them, plus the registry columns MTOP does not use, and why. */
(function () {
  "use strict";
  const A = window.MTOPApp;
  if (!A) return;
  const { h, tx, L } = A;

  // [model value (en, cs), registry values that map to it]
  const CODING = {
    age: [["Years (18–100)", "Roky (18–100)"], "Age; under 18 excluded"],
    sex_male: [[["Male", "Muž"], "Muž"], [["Female", "Žena"], "Žena"]],
    mrs_pre: [["0–5, one option per grade", "0–5, jedna volba na stupeň"], "mRS before"],
    nihss_adm: [["0–42", "0–42"], "Admission NIHSS"],
    aspects: [["0–10; recorded from 2016 (pc-ASPECTS in posterior circulation)", "0–10; evidováno od 2016 (v zadním povodí pc-ASPECTS)"], "ASPECTS"],
    occlusion: [
      [["Intracranial ICA / carotid T", "Intrakraniální ACI / karotické T"], "ACI-Intracranial, ACI+ACM"],
      [["MCA M1", "ACM M1"], "M1-ACM"],
      [["MCA M2–M3 / ACA", "ACM M2–M3 / ACA"], "M2-ACM, M3-ACM, ACA"],
      [["Cervical ICA/CCA (± tandem)", "Krční ACI/ACC (± tandem)"], "ACI-odstup, ACI-odstup +T typ, ACC, ACC+ACI+ACM"],
      [["Basilar artery", "Arteria basilaris"], "AB, AB+AV"],
      [["Vertebral / PCA", "AV / ACP"], "AV, ACP"],
      [["Missing", "Chybí"], "Rekanalizace (already open at angiography), empty"],
    ],
    side_left: [[["Left", "Levá"], "levá hemisféra"], [["Right", "Pravá"], "pravá hemisféra"],
      [["Right (0) for posterior circulation; the occlusion site carries it", "Pravá (0) u zadního povodí; nese ji místo okluze"], "zadní povodí"]],
    onset_to_puncture: [["Minutes from onset or last known well", "Minuty od vzniku nebo od doby, kdy byl naposledy v pořádku"], "Stroke onset to puncture (min)"],
    ivt: [[["Yes / No", "Ano / Ne"], "IVT: ANO / NE"]],
    transfer: [[["Drip-and-ship", "Drip-and-ship"], "name of a referring hospital"], [["Direct to FNO", "Přímo do FNO"], "FNO, žádné"]],
    glucose: [["mmol/L", "mmol/l"], "Glycemia"],
    sbp: [["mmHg; systolic ≤ diastolic → both missing", "mmHg; systolický ≤ diastolický → oba chybí"], "Systolic BP"],
    htn: [[["Yes / No", "Ano / Ne"], "Hypertension: ANO / NE"]],
    dm: [[["Yes / No", "Ano / Ne"], "DM: ANO / NE"]],
    arrhythmia: [[["Yes / No", "Ano / Ne"], "Arrhythmia: ANO / NE"]],
    stroke_prior: [[["Yes / No", "Ano / Ne"], "CMP before: ANO / NE"]],
    anticoag: [[["Yes", "Ano"], "VKA-Warfarin, NOAC-rivaroxaban, NOAC-apixaban, NOAC-dabigatran, LMWH"],
      [["No", "Ne"], "žádná, or antiplatelets only (AT-ASA, AC-acylpyrin, AP-apotic, AT-Clopidogrel, AT-dual ASA+Clop)"]],
    tici: [[["0, 1, 2a, 2b, 3", "0, 1, 2a, 2b, 3"], "TICI: 0, 1, 2a, 2b, 3"],
      [["2 without subgrade: coded between 2a and 2b, not offered as an input", "2 bez podstupně: kódováno mezi 2a a 2b, jako vstup se nenabízí"], "TICI: 2"]],
    procedure_min: [["Minutes from puncture to recanalisation or end", "Minuty od punkce do rekanalizace nebo konce"], "Procedure duration"],
    technique: [
      [["Direct aspiration (ADAPT)", "Přímá aspirace (ADAPT)"], "procedure ADAPT; other MT procedures with device PENUMBRA"],
      [["Stent retriever", "Stentriever"], "procedure MT-stent retriever; other MT procedures with Catch, Preset, Solitaire, TIGERTRIEVER, Embotrap, Bonnet, Merci"],
      [["Combined (e.g. Solumbra)", "Kombinovaná (např. Solumbra)"], "procedure Solumbra; other MT procedures with PENUMBRA + Catch / Preset / TIGERTRIEVER"],
      [["No mechanical thrombectomy", "Bez mechanické trombektomie"], "cPTA, PTA, Stent, Stent+PTA, cPTA+stent, EKOS+stent, LIAT, EKOS+LIAT, cPTA+LIAT"],
      [["Not analysed", "Neanalyzováno"], "PAG (diagnostic angiography only)"],
    ],
    stenting: [[["Yes", "Ano"], "any stent in Stent combo, or MT+stent, Stent, Stent+PTA, cPTA+stent, EKOS+stent"], [["No", "Ne"], "žádný"]],
    anesthesia: [[["General", "Celková"], "Celková anestezie"], [["Conscious sedation", "Analgosedace"], "Analgosedace"], [["Local only", "Lokální (bez ARO)"], "bez ARO"]],
    nihss_24h: [["0–42; 43 (died) and 99 (not assessable) are never used as scores", "0–42; 43 (zemřel) a 99 (nehodnotitelné) se nikdy nepoužívají jako skóre"], "NIHSS 24h"],
    sich: [[["Yes / No", "Ano / Ne"], "SICH: ANO / NE"]],
    nihss_change_24h: [["NIHSS at 24 h minus NIHSS on admission", "NIHSS za 24 h minus NIHSS při přijetí"], "computed"],
    nihss_7d: [["0–42; 43 and 99 as above", "0–42; 43 a 99 jako výše"], "NIHSS 7d"],
    edema: [[["Yes / No", "Ano / Ne"], "Cerebral edema: ANO / NE"]],
  };

  const UNUSED = [
    ["Date and time of onset", ["Year only splits development from validation and sets the recalibration; time of day is not a predictor.", "Rok slouží jen k rozdělení na vývoj a validaci a k rekalibraci; denní doba prediktorem není."]],
    ["Thrombolytics", ["IV thrombolysis yes/no is used; the agent (alteplase, tenecteplase) changed over the years. Integrilin is an antiplatelet drug, not a thrombolytic.", "Používá se IV trombolýza ano/ne; přípravek (alteplasa, tenekteplasa) se v čase měnil. Integrilin je antiagregancium, ne trombolytikum."]],
    ["IVT in different hospital", ["Covered by the admission route (transfer).", "Pokrývá ho způsob přijetí (překlad)."]],
    ["IAT", ["Intra-arterial lysis is rare and part of the procedure details.", "Intraarteriální lýza je vzácná a je součástí detailů výkonu."]],
    ["Stroke onset to IVT, to recanalization (min)", ["Onset to puncture and procedure duration carry the same timing.", "Stejné časování nesou čas od vzniku do punkce a délka výkonu."]],
    ["non-SICH", ["Only symptomatic haemorrhage is used.", "Používá se jen symptomatické krvácení."]],
    ["Hyperlipidemia, Smoking, Alcohol abusus, Statins before, TIA before, BMI, Heart condition, Diastolic BP, antiplatelets",
      ["No added value in this registry: adding twelve such items changed the 2021–2025 AUROC by +0.001 for MTOP's model and −0.022 for spline regression (on the earlier report's 2023–2025 test years by +0.006 and +0.018); several are often missing.",
        "V tomto registru bez přínosu: přidání dvanácti takových položek změnilo AUROC 2021–2025 o +0,001 u modelu MTOP a o −0,022 u regrese se splajny (na testovacích letech dřívější zprávy 2023–2025 o +0,006 a +0,018); několik z nich často chybí."]],
    ["Cholesterol", ["Measured later in the stay; whether it is missing tracks early death, which would leak the outcome.", "Měří se později během hospitalizace; jeho chybění souvisí s časným úmrtím, což by prozradilo výsledek."]],
    ["Etiology", ["Known only after the work-up, not at the prediction moments.", "Známa až po vyšetřeních, ne v okamžicích predikce."]],
    ["Decompression surgery", ["Happens after the prediction moments.", "Probíhá až po okamžicích predikce."]],
  ];

  function codingCell(key) {
    const c = CODING[key];
    if (!c) return h("td");
    if (Array.isArray(c[0]) && typeof c[0][0] === "string" && typeof c[1] === "string" && c.length === 2) {
      // [description (en, cs), registry column]
      return h("td", null, h("div", { text: tx(c[0][0], c[0][1]) }), h("div", { class: "dd-src", text: c[1] }));
    }
    return h("td", null, h("ul", { class: "dd-map" }, c.map(([lbl, src]) =>
      h("li", null, h("b", { text: tx(lbl[0], lbl[1]) }), h("span", { class: "dd-src", text: `← ${src}` })))));
  }

  const RAW = {
    age: "Age", sex_male: "Sex", mrs_pre: "mRS before", nihss_adm: "Admission NIHSS", aspects: "ASPECTS", occlusion: "Occlusion site",
    side_left: "Hemisphere", onset_to_puncture: "Stroke onset to puncture (min)", ivt: "IVT", transfer: "Transfer from different hospital",
    glucose: "Glycemia", sbp: "Systolic BP", htn: "Hypertension", dm: "DM", arrhythmia: "Arrhythmia", stroke_prior: "CMP before",
    anticoag: "Anti-thrombotics before", tici: "TICI", procedure_min: "Procedure duration", technique: "Intervention procedure + Extraction system",
    stenting: "Stent combo + Intervention procedure", anesthesia: "Anesthesia", nihss_24h: "NIHSS 24h", sich: "SICH",
    nihss_change_24h: "NIHSS 24h − Admission NIHSS", nihss_7d: "NIHSS 7d", edema: "Cerebral edema",
  };

  A.dictionaryBlock = function () {
    const det = h("details", { class: "card card-pad block dict" });
    det.appendChild(h("summary", null, h("span", { class: "chart-title", text: tx("Data dictionary: every predictor, its options and how the registry was coded",
      "Datový slovník: všechny prediktory, jejich volby a kódování registru") })));
    det.appendChild(h("p", { class: "note", text: tx(
      "Options with several values enter the model as one yes/no column each (one-hot coding); if the value is missing, all of them are missing and the trees learn where such patients belong. Ordered values (pre-stroke mRS, mTICI) stay single numbers so their order is used. Several registry values were grouped into one option where they mean the same for prognosis or are too rare to stand alone.",
      "Volby s více hodnotami vstupují do modelu jako samostatné sloupce ano/ne (one-hot kódování); chybí-li hodnota, chybí všechny a stromy se naučí, kam tito pacienti patří. Uspořádané hodnoty (mRS před CMP, mTICI) zůstávají jedním číslem, aby se využilo jejich pořadí. Několik hodnot registru bylo sloučeno do jedné volby tam, kde pro prognózu znamenají totéž nebo jsou příliš vzácné na samostatnou kategorii.") }));
    const tbl = h("table", { class: "data dd dd-main" });
    tbl.appendChild(h("thead", null, h("tr", null, [tx("Predictor", "Prediktor"), tx("Used from", "Použito od"), tx("Registry column", "Sloupec registru"),
      tx("In the model ← registry values", "V modelu ← hodnoty registru")].map((c) => h("th", { scope: "col", text: c })))));
    const tb = h("tbody");
    for (const f of A.B.features) {
      const cell = codingCell(f.key);
      tb.appendChild(h("tr", null, h("th", { scope: "row", text: L(f.label) }),
        h("td", { "data-l": tx("Used from", "Použito od"), text: A.stageLabel(f.stage) }),
        h("td", { class: "dd-src", "data-l": tx("Registry column", "Sloupec registru"), text: RAW[f.key] || "" }), cell));
    }
    tb.appendChild(h("tr", { class: "dd-outcome" }, h("th", { scope: "row", text: tx("Outcome", "Výsledek") }), h("td", { text: tx("90 days", "90 dní") }),
      h("td", { class: "dd-src", text: "90-day mRS" }), h("td", { text: tx("mRS 0–2 = functional independence; mRS 6 = death", "mRS 0–2 = funkční nezávislost; mRS 6 = úmrtí") })));
    tbl.appendChild(tb);
    det.appendChild(h("div", { class: "table-scroll" }, tbl));
    det.appendChild(h("h3", { class: "dd-h", text: tx("Registry columns MTOP does not use", "Sloupce registru, které MTOP nepoužívá") }));
    const t2 = h("table", { class: "data dd" });
    t2.appendChild(h("thead", null, h("tr", null, h("th", { scope: "col", text: tx("Registry column", "Sloupec registru") }), h("th", { scope: "col", text: tx("Why", "Proč") }))));
    t2.appendChild(h("tbody", null, UNUSED.map(([col, why]) => h("tr", null, h("th", { scope: "row", class: "dd-src", text: col }), h("td", { text: tx(why[0], why[1]) })))));
    det.appendChild(h("div", { class: "table-scroll" }, t2));
    return det;
  };
})();
