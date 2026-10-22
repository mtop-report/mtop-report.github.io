/* MTOP "Roadmap" tab: what works now, what is new, what is proposed. */
(function () {
  "use strict";
  const A = window.MTOPApp;
  if (!A) return;
  const { h, tx, num } = A;

  const COLS = [
    { key: "live", title: ["Live", "Funguje"], sub: ["Validated and in the app", "Ověřeno a v aplikaci"] },
    { key: "beta", title: ["New in this version", "Nové v této verzi"], sub: ["Computed from the registry, under review", "Spočteno z registru, v ověřování"] },
    { key: "progress", title: ["In progress", "Probíhá"], sub: ["Under way at FNO", "Probíhá ve FNO"] },
    { key: "proposed", title: ["Proposed next steps", "Navržené další kroky"], sub: ["Need partners, data or funding", "Vyžadují partnery, data nebo financování"] },
  ];

  function items() {
    const SP = A.V.splits.primary.stages;
    const ch = (s) => A.chosenOf(s, "good_outcome");
    const auc = (s) => num(SP[s].targets.good_outcome.models[ch(s)].auroc, 2);
    const yrs = A.yearsOf("primary");
    const AD = A.DATA.adaptive;
    const adm = AD ? (AD.learners ? (AD.results.admission.good_outcome[ch("admission")] || AD.results.admission.good_outcome[AD.learners[0]]) : AD.results.admission.good_outcome) : null;
    const pr = adm ? adm.pooled : null;
    const nL = (AD && AD.learners) ? AD.learners.length : 1;
    return [
      { s: "live", tab: "calc", t: ["Staged outcome calculator", "Postupná kalkulačka výsledku"],
        d: [`Estimate updated before the procedure, at its end, at 24 h and on day 7 (AUROC ${auc("admission")} → ${auc("d7")} in ${yrs}).`, `Odhad aktualizovaný před výkonem, na jeho konci, za 24 h a 7. den (AUROC ${auc("admission")} → ${auc("d7")} v letech ${yrs}).`] },
      { s: "live", tab: "calc", t: ["Exact explanations", "Přesná vysvětlení"], d: ["For every published learner, computed in the browser for each estimate: TreeSHAP for the tree models, the EBM's own terms, exact linear contributions for the regressions and their weighted sum for stacking.", "U každého zveřejněného modelu, počítané v prohlížeči pro každý odhad: TreeSHAP u stromových modelů, vlastní členy EBM, přesné lineární příspěvky u regresí a jejich vážený součet u stackingu."] },
      { s: "live", tab: "validation", t: ["Temporal validation of every learner", "Časová validace všech modelů"], d: ["Eight candidates compete for the calculator under a rule fixed before testing (amendment 1, Holm-adjusted): the regression with splines (reference), linear regression, LightGBM, XGBoost, EBM, regression with interactions, regression on multiple imputations and stacking. MLP, CatBoost, the default EBM, averaged probabilities and FT-Transformers are validated for comparison, and SHAP shows what each model relies on. Benchmarked against the FNO bedside score and published scores.", "O kalkulačku soutěží osm kandidátů podle pravidla stanoveného před testováním (dodatek 1, s Holmovou korekcí): regrese se splajny (reference), lineární regrese, LightGBM, XGBoost, EBM, regrese s interakcemi, regrese na vícenásobných imputacích a stacking. MLP, CatBoost, výchozí EBM, průměr pravděpodobností a FT-Transformery jsou validovány pro srovnání a SHAP ukazuje, na čem každý model stojí. Srovnáno s bedside skóre FNO a publikovanými skóre."] },
      { s: "live", tab: "registry", t: ["Registry explorer and data report", "Průzkumník registru a datová zpráva"], d: ["Two decades of FNO thrombectomy outcomes, with the full preprocessing report: patient flow, data checks, variable coding, missing data and Table 1.", "Dvě desetiletí výsledků trombektomií ve FNO s úplnou zprávou o předzpracování: tok pacientů, kontroly dat, kódování proměnných, chybějící data a tabulka 1."] },
      { s: "live", tab: "reporting", t: ["Reporting standards", "Standardy reportování"], d: ["TRIPOD+AI item by item, a PROBAST+AI self-assessment and the sample-size criteria.", "TRIPOD+AI položku po položce, vlastní hodnocení PROBAST+AI a kritéria velikosti souboru."] },
      { s: "live", tab: "about", t: ["Open, private by design", "Otevřené a soukromé už z návrhu"], d: ["Open code and model, no patient data on the page, runs offline.", "Otevřený kód i model, žádná data pacientů na stránce, funguje offline."] },
      { s: "beta", tab: "calc", t: ["Full 90-day mRS distribution", "Celé rozložení mRS za 90 dní"], d: ["Single-grade prediction with a 100-patient view for explaining prognosis.", "Predikce jednotlivých stupňů se zobrazením 100 pacientů pro vysvětlení prognózy."] },
      { s: "beta", tab: "calc", t: ["Uncertainty ranges", "Rozpětí nejistoty"], d: ["How much an estimate moves when the model is refitted on resampled data.", "O kolik se odhad pohne při přetrénování modelu na převzorkovaných datech."] },
      { s: "beta", tab: "calc", t: ["Every model for the same patient", "Všechny modely pro stejného pacienta"], d: ["Each published learner, the FNO bedside score and the published scores side by side; any learner can drive the calculator.", "Každý zveřejněný model, bedside skóre FNO a publikovaná skóre vedle sebe; kalkulačku může řídit kterýkoli model."] },
      { s: "beta", tab: "adaptive", t: ["Adaptive updating lab", "Laboratoř adaptivní aktualizace"],
        d: pr ? [`Six update strategies for ${nL} models, tested 2018–2025; for the calculator's model, refit + recalibration moved calibration-in-the-large from ${A.signed(pr.frozen.citl)} to ${A.signed(pr.refit_recal.citl)}.`, `Šest strategií aktualizace pro ${nL} modely, testováno 2018–2025; u modelu kalkulačky přetrénování + rekalibrace posunuly kalibraci z ${A.signed(pr.frozen.citl)} na ${A.signed(pr.refit_recal.citl)}.`]
          : ["Update strategies compared year by year.", "Strategie aktualizace porovnané rok po roce."] },
      { s: "beta", tab: "validate", t: ["Validation at partner centres", "Validace v partnerských centrech"], d: ["Any centre can test every published MTOP model on its own data in the browser and share only summary numbers.", "Kterékoli centrum může na vlastních datech v prohlížeči ověřit každý zveřejněný model MTOP a sdílet jen souhrnná čísla."] },
      { s: "progress", t: ["CT perfusion and serum GFAP", "CT perfuze a sérový GFAP"], d: ["Perfusion volumes (core, Tmax > 6 s, mismatch) are recorded for new patients at FNO and GFAP is being collected; both are optional columns of the partner template and enter a model only after their added value is tested (see below).", "Objemy z perfuze (jádro, Tmax > 6 s, mismatch) se ve FNO evidují u nových pacientů a GFAP se sbírá; obojí je nepovinnými sloupci partnerské šablony a do modelu vstoupí až po ověření přínosu (viz níže)."] },
      { s: "progress", tab: "reporting", t: ["Manuscript", "Rukopis"], d: ["Development and temporal validation, reported to TRIPOD+AI.", "Vývoj a časová validace, podle TRIPOD+AI."] },
      { s: "proposed", tab: "reporting", t: ["Protocol and registration", "Protokol a registrace"], d: ["Publish the analysis protocol and register it (for example on OSF) before the next analysis round.", "Zveřejnit protokol analýzy a registrovat jej (např. na OSF) před dalším kolem analýzy."] },
      { s: "proposed", t: ["Completing recent follow-up", "Doplnění nedávného sledování"], d: ["Filling missing 24-h NIHSS and 90-day mRS since 2021 strengthens the later stages most.", "Doplnění chybějícího NIHSS za 24 h a mRS za 90 dní od roku 2021 nejvíce posílí pozdější fáze."] },
      { s: "proposed", t: ["External validation", "Externí validace"], d: ["Other Czech and Central European stroke centres, using the in-browser tool.", "Další česká a středoevropská iktová centra pomocí nástroje v prohlížeči."] },
      { s: "proposed", t: ["Prospective silent validation", "Prospektivní tichá validace"], d: [`Model v${A.B.version} frozen (fingerprint ${(A.B.fingerprint || "").slice(0, 12)}…) and scored on new patients without showing clinicians.`, `Model v${A.B.version} zmrazen (otisk ${(A.B.fingerprint || "").slice(0, 12)}…) a hodnocen na nových pacientech bez zobrazení lékařům.`] },
      { s: "proposed", t: ["Scheduled recalibration", "Plánovaná rekalibrace"], d: ["Under the written update policy, once 2026 outcomes are complete.", "Podle psaných pravidel aktualizace, až budou výsledky roku 2026 kompletní."] },
      { s: "proposed", t: ["Collaterals and procedure detail", "Kolaterály a detaily výkonu"], d: ["The CTA collateral score (Tan, 0–3) is in the partner template; once recorded, MR PREDICTS can also be tested with its published formula. Number of passes and eTICI 2c would refine the end-of-procedure stage.", "Skóre kolaterál na CTA (Tan, 0–3) je v partnerské šabloně; jakmile se bude evidovat, lze MR PREDICTS otestovat i publikovaným vzorcem. Počet průchodů a eTICI 2c by zpřesnily fázi na konci výkonu."] },
      { s: A.DATA.federated ? "beta" : "proposed", tab: "federated", t: ["Federated learning", "Federované učení"],
        d: A.DATA.federated ? ["Working network: secure aggregation, exact federated boosting and regression, validation; demonstrated on four virtual centres. Needs partner centres next.", "Funkční síť: bezpečná agregace, přesný federovaný boosting i regrese, validace; předvedeno na čtyřech virtuálních centrech. Dále potřebuje partnerská centra."]
          : ["A shared model trained across centres without moving data, coordinated on HPC.", "Společný model trénovaný napříč centry bez přesunu dat, koordinovaný na HPC."] },
      { s: "proposed", t: ["Clinicians vs model", "Lékaři vs model"], d: ["Compare prognostic estimates of neurologists and interventionalists with MTOP.", "Porovnat prognostické odhady neurologů a intervenčních radiologů s MTOP."] },
      { s: "proposed", t: ["Hospital system link", "Napojení na nemocniční systém"], d: ["Prefill inputs from the hospital record (FHIR), keeping computation local.", "Předvyplnění vstupů z nemocniční dokumentace (FHIR) s lokálním výpočtem."] },
      { s: "proposed", tab: "reporting", t: ["Regulatory pathway", "Regulatorní cesta"], d: ["Only if clinical use is pursued: EU MDR and AI Act requirements for decision-support software.", "Jen pokud se bude usilovat o klinické použití: požadavky EU MDR a AI Act na software pro podporu rozhodování."] },
    ];
  }

  function render(root) {
    root.replaceChildren();
    root.appendChild(h("div", { class: "view-head" }, h("div", null,
      h("h2", { text: tx("Where MTOP is going", "Kam MTOP směřuje") }),
      h("p", { text: tx(
        "What works today, what this version adds, and what could come next. Proposed items are ideas for discussion, not commitments.",
        "Co funguje dnes, co přináší tato verze a co může následovat. Navržené položky jsou náměty k diskusi, nikoli závazky.") }))));
    const all = items();
    const board = h("div", { class: "board" });
    for (const col of COLS) {
      const list = all.filter((it) => it.s === col.key);
      const colEl = h("section", { class: `board-col col-${col.key}`, "aria-label": tx(col.title[0], col.title[1]) },
        h("div", { class: "board-head" }, A.statusChip(col.key === "proposed" ? "proposed" : col.key), h("span", { class: "board-count", text: String(list.length) })),
        h("p", { class: "board-sub", text: tx(col.sub[0], col.sub[1]) }));
      for (const it of list) {
        const card = h("article", { class: "board-card" }, h("h3", { text: tx(it.t[0], it.t[1]) }), h("p", { text: tx(it.d[0], it.d[1]) }));
        if (it.tab) {
          const b = h("button", { class: "btn btn-ghost", type: "button" }, tx(`Open ${A.viewLabel(it.tab)}`, `Otevřít: ${A.viewLabel(it.tab)}`));
          b.addEventListener("click", () => A.goTo(it.tab));
          card.appendChild(b);
        }
        colEl.appendChild(card);
      }
      board.appendChild(colEl);
    }
    root.appendChild(board);

    root.appendChild(futureCard());

    const steps = [
      [tx("Now", "Nyní"), tx(`v${A.B.version} public; adaptive, multicentre and federated tools in beta`, `v${A.B.version} veřejně; adaptivní, multicentrické a federované nástroje v betě`)],
      [tx("Next 6–12 months", "Dalších 6–12 měsíců"), tx("Complete recent follow-up, recalibrate on 2026 outcomes, test GFAP, first partner validations, first centres in the federated network", "Doplnit nedávné sledování, rekalibrovat na výsledcích 2026, otestovat GFAP, první partnerské validace, první centra ve federované síti")],
      [tx("Beyond", "Dále"), tx("Multicentre model, prospective silent validation, imaging predictors, impact study", "Multicentrický model, prospektivní tichá validace, zobrazovací prediktory, studie dopadu")],
    ];
    const tl = h("div", { class: "card card-pad block" }, h("div", { class: "chart-title", text: tx("Proposed sequence", "Navržené pořadí"), style: "margin-bottom:10px" }),
      h("ol", { class: "steps" }, steps.map(([a, b]) => h("li", null, h("b", { text: a }), h("span", { text: b })))));
    root.appendChild(tl);
  }

  // how close each group of new predictors is to the 100 + 100 rule of amendment 1: FNO's registry
  // export plus the latest reviewed result of each partner centre (counts of 1-9 are not published)
  function futureCard() {
    const F = A.B.future || {};
    const nMin = F.min_per_outcome || 100;
    const latest = {};
    for (const e of ((A.DATA.validation_pool || {}).results || [])) {
      if (!e.result || !e.result.future) continue;
      if (!latest[e.site] || String(e.accepted) > String(latest[e.site].accepted)) latest[e.site] = e;
    }
    const sites = Object.values(latest);
    const G = [["perfusion", ["CT perfusion: core and Tmax > 6 s volumes", "CT perfuze: objem jádra a Tmax > 6 s"]],
      ["collaterals", ["CTA collateral score", "Skóre kolaterál na CTA"]],
      ["gfap", ["Serum GFAP with sampling time", "Sérový GFAP s časem odběru"]]];
    const cell = (c, k) => (c === null || c === undefined ? null : c[k]);
    const show = (v) => (v === null ? tx("1–9", "1–9") : num(v));
    const rows = G.map(([g, lbl]) => {
      const fno = F.fno ? F.fno[g] : null;
      const part = sites.map((e) => e.result.future[g]).filter(Boolean);
      const sum = (k) => part.reduce((a, c) => a + (c[k] || 0), 0);
      const plus = (k) => part.some((c) => c[k] === null);
      const good = (fno ? (cell(fno, "good") || 0) : 0) + sum("good");
      const poor = (fno ? (cell(fno, "poor") || 0) : 0) + sum("poor");
      const ready = good >= nMin && poor >= nMin;
      return [tx(lbl[0], lbl[1]),
        fno ? `${show(cell(fno, "good"))} / ${show(cell(fno, "poor"))}` : tx("not in the export yet", "zatím není v exportu"),
        part.length ? `${num(sum("good"))}${plus("good") ? "+" : ""} / ${num(sum("poor"))}${plus("poor") ? "+" : ""}` : "–",
        ready ? tx("ready to test", "lze testovat") : tx(`needs ${num(Math.max(0, nMin - good))} more with mRS 0–2, ${num(Math.max(0, nMin - poor))} with mRS 3–6`,
          `chybí ${num(Math.max(0, nMin - good))} s mRS 0–2 a ${num(Math.max(0, nMin - poor))} s mRS 3–6`)];
    });
    return h("div", { class: "card card-pad block" },
      h("div", { class: "chart-title", text: tx("New predictors: when they can be tested", "Nové prediktory: kdy je lze otestovat") }),
      h("p", { class: "chart-sub", text: tx(
        `A predictor collected for a future version is tested once at least ${nMin} patients with mRS 0–2 and ${nMin} with mRS 3–6 at 90 days have it recorded (for death: ${nMin} deaths and ${nMin} survivors). The current model's prediction is kept as an offset and the new variable is added; it enters the next version only if the likelihood-ratio test, the change in AUROC and calibration on the most recent patients show added value (amendment 1).`,
        `Prediktor sbíraný pro budoucí verzi se otestuje, až jej bude mít zaznamenáno alespoň ${nMin} pacientů s mRS 0–2 a ${nMin} s mRS 3–6 za 90 dní (pro úmrtí ${nMin} zemřelých a ${nMin} přeživších). Predikce stávajícího modelu se ponechá jako offset a přidá se nová proměnná; do další verze vstoupí, jen pokud test věrohodnostním poměrem, změna AUROC a kalibrace na nejnovějších pacientech ukážou přínos (dodatek 1).`) }),
      h("div", { class: "table-scroll" }, h("table", { class: "data compact" },
        h("thead", null, h("tr", null, [tx("Predictor", "Prediktor"), tx("FNO, mRS 0–2 / 3–6", "FNO, mRS 0–2 / 3–6"),
          tx(`Partner centres (${sites.length})`, `Partnerská centra (${sites.length})`), tx("Status", "Stav")].map((x) => h("th", { scope: "col", text: x })))),
        h("tbody", null, rows.map((r) => h("tr", null, h("th", { scope: "row", text: r[0] }), h("td", { text: r[1] }), h("td", { text: r[2] }), h("td", { text: r[3] })))))),
      h("p", { class: "note", text: tx("FNO counts come from the registry export behind this version; partner counts from reviewed results files (the latest per centre). 1–9 marks a count too small to publish.",
        "Počty FNO pocházejí z exportu registru této verze; počty partnerů z posouzených souborů výsledků (poslední za každé centrum). 1–9 označuje počet příliš malý ke zveřejnění.") }));
  }

  A.registerView("roadmap", () => tx("Roadmap", "Plán rozvoje"), render);
})();
