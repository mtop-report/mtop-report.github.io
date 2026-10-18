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
      { s: "live", tab: "calc", t: ["Exact explanations", "Přesná vysvětlení"], d: ["TreeSHAP for the tree models and exact linear contributions for the regressions, computed in the browser for every estimate.", "TreeSHAP u stromových modelů a přesné lineární příspěvky u regresí, počítané v prohlížeči pro každý odhad."] },
      { s: "live", tab: "validation", t: ["Temporal validation of five learners", "Časová validace pěti modelů"], d: ["Regression with splines (reference), linear regression, LightGBM, XGBoost and an MLP with the same tuning budget; a rule fixed before testing picks the calculator's model. Benchmarked against the FNO bedside score, published scores and NIHSS alone.", "Regrese se splajny (reference), lineární regrese, LightGBM, XGBoost a MLP se stejným rozpočtem ladění; model kalkulačky vybírá pravidlo stanovené před testováním. Srovnáno s bedside skóre FNO, publikovanými skóre a samotným NIHSS."] },
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
      { s: "progress", t: ["Serum GFAP", "Sérový GFAP"], d: ["Being collected at FNO; enters the model after its added value is tested.", "Sbírán ve FNO; do modelu vstoupí po ověření přínosu."] },
      { s: "progress", tab: "reporting", t: ["Manuscript", "Rukopis"], d: ["Development and temporal validation, reported to TRIPOD+AI.", "Vývoj a časová validace, podle TRIPOD+AI."] },
      { s: "proposed", tab: "reporting", t: ["Protocol and registration", "Protokol a registrace"], d: ["Publish the analysis protocol and register it (for example on OSF) before the next analysis round.", "Zveřejnit protokol analýzy a registrovat jej (např. na OSF) před dalším kolem analýzy."] },
      { s: "proposed", t: ["Completing recent follow-up", "Doplnění nedávného sledování"], d: ["Filling missing 24-h NIHSS and 90-day mRS since 2021 strengthens the later stages most.", "Doplnění chybějícího NIHSS za 24 h a mRS za 90 dní od roku 2021 nejvíce posílí pozdější fáze."] },
      { s: "proposed", t: ["External validation", "Externí validace"], d: ["Other Czech and Central European stroke centres, using the in-browser tool.", "Další česká a středoevropská iktová centra pomocí nástroje v prohlížeči."] },
      { s: "proposed", t: ["Prospective silent validation", "Prospektivní tichá validace"], d: [`Model v${A.B.version} frozen (fingerprint ${(A.B.fingerprint || "").slice(0, 12)}…) and scored on new patients without showing clinicians.`, `Model v${A.B.version} zmrazen (otisk ${(A.B.fingerprint || "").slice(0, 12)}…) a hodnocen na nových pacientech bez zobrazení lékařům.`] },
      { s: "proposed", t: ["Scheduled recalibration", "Plánovaná rekalibrace"], d: ["Under the written update policy, once 2026 outcomes are complete.", "Podle psaných pravidel aktualizace, až budou výsledky roku 2026 kompletní."] },
      { s: "proposed", t: ["Imaging and procedure detail", "Zobrazení a detaily výkonu"], d: ["CT perfusion core, collaterals, number of passes, eTICI 2c.", "Jádro infarktu z CT perfuze, kolaterály, počet průchodů, eTICI 2c."] },
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

    const steps = [
      [tx("Now", "Nyní"), tx(`v${A.B.version} public; adaptive, multicentre and federated tools in beta`, `v${A.B.version} veřejně; adaptivní, multicentrické a federované nástroje v betě`)],
      [tx("Next 6–12 months", "Dalších 6–12 měsíců"), tx("Complete recent follow-up, recalibrate on 2026 outcomes, test GFAP, first partner validations, first centres in the federated network", "Doplnit nedávné sledování, rekalibrovat na výsledcích 2026, otestovat GFAP, první partnerské validace, první centra ve federované síti")],
      [tx("Beyond", "Dále"), tx("Multicentre model, prospective silent validation, imaging predictors, impact study", "Multicentrický model, prospektivní tichá validace, zobrazovací prediktory, studie dopadu")],
    ];
    const tl = h("div", { class: "card card-pad block" }, h("div", { class: "chart-title", text: tx("Proposed sequence", "Navržené pořadí"), style: "margin-bottom:10px" }),
      h("ol", { class: "steps" }, steps.map(([a, b]) => h("li", null, h("b", { text: a }), h("span", { text: b })))));
    root.appendChild(tl);
  }

  A.registerView("roadmap", () => tx("Roadmap", "Plán rozvoje"), render);
})();
