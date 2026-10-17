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
    const V = A.V.stages;
    const auc = (s) => num(V[s].targets.good_outcome.temporal.LightGBM.auroc, 2);
    const AD = A.DATA.adaptive;
    const pr = AD ? AD.results.admission.good_outcome.pooled : null;
    return [
      { s: "live", tab: "calc", t: ["Staged outcome calculator", "Postupná kalkulačka výsledku"],
        d: [`Estimate updated at admission, end of MT, 24 h and day 7 (AUROC ${auc("admission")} → ${auc("d7")} in 2021–2025).`, `Odhad aktualizovaný při přijetí, na konci MT, za 24 h a 7. den (AUROC ${auc("admission")} → ${auc("d7")} v letech 2021–2025).`] },
      { s: "live", tab: "calc", t: ["Exact explanations", "Přesná vysvětlení"], d: ["TreeSHAP contributions computed in the browser for every estimate.", "Příspěvky TreeSHAP počítané v prohlížeči pro každý odhad."] },
      { s: "live", tab: "validation", t: ["Temporal validation and benchmarking", "Časová validace a srovnání"], d: ["Against logistic regression (linear and with splines), XGBoost, an MLP, the FNO bedside score, six published scores or predictor sets and NIHSS alone.", "Proti logistické regresi (lineární i se splajny), XGBoost, MLP, bedside skóre FNO, šesti publikovaným skóre či sadám prediktorů a samotnému NIHSS."] },
      { s: "live", tab: "registry", t: ["Registry explorer", "Průzkumník registru"], d: ["Two decades of FNO thrombectomy outcomes, aggregated and privacy-protected.", "Dvě desetiletí výsledků trombektomií ve FNO, souhrnně a s ochranou soukromí."] },
      { s: "live", tab: "about", t: ["Open, private by design", "Otevřené a soukromé už z návrhu"], d: ["Open code and model, no patient data on the page, runs offline.", "Otevřený kód i model, žádná data pacientů na stránce, funguje offline."] },
      { s: "beta", tab: "calc", t: ["Full 90-day mRS distribution", "Celé rozložení mRS za 90 dní"], d: ["Single-grade prediction with a 100-patient view for explaining prognosis.", "Predikce jednotlivých stupňů se zobrazením 100 pacientů pro vysvětlení prognózy."] },
      { s: "beta", tab: "calc", t: ["Uncertainty ranges", "Rozpětí nejistoty"], d: ["How much an estimate moves when the model is refitted on resampled data.", "O kolik se odhad pohne při přetrénování modelu na převzorkovaných datech."] },
      { s: "beta", tab: "calc", t: ["Other models side by side", "Další modely vedle sebe"], d: ["The FNO bedside score, THRIVE, THRIVE-c, SPAN-100, HIAT2, PRE and MR PREDICTS@24H for the same patient.", "Bedside skóre FNO, THRIVE, THRIVE-c, SPAN-100, HIAT2, PRE a MR PREDICTS@24H pro stejného pacienta."] },
      { s: "beta", tab: "adaptive", t: ["Adaptive updating lab", "Laboratoř adaptivní aktualizace"],
        d: pr ? [`Six update strategies tested 2018–2025; refit + recalibration cut calibration-in-the-large from ${A.signed(pr.frozen.citl)} to ${A.signed(pr.refit_recal.citl)}.`, `Šest strategií aktualizace testováno 2018–2025; přetrénování + rekalibrace snížily kalibrační chybu z ${A.signed(pr.frozen.citl)} na ${A.signed(pr.refit_recal.citl)}.`]
          : ["Update strategies compared year by year.", "Strategie aktualizace porovnané rok po roce."] },
      { s: "beta", tab: "validate", t: ["Validation at partner centres", "Validace v partnerských centrech"], d: ["Any centre can test MTOP on its own data in the browser and share only summary numbers.", "Kterékoli centrum může MTOP ověřit na vlastních datech v prohlížeči a sdílet jen souhrnná čísla."] },
      { s: "progress", t: ["Serum GFAP", "Sérový GFAP"], d: ["Collected prospectively at FNO; enters the model after its added value is tested.", "Prospektivně sbírán ve FNO; do modelu vstoupí po ověření přínosu."] },
      { s: "progress", t: ["Manuscript", "Rukopis"], d: ["Development and temporal validation, reported to TRIPOD+AI.", "Vývoj a časová validace, podle TRIPOD+AI."] },
      { s: "proposed", t: ["Completing recent follow-up", "Doplnění nedávného sledování"], d: ["Filling missing 24-h NIHSS and 90-day mRS since 2021 strengthens the later stages most.", "Doplnění chybějícího NIHSS za 24 h a mRS za 90 dní od roku 2021 nejvíce posílí pozdější fáze."] },
      { s: "proposed", t: ["External validation", "Externí validace"], d: ["Other Czech and Central European stroke centres, using the in-browser tool.", "Další česká a středoevropská iktová centra pomocí nástroje v prohlížeči."] },
      { s: "proposed", t: ["Prospective silent validation", "Prospektivní tichá validace"], d: [`Model v${A.B.version} frozen (fingerprint ${(A.B.fingerprint || "").slice(0, 12)}…) and scored on new patients without showing clinicians.`, `Model v${A.B.version} zmrazen (otisk ${(A.B.fingerprint || "").slice(0, 12)}…) a hodnocen na nových pacientech bez zobrazení lékařům.`] },
      { s: "proposed", t: ["Scheduled recalibration", "Plánovaná rekalibrace"], d: ["Under the written update policy, once 2026 outcomes are complete.", "Podle psaných pravidel aktualizace, až budou výsledky roku 2026 kompletní."] },
      { s: "proposed", t: ["Imaging and procedure detail", "Zobrazení a detaily výkonu"], d: ["CT perfusion core, collaterals, number of passes, eTICI 2c.", "Jádro infarktu z CT perfuze, kolaterály, počet průchodů, eTICI 2c."] },
      { s: A.DATA.federated ? "beta" : "proposed", tab: "federated", t: ["Federated learning", "Federované učení"],
        d: A.DATA.federated ? ["Working network: secure aggregation, exact federated boosting and validation; demonstrated on four virtual centres. Needs partner centres next.", "Funkční síť: bezpečná agregace, přesný federovaný boosting a validace; předvedeno na čtyřech virtuálních centrech. Dále potřebuje partnerská centra."]
          : ["A shared model trained across centres without moving data, coordinated on HPC.", "Společný model trénovaný napříč centry bez přesunu dat, koordinovaný na HPC."] },
      { s: "proposed", t: ["Clinicians vs model", "Lékaři vs model"], d: ["Compare prognostic estimates of neurologists and interventionalists with MTOP.", "Porovnat prognostické odhady neurologů a intervenčních radiologů s MTOP."] },
      { s: "proposed", t: ["Hospital system link", "Napojení na nemocniční systém"], d: ["Prefill inputs from the hospital record (FHIR), keeping computation local.", "Předvyplnění vstupů z nemocniční dokumentace (FHIR) s lokálním výpočtem."] },
      { s: "proposed", t: ["Regulatory pathway", "Regulatorní cesta"], d: ["Only if clinical use is pursued: EU MDR and AI Act requirements for decision-support software.", "Jen pokud se bude usilovat o klinické použití: požadavky EU MDR a AI Act na software pro podporu rozhodování."] },
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
