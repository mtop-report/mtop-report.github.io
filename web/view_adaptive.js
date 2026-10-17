/* MTOP "Adaptive" tab: how the model is kept up to date (results from src/mtop/adaptive.py). */
(function () {
  "use strict";
  const A = window.MTOPApp;
  if (!A || !A.DATA.adaptive) return;
  const { h, C, tx, num, signed, STAGES, STAGE } = A;
  const AD = A.DATA.adaptive;
  const S = A.S;
  S.adStage = S.adStage || "admission";
  S.adTarget = S.adTarget || "good_outcome";
  S.adShow = S.adShow || { refit_recal: true, frozen: true, refit: true, recal_int: true, recal_log: false, sliding: false };

  // colour follows the strategy (fixed order), MTOP policy first
  const ORDER = ["refit_recal", "frozen", "refit", "recal_int", "recal_log", "sliding"];
  const COLOR = { refit_recal: "var(--c1)", frozen: "var(--c2)", refit: "var(--c3)", recal_int: "var(--c4)", recal_log: "var(--c5)", sliding: "var(--c6)" };
  const LABEL = Object.fromEntries(AD.strategies.map((s) => [s.key, s.label]));
  const lbl = (k) => A.L(LABEL[k]);

  function kpi(label, value, sub) {
    return h("div", { class: "kpi" }, h("div", { class: "l", text: label }), h("div", { class: "v", text: value }), sub ? h("div", { class: "s", text: sub }) : null);
  }

  function render(root) {
    root.replaceChildren();
    const st = S.adStage, tg = S.adTarget;
    const R = AD.results[st][tg];
    const set = AD.settings;
    root.appendChild(h("div", { class: "view-head" }, h("div", null,
      h("h2", { text: tx("Keeping the model up to date", "Jak udržet model aktuální") }),
      h("p", { text: tx(
        `Practice and outcomes change, so a prediction model is only as good as its last update. Here a model trained on patients up to ${set.base_end} is followed through ${set.last_test}, and six ways of updating it are compared year by year, each using only data that would have been available at the time.`,
        `Praxe i výsledky se mění, a proto je predikční model jen tak dobrý jako jeho poslední aktualizace. Model natrénovaný na pacientech do roku ${set.base_end} je zde sledován až do roku ${set.last_test} a šest způsobů jeho aktualizace je porovnáno rok po roce, vždy jen s daty, která by byla v dané době k dispozici.`) }))));

    // controls
    const seg = h("div", { class: "mini-seg", role: "group", "aria-label": A.t().outcome },
      A.TARGETS.map((k) => h("button", { type: "button", "aria-pressed": String(tg === k), onclick: () => { S.adTarget = k; render(root); } }, k === "good_outcome" ? A.t().targetGood : A.t().targetDeath)));
    const stSel = h("select", { class: "sel", id: "ad-stage" });
    for (const s2 of STAGES) stSel.appendChild(h("option", { value: s2, selected: s2 === st ? true : null }, `${A.idx(s2) + 1} · ${A.L(STAGE[s2].title)}`));
    stSel.addEventListener("change", () => { S.adStage = stSel.value; render(root); });
    root.appendChild(h("div", { class: "controls" }, h("span", { text: A.t().outcome }), seg, h("label", { for: "ad-stage" }, A.t().stage), stSel));

    // headline tiles
    const P = R.pooled;
    const firstWarn = R.monitor.find((m) => m.flag !== "ok");
    const firstAlarm = R.monitor.find((m) => m.flag === "alarm");
    root.appendChild(h("div", { class: "kpis" },
      kpi(tx(`First drift signal, model frozen in ${set.base_end}`, `První signál driftu, model zmrazený v roce ${set.base_end}`),
        firstWarn ? String(firstWarn.year) : tx("none", "žádný"),
        firstAlarm ? tx(`alarm (beyond 3σ) in ${firstAlarm.year}`, `alarm (za 3σ) v roce ${firstAlarm.year}`) : tx("no 3σ alarm", "bez alarmu za 3σ")),
      kpi(tx(`Calibration-in-the-large ${set.first_test}–${set.last_test}`, `Kalibrace celkově ${set.first_test}–${set.last_test}`),
        `${signed(P.frozen.citl)} → ${signed(P.refit_recal.citl)}`, tx("frozen → MTOP policy (0 is ideal)", "zmrazený → postup MTOP (ideál 0)")),
      kpi(tx("Discrimination (AUROC)", "Diskriminace (AUROC)"), `${num(P.frozen.auroc, 3)} → ${num(P.refit_recal.auroc, 3)}`, tx("frozen → MTOP policy", "zmrazený → postup MTOP")),
      kpi(tx("Calibration error (ICI)", "Chyba kalibrace (ICI)"), `${num(P.frozen.ici, 3)} → ${num(P.refit_recal.ici, 3)}`, tx("lower is better", "nižší je lepší"))));

    // drift monitor
    const mon = R.monitor;
    const monCard = h("div", { class: "card card-pad block" });
    const yrsM = mon.map((m) => String(m.year));
    const fm = C.figure(monCard, {
      title: tx(`Drift monitor: model frozen in ${set.base_end}`, `Monitor driftu: model zmrazený v roce ${set.base_end}`),
      sub: tx("Observed outcome rate against the model's expectation, with 95% control limits. A drift signal is raised when a year falls outside the limits.",
        "Pozorovaný podíl výsledku proti očekávání modelu s 95% kontrolními mezemi. Signál driftu vzniká, když rok vybočí z mezí."),
      legend: [{ label: tx("Expected (95% limits)", "Očekáváno (95% meze)"), color: "var(--ink-2)", type: "line" }, { label: tx("Observed", "Pozorováno"), color: "var(--c1)", type: "line" }],
      table: { columns: [tx("Year", "Rok"), "n", tx("Observed", "Pozorováno"), tx("Expected", "Očekáváno"), "z", tx("Signal", "Signál")],
        rows: mon.map((m) => [m.year, m.n, A.pct(m.obs, 1), A.pct(m.exp, 1), num(m.z, 2), flagText(m.flag)]) },
      labels: { table: A.t().showTable, chart: A.t().showChart } });
    const lo = Math.max(0, Math.min(...mon.map((m) => Math.min(m.obs, m.lo3))) - 0.05);
    const hi = Math.min(1, Math.max(...mon.map((m) => Math.max(m.obs, m.hi3))) + 0.05);
    C.lineChart(fm.body, { height: 230, x: { type: "cat", cats: yrsM }, y: { domain: [lo, hi], fmt: (v) => A.pct(v) },
      series: [
        { label: tx("Expected", "Očekáváno"), color: "var(--ink-2)", width: 1.5, points: mon.map((m, i) => ({ x: i, y: m.exp, lo: m.lo95, hi: m.hi95 })) },
        { label: tx("Observed", "Pozorováno"), color: "var(--c1)", dots: true, points: mon.map((m, i) => ({ x: i, y: m.obs, n: m.n, flag: m.flag })) },
      ],
      tipFmt: (v, p, se) => (se.label === tx("Observed", "Pozorováno") ? `${A.pct(v, 1)} · ${flagText(p.flag)}` : `${A.pct(v, 1)} (${A.pct(p.lo, 1)}–${A.pct(p.hi, 1)})`),
      aria: "drift monitor" });
    const chips = h("div", { class: "chips" });
    for (const m of mon) if (m.flag !== "ok") chips.appendChild(flagChip(m));
    if (chips.childNodes.length) monCard.appendChild(chips);
    root.appendChild(monCard);

    // strategy toggles
    const tog = h("div", { class: "chips toggles", role: "group", "aria-label": tx("Strategies", "Strategie") });
    for (const k of ORDER) {
      const b = h("button", { type: "button", class: "chip-btn", "aria-pressed": String(!!S.adShow[k]) }, h("i", { style: `background:${COLOR[k]}` }), lbl(k));
      b.addEventListener("click", () => { S.adShow[k] = !S.adShow[k]; render(root); });
      tog.appendChild(b);
    }
    root.appendChild(h("div", { class: "block" }, h("div", { class: "chart-sub", style: "margin-bottom:6px", text: tx("Show strategies:", "Zobrazit strategie:") }), tog));

    const shown = ORDER.filter((k) => S.adShow[k]);
    const yrs = R.years.map((y) => String(y.year));
    const two = h("div", { class: "grid-2 block" });
    const cCard = h("div", { class: "card card-pad" });
    const fc = C.figure(cCard, {
      title: tx("Calibration-in-the-large by year", "Kalibrace celkově podle roku"),
      sub: tx("Below 0: the model expects better outcomes than were observed", "Pod 0: model očekává lepší výsledky, než byly pozorovány"),
      legend: shown.map((k) => ({ label: lbl(k), color: COLOR[k], type: "line" })),
      table: { columns: [tx("Year", "Rok")].concat(shown.map(lbl)), rows: R.years.map((y) => [y.year].concat(shown.map((k) => signed(y.by[k].citl)))) },
      labels: { table: A.t().showTable, chart: A.t().showChart } });
    const allC = R.years.flatMap((y) => shown.map((k) => y.by[k].citl));
    C.lineChart(fc.body, { height: 230, x: { type: "cat", cats: yrs }, y: { domain: [Math.min(-0.5, Math.floor(Math.min(...allC) * 4) / 4), Math.max(0.5, Math.ceil(Math.max(...allC) * 4) / 4)], fmt: (v) => num(v, 1) },
      zeroLine: true, series: shown.slice().reverse().map((k) => ({ label: lbl(k), color: COLOR[k], width: k === "refit_recal" ? 2.5 : 1.75, dots: true, points: R.years.map((y, i) => ({ x: i, y: y.by[k].citl })) })),
      tipFmt: (v) => signed(v), aria: "calibration by year" });
    two.appendChild(cCard);

    const aCard = h("div", { class: "card card-pad" });
    const fa = C.figure(aCard, {
      title: tx("Discrimination by year (AUROC)", "Diskriminace podle roku (AUROC)"),
      sub: tx(`Shown when a year has at least ${set.min_events} patients in each outcome group`, `Zobrazeno, má-li rok alespoň ${set.min_events} pacientů v každé skupině výsledku`),
      legend: shown.map((k) => ({ label: lbl(k), color: COLOR[k], type: "line" })),
      table: { columns: [tx("Year", "Rok"), "n"].concat(shown.map(lbl)), rows: R.years.map((y) => [y.year, y.n].concat(shown.map((k) => (y.by[k].auroc === null ? "–" : num(y.by[k].auroc, 3))))) },
      labels: { table: A.t().showTable, chart: A.t().showChart } });
    C.lineChart(fa.body, { height: 230, x: { type: "cat", cats: yrs }, y: { domain: [0.5, 1], ticks: [0.5, 0.6, 0.7, 0.8, 0.9, 1], fmt: (v) => num(v, 1) },
      series: shown.slice().reverse().map((k) => ({ label: lbl(k), color: COLOR[k], width: k === "refit_recal" ? 2.5 : 1.75, dots: true, points: R.years.map((y, i) => ({ x: i, y: y.by[k].auroc })) })),
      tipFmt: (v) => num(v, 3), aria: "auroc by year" });
    aCard.appendChild(h("p", { class: "note", text: tx(
      "Recalibration shifts every estimate the same way, so it leaves AUROC unchanged: recalibrated lines lie exactly on their base model (MTOP policy on top of refit, intercept recalibration on top of frozen).",
      "Rekalibrace posouvá všechny odhady stejně, a AUROC proto nemění: rekalibrované čáry leží přesně na svém základním modelu (postup MTOP na přetrénování, rekalibrace interceptu na zmrazeném modelu).") }));
    two.appendChild(aCard);
    root.appendChild(two);

    // pooled table
    const tCard = h("div", { class: "card card-pad block" });
    tCard.appendChild(h("div", { class: "chart-head" }, h("div", null,
      h("div", { class: "chart-title", text: tx(`All test years pooled (${set.first_test}–${set.last_test})`, `Všechny testovací roky dohromady (${set.first_test}–${set.last_test})`) }),
      h("div", { class: "chart-sub", text: `n = ${num(P.frozen.n)} · ${tx("events", "události")} ${num(P.frozen.events)}` }))));
    const bestAuc = Math.max(...ORDER.map((k) => P[k].auroc));
    const bestIci = Math.min(...ORDER.map((k) => P[k].ici));
    const tbl = h("table", { class: "data" },
      h("thead", null, h("tr", null, [tx("Strategy", "Strategie"), "AUROC (95% CI)", tx("Cal. intercept", "Kal. intercept"), tx("Cal. slope", "Kal. sklon"), "Brier", "ICI", tx("Mean |yearly intercept|", "Průměr |ročního interceptu|")].map((x) => h("th", { scope: "col", text: x })))),
      h("tbody", null, ORDER.map((k) => {
        const r = P[k];
        return h("tr", null,
          h("th", { scope: "row" }, h("span", { class: "key" }, h("i", { class: "line", style: `background:${COLOR[k]}` }), lbl(k))),
          h("td", { class: r.auroc === bestAuc ? "best" : "" }, `${num(r.auroc, 3)} `, h("span", { class: "ci", text: `${num(r.auroc_ci[0], 3)}–${num(r.auroc_ci[1], 3)}` })),
          h("td", { text: signed(r.citl) }), h("td", { text: num(r.slope, 2) }), h("td", { text: num(r.brier, 3) }),
          h("td", { class: r.ici === bestIci ? "best" : "", text: num(r.ici, 3) }), h("td", { text: num(r.mean_abs_citl, 2) }));
      })));
    tCard.appendChild(h("div", { class: "table-scroll" }, tbl));
    tCard.appendChild(h("p", { class: "note", text: tx(
      `Every model uses the same fixed hyperparameters (number of trees chosen by cross-validation within each training set). Recalibration uses the previous ${set.recal_years} years; the sliding window uses the previous ${set.window}. Refitting can also pick up predictors that started being recorded later, such as ASPECTS from 2016, which recalibration alone cannot.`,
      `Všechny modely mají stejné pevné hyperparametry (počet stromů zvolen křížovou validací v rámci každé trénovací sady). Rekalibrace používá předchozí ${set.recal_years} roky, posuvné okno předchozích ${set.window} let. Přetrénování navíc využije prediktory, které se začaly evidovat později, např. ASPECTS od roku 2016, což samotná rekalibrace nedokáže.`) }));
    root.appendChild(tCard);

    // availability heat grid
    const av = AD.availability;
    const hCard = h("div", { class: "card card-pad block" });
    const fh = C.figure(hCard, {
      title: tx("What the registry recorded, year by year", "Co registr v jednotlivých letech zaznamenal"),
      sub: tx("Share of treated adults with each item recorded (%). New items and new gaps both change what a model can learn.", "Podíl léčených dospělých se zaznamenanou položkou (%). Nové položky i nové mezery mění, co se model může naučit."),
      table: { columns: [tx("Item", "Položka")].concat(av.years), rows: av.variables.map((v) => [A.L(v.label)].concat(v.values.map((x) => (x === null ? "–" : num(x))))) },
      labels: { table: A.t().showTable, chart: A.t().showChart } });
    C.heat(fh.body, { rows: av.variables.map((v) => ({ label: A.L(v.label), values: v.values })), cols: av.years, fmt: (v) => `${num(v)} %`, valueLabel: tx("recorded", "zaznamenáno"), aria: "data availability" });
    root.appendChild(hCard);

    // policy + versions
    const two2 = h("div", { class: "grid-2 block" });
    two2.appendChild(policyCard());
    two2.appendChild(versionsCard());
    root.appendChild(two2);
  }

  function flagText(f) {
    return f === "alarm" ? tx("alarm (beyond 3σ)", "alarm (za 3σ)") : f === "warning" ? tx("warning (beyond 95%)", "varování (za 95 %)") : tx("in control", "v mezích");
  }
  function flagChip(m) {
    const cls = m.flag === "alarm" ? "st-alarm" : "st-warn";
    return h("span", { class: `status ${cls}` }, h("i", { "aria-hidden": "true" }), `${m.year}: ${flagText(m.flag)}`);
  }

  function policyCard() {
    const card = h("div", { class: "card card-pad" });
    card.appendChild(h("div", { class: "chart-head" }, h("div", null,
      h("div", { class: "chart-title", text: tx("Update policy (draft)", "Pravidla aktualizace (návrh)") }),
      h("div", { class: "chart-sub", text: tx("Written in advance, in the spirit of a predetermined change control plan", "Stanovena předem, v duchu předem schváleného plánu změn") }))));
    const rows = [
      [tx("Each quarter", "Každé čtvrtletí"), tx("Observed vs expected outcome for patients whose 90-day mRS is complete", "Pozorovaný vs očekávaný výsledek u pacientů s dokončeným mRS za 90 dní"),
        tx("Outside 95% limits twice in a row → re-estimate the intercept on the last 2 years; minor version", "Dvakrát po sobě mimo 95% meze → přepočet interceptu na posledních 2 letech; menší verze")],
      [tx("Each year", "Každý rok"), tx("AUROC and calibration slope on the latest year", "AUROC a kalibrační sklon na posledním roce"),
        tx("AUROC below its validation interval or slope outside 0.8–1.2 → full refit with temporal validation; major version", "AUROC pod validačním intervalem nebo sklon mimo 0,8–1,2 → úplné přetrénování s časovou validací; vyšší verze")],
      [tx("New predictor", "Nový prediktor"), tx("For example GFAP, once at least 100 outcome events are available", "Například GFAP, jakmile bude k dispozici alespoň 100 událostí"),
        tx("Adopt only if it improves discrimination and calibration on held-out recent patients", "Přijmout jen tehdy, zlepší-li diskriminaci i kalibraci na odložených nedávných pacientech")],
      [tx("Every change", "Každá změna"), tx("Model fingerprint, changelog and validation report", "Otisk modelu, seznam změn a validační zpráva"),
        tx("Published on this page and in the code repository before use", "Zveřejněno na této stránce a v repozitáři kódu před použitím")],
    ];
    const tbl = h("table", { class: "data policy" },
      h("thead", null, h("tr", null, [tx("When", "Kdy"), tx("Check", "Kontrola"), tx("Action", "Akce")].map((x) => h("th", { scope: "col", text: x })))),
      h("tbody", null, rows.map((r) => h("tr", null, h("th", { scope: "row", text: r[0] }), h("td", { text: r[1] }), h("td", { text: r[2] })))));
    card.appendChild(h("div", { class: "table-scroll" }, tbl));
    card.appendChild(h("p", { class: "note", text: tx("Proposed thresholds; to be agreed by the study team before prospective use.", "Navržené prahy; před prospektivním použitím je musí schválit řešitelský tým.") }));
    return card;
  }

  function versionsCard() {
    const B = A.B;
    const card = h("div", { class: "card card-pad" });
    card.appendChild(h("div", { class: "chart-title", text: tx("Model versions", "Verze modelu"), style: "margin-bottom:10px" }));
    const items = [
      { v: `v${B.version}`, date: B.generated, status: "live",
        text: tx(`Trained on 2005–2025, intercepts recalibrated to 2021–2025. Fingerprint ${(B.fingerprint || "").slice(0, 12)}…`, `Trénováno na letech 2005–2025, intercepty rekalibrovány na 2021–2025. Otisk ${(B.fingerprint || "").slice(0, 12)}…`) },
      { v: "v1.1", date: tx("after 2026 follow-up", "po dokončení sledování 2026"), status: "proposed",
        text: tx("Scheduled recalibration once 90-day outcomes of 2026 patients are complete.", "Plánovaná rekalibrace po dokončení 90denních výsledků pacientů z roku 2026.") },
      { v: "v2.0", date: tx("when data allow", "až to data dovolí"), status: "candidate",
        text: tx("Refit with new predictors (serum GFAP; imaging if recorded), external validation first.", "Přetrénování s novými prediktory (sérový GFAP; zobrazení, bude-li evidováno), nejprve externí validace.") },
    ];
    const list = h("ol", { class: "timeline" });
    for (const it of items) {
      list.appendChild(h("li", { class: `tl-${it.status}` },
        h("div", { class: "tl-head" }, h("b", { text: it.v }), h("span", { class: "tl-date", text: it.date }), A.statusChip(it.status)),
        h("p", { text: it.text })));
    }
    card.appendChild(list);
    return card;
  }

  A.registerView("adaptive", () => tx("Adaptive", "Adaptace"), render);
})();
