/* MTOP "Multicentre" tab: validation on a partner's own data, inside the browser, and pooling
 * of results across sites. No file content is uploaded anywhere. */
(function () {
  "use strict";
  const A = window.MTOPApp;
  if (!A) return;
  const MX = window.MTOPMetrics;
  const { h, C, tx, num, signed, B, STAGES, STAGE, FEAT } = A;
  const S = A.S;
  S.netStage = S.netStage || "admission";
  S.netTarget = S.netTarget || "good_outcome";
  const NS = { file: null, records: null, issues: null, results: null, preds: null, imports: [], includeFno: true, label: "", running: false, progress: 0, error: null };

  const KEYS = B.features.filter((f) => f.type !== "derived").map((f) => f.key).concat(["mrs90"]);
  const POSTERIOR = new Set(["basilar", "other_posterior"]);
  const norm = (s) => String(s).normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase().replace(/[^a-z0-9]/g, "");
  const SYN = {
    age: ["age", "vek"], sex_male: ["sexmale", "male", "sex", "pohlavi", "gender"], mrs_pre: ["mrspre", "premrs", "mrsbefore", "prestrokemrs"],
    nihss_adm: ["nihssadm", "nihss", "admissionnihss", "baselinenihss", "nihss0"], aspects: ["aspects"], occlusion: ["occlusion", "occlusionsite", "okluze"],
    side_left: ["sideleft", "leftside", "hemisphereleft"], onset_to_puncture: ["onsettopuncture", "otp", "onsettopuncturemin", "strokeonsettopuncturemin"],
    ivt: ["ivt", "ivthrombolysis"], ivt_external: ["ivtexternal", "ivtotherhospital", "ivtindifferenthospital", "dripandshipivt"],
    onset_to_ivt: ["onsettoivt", "otn", "onsettoneedle", "strokeonsettoivtmin"], transfer: ["transfer", "dripandship", "transferred"],
    onset_hour: ["onsethour", "hourofonset"], weekend: ["weekend", "weekendonset"],
    glucose: ["glucose", "glycemia", "glykemie"], sbp: ["sbp", "systolicbp"], dbp: ["dbp", "diastolicbp"], bmi: ["bmi"],
    htn: ["htn", "hypertension"], dm: ["dm", "diabetes"], hld: ["hld", "hyperlipidemia", "dyslipidemia"], arrhythmia: ["arrhythmia", "af", "atrialfibrillation"],
    stroke_prior: ["strokeprior", "priorstroke", "previousstroke"], tia_prior: ["tiaprior", "priortia", "previoustia"], smoking: ["smoking", "smoker"],
    alcohol: ["alcohol", "alcoholabuse"], statin: ["statin", "statins", "statinsbefore"], heart3: ["heart3", "heartcondition", "heartdisease"],
    antithrombotic: ["antithrombotic", "antithromboticsbefore", "antithrombotics"],
    tici: ["tici", "mtici", "etici", "finaltici"], technique: ["technique", "firstlinetechnique"], stenting: ["stenting", "acutestenting", "stent"],
    angioplasty: ["angioplasty", "pta"], ia_lysis: ["ialysis", "iat", "intraarteriallysis"], procedure_min: ["proceduremin", "procedureduration"],
    onset_to_recan: ["onsettorecan", "onsettorecanalization", "onsettoreperfusion"], anesthesia: ["anesthesia", "anaesthesia"],
    nihss_24h: ["nihss24h", "nihss24"], sich: ["sich"], ich_other: ["ichother", "nonsich", "asymptomaticich"], nihss_7d: ["nihss7d", "nihss7"],
    edema: ["edema", "oedema", "cerebraledema"], decompression: ["decompression", "decompressivecraniectomy", "hemicraniectomy"],
    mrs90: ["mrs90", "90daymrs", "mrs3m", "mrs3months"],
  };
  const BIN = { "1": 1, "0": 0, yes: 1, no: 0, y: 1, n: 0, true: 1, false: 0, ano: 1, ne: 0 };
  const SEX = { "1": 1, "0": 0, m: 1, male: 1, man: 1, muz: 1, f: 0, female: 0, woman: 0, w: 0, z: 0, zena: 0 };
  const SIDE = { "1": 1, "0": 0, left: 1, l: 1, leva: 1, right: 0, r: 0, prava: 0 };
  const CAT = {
    occlusion: { ica: "ICA", icat: "ICA", t: "ICA", intracranialica: "ICA", m1: "M1", m2: "distal", m3: "distal", aca: "distal", distal: "distal",
      cervicalica: "cervical_ICA", tandem: "cervical_ICA", cca: "cervical_ICA", basilar: "basilar", ba: "basilar", vertebral: "other_posterior", va: "other_posterior", pca: "other_posterior", otherposterior: "other_posterior" },
    technique: { aspiration: "aspiration", adapt: "aspiration", stentretriever: "stent_retriever", sr: "stent_retriever", combined: "combined", solumbra: "combined", none: "none",
      diagnostic: "diagnostic", pag: "diagnostic", diagnosticangiography: "diagnostic" },
    heart3: { none: "none", no: "none", zadne: "none", coronary: "coronary", chd: "coronary", cad: "coronary", ichs: "coronary", other: "other" },
    antithrombotic: { none: "none", no: "none", zadna: "none", antiplatelet: "antiplatelet", asa: "antiplatelet", aspirin: "antiplatelet", clopidogrel: "antiplatelet",
      anticoagulant: "anticoagulant", anticoagulation: "anticoagulant", warfarin: "anticoagulant", noac: "anticoagulant", doac: "anticoagulant", lmwh: "anticoagulant" },
    anesthesia: { general: "general", ga: "general", sedation: "sedation", cs: "sedation", consciousseda: "sedation", local: "local", la: "local" },
  };
  const TICI = { "0": 0, "1": 1, "2a": 2, "2": 2.5, "2b": 3, "2c": 4, "3": 4 };

  function parseValue(key, raw, issues) {
    if (raw === null || raw === undefined) return null;
    const s = String(raw).trim();
    if (s === "" || /^(na|nan|null|unknown|-)$/i.test(s)) return null;
    const f = FEAT[key];
    const bad = () => { issues[key] = (issues[key] || 0) + 1; return null; };
    if (key === "mrs90") { const v = Number(s.replace(",", ".")); return Number.isInteger(v) && v >= 0 && v <= 6 ? v : bad(); }
    if (key === "tici") { const k = norm(s).replace(/^m?e?tici/, ""); return k in TICI ? TICI[k] : bad(); }
    if (key === "sex_male") { const k = norm(s); return k in SEX ? SEX[k] : bad(); }
    if (key === "side_left") { const k = norm(s); return k in SIDE ? SIDE[k] : bad(); }
    if (f.type === "binary") { const k = norm(s); return k in BIN ? BIN[k] : bad(); }
    if (f.type === "categorical") {
      const k = norm(s);
      const direct = f.options.find((o) => norm(o.value) === k);
      if (direct) return direct.value;
      const m = (CAT[key] || {})[k];
      return m === undefined ? bad() : m;
    }
    const v = Number(s.replace(",", "."));
    if (Number.isNaN(v)) return bad();
    if (f.type === "ordinal") return f.options.some((o) => o.value === v) ? v : bad();
    if ((f.min !== undefined && v < f.min) || (f.max !== undefined && v > f.max)) return bad();
    return v;
  }

  // ------------------------------------------------------------------ file reading
  function parseCSV(text) {
    const first = text.split(/\r?\n/, 1)[0] || "";
    const cnt = (c) => first.split(c).length - 1;
    const delim = [",", ";", "\t"].reduce((a, c) => (cnt(c) > cnt(a) ? c : a), ",");
    const rows = [];
    let row = [], field = "", q = false;
    for (let i = 0; i < text.length; i++) {
      const ch = text[i];
      if (q) {
        if (ch === '"') { if (text[i + 1] === '"') { field += '"'; i++; } else q = false; } else field += ch;
      } else if (ch === '"') q = true;
      else if (ch === delim) { row.push(field); field = ""; }
      else if (ch === "\n" || ch === "\r") {
        if (ch === "\r" && text[i + 1] === "\n") i++;
        row.push(field); field = "";
        if (row.length > 1 || row[0] !== "") rows.push(row);
        row = [];
      } else field += ch;
    }
    if (field !== "" || row.length) { row.push(field); rows.push(row); }
    const header = rows.shift() || [];
    return { header, rows: rows.map((r) => Object.fromEntries(header.map((hd, j) => [hd, r[j]]))) };
  }

  function loadScript(src) {
    return new Promise((res, rej) => {
      const el = document.createElement("script");
      el.src = src; el.onload = res; el.onerror = () => rej(new Error("load failed"));
      document.head.appendChild(el);
    });
  }

  async function readFile(file) {
    if (/\.xlsx?$/i.test(file.name)) {
      if (!window.XLSX) await loadScript("https://cdnjs.cloudflare.com/ajax/libs/xlsx/0.18.5/xlsx.full.min.js");
      const wb = window.XLSX.read(await file.arrayBuffer(), { type: "array" });
      const rows = window.XLSX.utils.sheet_to_json(wb.Sheets[wb.SheetNames[0]], { defval: "", raw: false });
      return { header: rows.length ? Object.keys(rows[0]) : [], rows };
    }
    return parseCSV(await file.text());
  }

  function mapColumns(header) {
    const map = {};
    const byNorm = Object.fromEntries(header.map((c) => [norm(c), c]));
    for (const k of KEYS) {
      if (byNorm[norm(k)] !== undefined) { map[k] = byNorm[norm(k)]; continue; }
      const hit = (SYN[k] || []).find((sn) => byNorm[sn] !== undefined);
      if (hit) map[k] = byNorm[hit];
    }
    return map;
  }

  function toRecords(table) {
    const map = mapColumns(table.header);
    const issues = {};
    const records = table.rows.map((r) => {
      const x = {};
      for (const k of KEYS) x[k] = map[k] === undefined ? null : parseValue(k, r[map[k]], issues);
      if (POSTERIOR.has(x.occlusion)) x.side_left = 0;
      return x;
    });
    return { records, map, issues };
  }

  // ------------------------------------------------------------------ validation run
  const tick = () => new Promise((r) => setTimeout(r, 0));
  const eligible = (x, st) => x.mrs90 !== null && (A.idx(st) < 1 || x.tici !== null)
    && (st !== "h24" || x.nihss_24h !== null) && (st !== "d7" || x.nihss_7d !== null);
  // every published learner of a stage and outcome (the calculator's choice first)
  const learnersOf = (st, tg) => {
    const e = A.entryOf(st, tg);
    return [e.chosen].concat(Object.keys(e.learners).filter((k) => k !== e.chosen));
  };
  const ALL_LEARNERS = Array.from(new Set(STAGES.flatMap((st) => A.TARGETS.flatMap((tg) => Object.keys(A.entryOf(st, tg).learners)))));
  S.netLearner = S.netLearner || null;
  const netLearner = (st, tg) => (S.netLearner && A.entryOf(st, tg).learners[S.netLearner] ? S.netLearner : A.chosenOf(st, tg));

  async function run(root) {
    NS.running = true; NS.progress = 0; NS.results = null; render(root);
    const recs = NS.records;
    const preds = recs.map(() => ({}));
    const total = recs.length * STAGES.length;
    let done = 0;
    for (const st of STAGES) {
      for (let i = 0; i < recs.length; i++) {
        if (eligible(recs[i], st)) {
          const x = recs[i];
          const row = {};
          for (const tg of A.TARGETS) {
            row[tg] = {};
            for (const nm of learnersOf(st, tg)) row[tg][nm] = A.predictStage(st, x, tg, nm).p;
          }
          preds[i][st] = row;
        }
        done++;
        if (done % 10 === 0) { NS.progress = done / total; updateProgress(root); await tick(); }
      }
    }
    const results = {};
    for (const st of STAGES) {
      results[st] = {};
      const idxs = recs.map((x, i) => i).filter((i) => preds[i][st]);
      for (const tg of A.TARGETS) {
        results[st][tg] = {};
        const y = idxs.map((i) => (tg === "good_outcome" ? (recs[i].mrs90 <= 2 ? 1 : 0) : (recs[i].mrs90 === 6 ? 1 : 0)));
        for (const nm of learnersOf(st, tg)) {
          const p = idxs.map((i) => preds[i][st][tg][nm]);
          results[st][tg][nm] = idxs.length >= 20 ? MX.summary(y, p, { B: 500, minCell: 10 }) : { n: idxs.length, insufficient: true };
          await tick();
        }
      }
    }
    NS.preds = preds;
    NS.results = results;
    NS.running = false;
    render(root);
  }

  function updateProgress(root) {
    const bar = root.querySelector(".progress i");
    if (bar) bar.style.width = `${Math.round(NS.progress * 100)}%`;
  }

  // results file: every published learner per stage and outcome (format 2); "chosen" names the
  // calculator's model, whose numbers also sit at the stage level for readers of format 1
  function exportJSON() {
    const out = {
      format: "mtop-validation/2", model_version: B.version, model_fingerprint: B.fingerprint || null,
      site: NS.label || "unnamed site", generated: new Date().toISOString().slice(0, 10),
      rows: NS.records.length, columns_found: Object.keys(NS.map || {}), stages: {},
    };
    const pick = (r) => (r.insufficient ? { n: r.n, insufficient: true }
      : { n: r.n, events: r.events, auroc: r.auroc, auroc_ci: r.auroc_ci, se_logit: r.se_logit, brier: r.brier, citl: r.citl, slope: r.slope, calibration: r.calibration });
    for (const st of STAGES) {
      out.stages[st] = {};
      for (const tg of A.TARGETS) {
        const ch = A.chosenOf(st, tg);
        const per = {};
        for (const [nm, r] of Object.entries(NS.results[st][tg])) per[nm] = pick(r);
        out.stages[st][tg] = Object.assign({ chosen: ch, models: per }, per[ch]);
      }
    }
    return JSON.stringify(out, null, 1);
  }
  // a site's result for one learner from either file format
  const siteResult = (j, st, tg, nm) => {
    const r = (j.stages[st] || {})[tg];
    if (!r) return null;
    if (r.models) return r.models[nm] || null;
    return nm === A.chosenOf(st, tg) ? r : null;
  };

  function copyText(text, btn) {
    const ok = () => { const old = btn.textContent; btn.textContent = tx("Copied", "Zkopírováno"); setTimeout(() => (btn.textContent = old), 1500); };
    try {
      navigator.clipboard.writeText(text).then(ok, () => fallback());
    } catch (e) { fallback(); }
    function fallback() {
      const ta = h("textarea", { class: "copy-fallback", readonly: true });
      ta.value = text;
      btn.after(ta);
      ta.select();
    }
  }

  function downloadLink(text, name, mime) {
    const url = URL.createObjectURL(new Blob([text], { type: mime }));
    return h("a", { class: "btn", href: url, download: name }, tx("Download", "Stáhnout"));
  }

  // ------------------------------------------------------------------ render
  // The tab has two parts: learning together (federated network, view_federated.js) and validating
  // MTOP on a centre's own data in this browser (below).
  S.netMode = S.netMode || "federated";
  function modeSwitch() {
    const modes = [
      ["federated", tx("Learn together", "Učit se společně"), tx("Federated network · working demo", "Federovaná síť · funkční ukázka")],
      ["validate", tx("Validate on your data", "Ověřit na vlastních datech"), tx("In this browser · available now", "V tomto prohlížeči · dostupné hned")],
    ];
    return h("div", { class: "mode-switch", role: "tablist", "aria-label": tx("Multicentre", "Multicentrum") },
      modes.map(([key, title, sub]) => h("button", {
        type: "button", role: "tab", "aria-selected": String(S.netMode === key),
        onclick: () => { if (S.netMode !== key) A.goTo(key); },
      }, h("b", { text: title }), h("span", { text: sub }))));
  }

  function render(root) {
    root.replaceChildren();
    root.appendChild(modeSwitch());
    const body = h("div", { class: "mode-body" });
    root.appendChild(body);
    if (S.netMode === "federated" && A.renderFederated) A.renderFederated(body);
    else renderValidate(body, root);
  }

  function renderValidate(body, root) {
    body.appendChild(h("div", { class: "view-head" }, h("div", null,
      h("h2", { text: tx("Multicentre validation", "Multicentrická validace") }),
      h("p", { text: tx(
        "A model from one hospital has to be tested elsewhere before it can be trusted there. MTOP is built so that this needs no data sharing: the model travels to the data, and only summary numbers come back.",
        "Model z jedné nemocnice je třeba ověřit jinde, než mu tam lze věřit. MTOP je postaven tak, aby to nevyžadovalo sdílení dat: model putuje k datům a zpět se vracejí jen souhrnná čísla.") }))));

    body.appendChild(h("ol", { class: "flow" },
      [[tx("Open model", "Otevřený model"), tx("Code and trained model are public; this page runs offline.", "Kód i natrénovaný model jsou veřejné; stránka funguje i offline.")],
        [tx("Local run", "Lokální výpočet"), tx("A partner centre loads its own registry export here. Nothing is uploaded.", "Partnerské centrum zde načte vlastní export registru. Nic se nenahrává.")],
        [tx("Summary only", "Jen souhrn"), tx("The page produces a results file with aggregate metrics, no patient rows.", "Stránka vytvoří soubor výsledků se souhrnnými metrikami, bez řádků pacientů.")],
        [tx("Pooled evidence", "Souhrnný důkaz"), tx("Results from all centres are combined in a random-effects meta-analysis.", "Výsledky všech center se spojí metaanalýzou s náhodnými efekty.")]]
        .map(([a, b], i) => h("li", null, h("span", { class: "flow-n", text: String(i + 1) }), h("div", null, h("b", { text: a }), h("p", { text: b }))))));

    body.appendChild(toolCard(root));
    body.appendChild(poolCard(root));
    body.appendChild(nextStepCard());
  }

  function toolCard(root) {
    const card = h("div", { class: "card card-pad block", id: "local-validation" });
    card.appendChild(h("div", { class: "chart-head" },
      h("div", null, h("div", { class: "chart-title", text: tx("Validate MTOP on your own data", "Ověřte MTOP na vlastních datech") }),
        h("div", { class: "chart-sub", text: tx("CSV or Excel, one row per patient, columns as in the template below", "CSV nebo Excel, jeden řádek na pacienta, sloupce podle šablony níže") })),
      h("span", { class: "status st-live" }, h("i", { "aria-hidden": "true" }), tx("Your file never leaves this browser", "Soubor nikdy neopustí tento prohlížeč"))));

    const input = h("input", { type: "file", id: "lv-file", accept: ".csv,.txt,.xlsx,.xls", class: "visually-hidden" });
    input.addEventListener("change", () => input.files[0] && loadFile(input.files[0], root));
    const drop = h("label", { class: "dropzone", for: "lv-file" },
      h("b", { text: tx("Choose a file or drop it here", "Vyberte soubor nebo jej sem přetáhněte") }),
      h("span", { text: NS.file ? `${NS.file}` : tx("CSV (comma or semicolon) or Excel", "CSV (čárka nebo středník) nebo Excel") }));
    drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
    drop.addEventListener("dragleave", () => drop.classList.remove("over"));
    drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("over"); if (e.dataTransfer.files[0]) loadFile(e.dataTransfer.files[0], root); });
    const demo = h("button", { class: "btn", type: "button" }, tx("Try with synthetic demo data", "Vyzkoušet na syntetických datech"));
    demo.addEventListener("click", () => {
      NS.file = tx("Synthetic demo cohort (randomly generated, not real patients)", "Syntetická ukázková kohorta (náhodně vygenerovaná, nejde o skutečné pacienty)");
      ingest(parseCSV(A.DATA.demo_csv || ""), root);
      NS.label = "Synthetic demo";
    });
    card.appendChild(h("div", { class: "tool-row" }, input, drop, h("div", { class: "tool-side" }, demo, templateDetails())));
    if (NS.error) card.appendChild(h("p", { class: "notice", text: NS.error }));

    if (NS.records && !NS.results && !NS.running) {
      card.appendChild(mappingSummary());
      const go = h("button", { class: "btn btn-primary", type: "button" }, tx(`Run MTOP on ${num(NS.records.length)} rows`, `Spustit MTOP na ${num(NS.records.length)} řádcích`));
      go.addEventListener("click", () => run(root));
      card.appendChild(h("div", { class: "tool-actions" }, go));
    }
    if (NS.running) card.appendChild(h("div", { class: "progress", role: "progressbar", "aria-label": tx("Computing", "Výpočet") }, h("i", { style: `width:${Math.round(NS.progress * 100)}%` })));
    if (NS.results) card.appendChild(resultsBlock(root));
    return card;
  }

  async function loadFile(file, root) {
    NS.error = null;
    try {
      NS.file = file.name;
      NS.label = file.name.replace(/\.[^.]+$/, "");
      ingest(await readFile(file), root);
    } catch (e) {
      NS.error = tx(`The file could not be read (${e.message}). Save it as CSV and try again.`, `Soubor se nepodařilo načíst (${e.message}). Uložte jej jako CSV a zkuste to znovu.`);
      render(root);
    }
  }

  function ingest(table, root) {
    const { records, map, issues } = toRecords(table);
    NS.records = records; NS.map = map; NS.issues = issues; NS.results = null; NS.preds = null;
    if (!map.mrs90) NS.error = tx("No 90-day mRS column (mrs90) was found, so performance cannot be measured.", "Nebyl nalezen sloupec mRS za 90 dní (mrs90), výkon proto nelze změřit.");
    render(root);
  }

  function templateDetails() {
    const det = h("details", { class: "template" });
    det.appendChild(h("summary", { text: tx("Data format and template", "Formát dat a šablona") }));
    const header = KEYS.join(",");
    const copyBtn = h("button", { class: "btn", type: "button" }, tx("Copy column header", "Kopírovat záhlaví"));
    copyBtn.addEventListener("click", () => copyText(header, copyBtn));
    det.appendChild(h("div", { class: "tool-actions" }, copyBtn, downloadLink(`${header}\n`, "mtop_template.csv", "text/csv")));
    const rows = KEYS.map((k) => {
      const f = FEAT[k];
      if (k === "mrs90") return [k, tx("90-day mRS", "mRS za 90 dní"), "0–6", tx("required for validation", "nutné pro validaci")];
      if (k === "tici") return [k, A.L(f.label), "0, 1, 2a, 2b, 2c, 3", tx("2c is analysed as 3", "2c se hodnotí jako 3")];
      let vals;
      if (f.type === "categorical") vals = f.options.map((o) => o.value).join(", ");
      else if (f.type === "binary") vals = k === "sex_male" ? tx("1 = male, 0 = female", "1 = muž, 0 = žena") : k === "side_left" ? tx("1 = left, 0 = right", "1 = levá, 0 = pravá") : tx("1 = yes, 0 = no", "1 = ano, 0 = ne");
      else if (f.type === "ordinal") vals = f.options.map((o) => o.value).join(", ");
      else vals = `${f.min}–${f.max}${f.unit ? ` ${A.L(f.unit)}` : ""}`;
      return [k, A.L(f.label), vals, `${A.L(STAGE[f.stage].label)}${f.optional ? tx(" · may be empty", " · může chybět") : ""}`];
    });
    det.appendChild(h("div", { class: "table-scroll" }, h("table", { class: "data compact" },
      h("thead", null, h("tr", null, [tx("Column", "Sloupec"), tx("Meaning", "Význam"), tx("Values", "Hodnoty"), tx("Stage", "Fáze")].map((x) => h("th", { scope: "col", text: x })))),
      h("tbody", null, rows.map((r) => h("tr", null, h("th", { scope: "row", class: "mono", text: r[0] }), h("td", { text: r[1] }), h("td", { text: r[2] }), h("td", { text: r[3] })))))));
    det.appendChild(h("p", { class: "note", text: tx("Empty cells are treated as unknown. Common English and Czech spellings (ano/ne, M/F, M1/M2, ADAPT …) are recognised.", "Prázdné buňky se berou jako neznámé. Rozpoznávají se běžné anglické i české zápisy (ano/ne, M/Ž, M1/M2, ADAPT …).") }));
    return det;
  }

  function mappingSummary() {
    const found = KEYS.filter((k) => NS.map[k] !== undefined);
    const missing = KEYS.filter((k) => NS.map[k] === undefined);
    const box = h("div", { class: "mapping" });
    box.appendChild(h("p", null, h("b", { text: `${num(NS.records.length)} ${tx("rows", "řádků")}` }),
      ` · ${tx("recognised columns", "rozpoznané sloupce")}: ${found.length}/${KEYS.length}` +
      ` · ${tx("with 90-day mRS", "se mRS za 90 dní")}: ${num(NS.records.filter((x) => x.mrs90 !== null).length)}`));
    if (missing.length) box.appendChild(h("p", { class: "note", text: `${tx("Not found (treated as unknown)", "Nenalezeno (bráno jako neznámé)")}: ${missing.join(", ")}` }));
    const bad = Object.entries(NS.issues || {});
    if (bad.length) box.appendChild(h("p", { class: "note", text: `${tx("Unrecognised or out-of-range values set to unknown", "Nerozpoznané nebo neplatné hodnoty nastavené jako neznámé")}: ${bad.map(([k, n]) => `${k} ${n}`).join(", ")}` }));
    return box;
  }

  function resultsBlock(root) {
    const box = h("div", { class: "results" });
    const seg = h("div", { class: "mini-seg", role: "group", "aria-label": A.t().outcome },
      A.TARGETS.map((k) => h("button", { type: "button", "aria-pressed": String(S.netTarget === k), onclick: () => { S.netTarget = k; render(root); } }, k === "good_outcome" ? A.t().targetGood : A.t().targetDeath)));
    box.appendChild(h("div", { class: "controls" }, h("b", { text: tx("Results", "Výsledky") }), seg));
    const tg = S.netTarget;
    const lnSel = h("select", { class: "sel", id: "lv-learner" });
    lnSel.appendChild(h("option", { value: "", selected: S.netLearner ? null : true }, tx("Calculator's model for each stage", "Model kalkulačky pro každou fázi")));
    for (const k of ALL_LEARNERS) lnSel.appendChild(h("option", { value: k, selected: S.netLearner === k ? true : null }, A.mname(k)));
    lnSel.addEventListener("change", () => { S.netLearner = lnSel.value || null; render(root); });
    box.appendChild(h("div", { class: "controls" }, h("label", { for: "lv-learner" }, tx("Model", "Model")), lnSel));
    const fnoYrs = A.yearsOf("primary");
    const ref = (st) => A.V.splits.primary.stages[st].targets[tg].models[netLearner(st, tg)];
    const tbl = h("table", { class: "data" },
      h("thead", null, h("tr", null, [tx("Stage", "Fáze"), tx("Model", "Model"), "n", tx("Events", "Události"), "AUROC (95% CI)", "Brier", tx("Cal. intercept", "Kal. intercept"), tx("Cal. slope", "Kal. sklon"), `FNO ${fnoYrs}`].map((x) => h("th", { scope: "col", text: x })))),
      h("tbody", null, STAGES.map((st) => {
        const r = NS.results[st][tg][netLearner(st, tg)];
        const cells = r.insufficient
          ? [h("td", { text: num(r.n) }), h("td", { text: "–" }), h("td", { colspan: 4, text: tx("too few patients or events", "příliš málo pacientů nebo událostí") })]
          : [h("td", { text: num(r.n) }), h("td", { text: num(r.events) }),
            h("td", null, `${num(r.auroc, 3)} `, h("span", { class: "ci", text: `${num(r.auroc_ci[0], 3)}–${num(r.auroc_ci[1], 3)}` })),
            h("td", { text: num(r.brier, 3) }), h("td", { text: signed(r.citl) }), h("td", { text: num(r.slope, 2) })];
        return h("tr", null, h("th", { scope: "row", text: A.stageLabel(st) }), h("td", { text: A.mname(netLearner(st, tg)) }), ...cells, h("td", { text: ref(st) ? num(ref(st).auroc, 3) : "–" }));
      })));
    box.appendChild(h("div", { class: "table-scroll" }, tbl));
    // every learner at the selected stage
    const cmpRows = learnersOf(S.netStage, tg).map((nm) => {
      const r = NS.results[S.netStage][tg][nm];
      const f = A.V.splits.primary.stages[S.netStage].targets[tg].models[nm];
      return h("tr", { class: nm === A.chosenOf(S.netStage, tg) ? "chosen" : "" },
        h("th", { scope: "row" }, h("span", { class: "dot", style: `background:${A.mcolor(nm)}` }), A.mname(nm), nm === A.chosenOf(S.netStage, tg) ? h("span", { class: "tag", text: tx("calculator", "kalkulačka") }) : null),
        ...(r.insufficient ? [h("td", { colspan: 4, text: tx("too few patients or events", "příliš málo pacientů nebo událostí") })]
          : [h("td", null, `${num(r.auroc, 3)} `, h("span", { class: "ci", text: `${num(r.auroc_ci[0], 3)}–${num(r.auroc_ci[1], 3)}` })), h("td", { text: num(r.brier, 3) }), h("td", { text: signed(r.citl) }), h("td", { text: num(r.slope, 2) })]),
        h("td", { text: f ? num(f.auroc, 3) : "–" }));
    });
    box.appendChild(h("div", { class: "chart-sub", style: "margin-top:14px", text: tx(`Every published model at ${A.stageLabel(S.netStage).toLowerCase()}`, `Všechny zveřejněné modely ve fázi ${A.stageLabel(S.netStage).toLowerCase()}`) }));
    box.appendChild(h("div", { class: "table-scroll" }, h("table", { class: "data comp" },
      h("thead", null, h("tr", null, [tx("Model", "Model"), "AUROC (95% CI)", "Brier", tx("Cal. intercept", "Kal. intercept"), tx("Cal. slope", "Kal. sklon"), `FNO ${fnoYrs}`].map((x) => h("th", { scope: "col", text: x })))),
      h("tbody", null, cmpRows))));

    // calibration for the selected stage
    const stSel = h("select", { class: "sel", id: "lv-stage" });
    for (const st of STAGES) stSel.appendChild(h("option", { value: st, selected: st === S.netStage ? true : null }, A.stageLabel(st)));
    stSel.addEventListener("change", () => { S.netStage = stSel.value; render(root); });
    const r = NS.results[S.netStage][tg][netLearner(S.netStage, tg)];
    const calBox = h("div", { class: "card-inner" });
    calBox.appendChild(h("div", { class: "controls" }, h("label", { for: "lv-stage" }, tx("Calibration at", "Kalibrace ve fázi")), stSel));
    if (!r.insufficient && r.calibration.length) {
      const f = C.figure(calBox, { title: tx("Calibration on your data", "Kalibrace na vašich datech"), sub: tx("Groups of patients by predicted risk", "Skupiny pacientů podle predikovaného rizika"),
        table: { columns: [A.t().predicted, A.t().observed, "n"], rows: r.calibration.map((b) => [A.pct(b.pred, 1), A.pct(b.obs, 1), b.n]) },
        labels: { table: A.t().showTable, chart: A.t().showChart } });
      const mx = Math.min(1, Math.ceil(Math.max(...r.calibration.map((b) => Math.max(b.pred, b.obs))) * 10 + 0.5) / 10);
      C.lineChart(f.body, { height: 220, x: { type: "lin", domain: [0, mx], fmt: (v) => A.pct(v), title: A.t().predicted }, y: { domain: [0, mx], fmt: (v) => A.pct(v), title: A.t().observed }, diagonal: true,
        series: [{ label: A.t().observed, color: "var(--c1)", dots: true, points: r.calibration.map((b) => ({ x: b.pred, y: b.obs, n: b.n })) }],
        tipFmt: (v, p) => `${A.pct(v, 1)} · n ${p.n}`, tipHead: (xv) => `${A.t().predicted} ${A.pct(xv, 1)}`, aria: "local calibration" });
    } else calBox.appendChild(h("p", { class: "note", text: tx("Not enough patients or events at this stage.", "V této fázi není dost pacientů nebo událostí.") }));
    box.appendChild(calBox);

    // export
    const nameIn = h("input", { type: "text", id: "lv-site", class: "text-in", value: NS.label, placeholder: tx("Site name", "Název centra") });
    nameIn.addEventListener("input", () => { NS.label = nameIn.value; });
    const copyBtn = h("button", { class: "btn", type: "button" }, tx("Copy results file", "Kopírovat soubor výsledků"));
    copyBtn.addEventListener("click", () => copyText(exportJSON(), copyBtn));
    const dl = h("button", { class: "btn", type: "button" }, tx("Prepare download", "Připravit ke stažení"));
    dl.addEventListener("click", () => { dl.replaceWith(downloadLink(exportJSON(), `mtop_results_${(NS.label || "site").replace(/[^a-z0-9]+/gi, "_")}.json`, "application/json")); });
    const addPool = h("button", { class: "btn btn-primary", type: "button" }, tx("Add to pooled analysis below", "Přidat do souhrnné analýzy níže"));
    addPool.addEventListener("click", () => { NS.imports.push(JSON.parse(exportJSON())); render(root); document.getElementById("pool-card")?.scrollIntoView({ block: "start", behavior: "smooth" }); });
    box.appendChild(h("div", { class: "export" },
      h("label", { for: "lv-site", text: tx("Results file (aggregate metrics only):", "Soubor výsledků (jen souhrnné metriky):") }),
      h("div", { class: "tool-actions" }, nameIn, copyBtn, dl, addPool)));

    // predictions preview
    const det = h("details", { class: "template" });
    det.appendChild(h("summary", { text: tx("Predictions for each row (stay on this computer)", "Predikce pro každý řádek (zůstávají v tomto počítači)") }));
    const head = ["#"].concat(STAGES.map((st) => `${A.stageLabel(st)} · mRS 0–2 · ${A.mname(netLearner(st, "good_outcome"))}`)).concat(["mrs90"]);
    const lines = NS.records.map((x, i) => [i + 1].concat(STAGES.map((st) => (NS.preds[i][st] ? NS.preds[i][st].good_outcome[netLearner(st, "good_outcome")].toFixed(3) : ""))).concat([x.mrs90 === null ? "" : x.mrs90]));
    const cp = h("button", { class: "btn", type: "button" }, tx("Copy all as CSV", "Kopírovat vše jako CSV"));
    cp.addEventListener("click", () => copyText([head.join(",")].concat(lines.map((l) => l.join(","))).join("\n"), cp));
    det.appendChild(h("div", { class: "tool-actions" }, cp));
    det.appendChild(h("div", { class: "table-scroll" }, h("table", { class: "data compact" },
      h("thead", null, h("tr", null, head.map((x) => h("th", { scope: "col", text: x })))),
      h("tbody", null, lines.slice(0, 25).map((l) => h("tr", null, l.map((v, j) => (j === 0 ? h("th", { scope: "row", text: String(v) }) : h("td", { text: String(v) })))))))));
    if (lines.length > 25) det.appendChild(h("p", { class: "note", text: tx(`First 25 of ${num(lines.length)} rows shown.`, `Zobrazeno prvních 25 z ${num(lines.length)} řádků.`) }));
    box.appendChild(det);
    return box;
  }

  // ------------------------------------------------------------------ pooled analysis
  function poolCard(root) {
    const card = h("div", { class: "card card-pad block", id: "pool-card" });
    const st = S.netStage, tg = S.netTarget;
    const seg = h("div", { class: "mini-seg", role: "group", "aria-label": A.t().outcome },
      A.TARGETS.map((k) => h("button", { type: "button", "aria-pressed": String(tg === k), onclick: () => { S.netTarget = k; render(root); } }, k === "good_outcome" ? A.t().targetGood : A.t().targetDeath)));
    const stSel = h("select", { class: "sel", id: "pool-stage" });
    for (const s2 of STAGES) stSel.appendChild(h("option", { value: s2, selected: s2 === st ? true : null }, A.stageLabel(s2)));
    stSel.addEventListener("change", () => { S.netStage = stSel.value; render(root); });
    const fnoCb = h("input", { type: "checkbox", id: "pool-fno" });
    fnoCb.checked = NS.includeFno;
    fnoCb.addEventListener("change", () => { NS.includeFno = fnoCb.checked; render(root); });
    const fileIn = h("input", { type: "file", id: "pool-files", accept: ".json", multiple: true, class: "visually-hidden" });
    fileIn.addEventListener("change", async () => {
      for (const f of fileIn.files) {
        try {
          const j = JSON.parse(await f.text());
          if (/^mtop-validation\/[12]$/.test(j.format || "") && j.stages) NS.imports.push(j);
        } catch (e) { /* ignore unreadable files */ }
      }
      render(root);
    });
    card.appendChild(h("div", { class: "chart-head" }, h("div", null,
      h("div", { class: "chart-title", text: tx("Pooled performance across sites", "Souhrnný výkon napříč centry") }),
      h("div", { class: "chart-sub", text: tx("Random-effects meta-analysis of AUROC (logit scale) with 95% prediction interval", "Metaanalýza AUROC s náhodnými efekty (logitová škála) s 95% predikčním intervalem") }))));
    card.appendChild(h("div", { class: "controls" }, seg, h("label", { for: "pool-stage" }, A.t().stage), stSel,
      h("label", { for: "pool-fno" }, fnoCb, tx("FNO eras (internal–external)", "Období FNO (interně-externí)")),
      h("label", { class: "btn", for: "pool-files" }, tx("Add results files", "Přidat soubory výsledků")), fileIn));

    const rows = [];
    const nm = netLearner(st, tg);
    const AD = A.DATA.adaptive || {};
    const iecvAll = ((AD.iecv || {})[st] || {})[tg];
    const iecv = iecvAll ? (AD.learners ? iecvAll[nm] : iecvAll) : null;
    card.appendChild(h("p", { class: "note", style: "margin-top:-6px", text: tx(`Model: ${A.mname(nm)} (change it in the results above)`, `Model: ${A.mname(nm)} (lze změnit ve výsledcích výše)`) }));
    if (NS.includeFno && iecv) {
      for (const e of iecv.eras) if (!e.suppressed) rows.push({ label: `FNO ${e.label}`, sub: `n ${e.n}`, est: e.auroc, lo: e.ci[0], hi: e.ci[1], logit: e.logit, se: e.se_logit, color: "var(--c1)" });
    }
    NS.imports.forEach((j) => {
      const r = siteResult(j, st, tg, nm);
      if (r && !r.insufficient && r.se_logit) rows.push({ label: j.site, sub: `n ${r.n}`, est: r.auroc, lo: r.auroc_ci[0], hi: r.auroc_ci[1], logit: Math.log(r.auroc / (1 - r.auroc)), se: r.se_logit, color: "var(--c2)" });
    });
    if (rows.length < 2) {
      card.appendChild(h("p", { class: "note", text: tx("At least two sites or eras are needed for pooling. Include the FNO eras or add results files.", "Ke sloučení jsou potřeba alespoň dvě centra nebo období. Zahrňte období FNO nebo přidejte soubory výsledků.") }));
      return card;
    }
    const re = MX.randomEffects(rows.map((r) => r.logit), rows.map((r) => r.se));
    const ex = (v) => 1 / (1 + Math.exp(-v));
    rows.forEach((r, i) => { r.weight = re.weights[i]; r.display = `${num(r.est, 3)} (${num(r.lo, 2)}–${num(r.hi, 2)})`; });
    const pooled = { label: tx("Pooled", "Souhrn"), est: ex(re.mu), lo: ex(re.lo), hi: ex(re.hi), piLo: ex(re.piLo), piHi: ex(re.piHi), display: `${num(ex(re.mu), 3)} (${num(ex(re.lo), 2)}–${num(ex(re.hi), 2)})` };
    const allLo = Math.min(...rows.map((r) => r.lo), pooled.piLo), allHi = Math.max(...rows.map((r) => r.hi), pooled.piHi);
    const dom = [Math.max(0.4, Math.floor(allLo * 20) / 20), Math.min(1, Math.ceil(allHi * 20) / 20)];
    const f = C.figure(card, { title: "", table: { columns: [tx("Site", "Centrum"), "n", "AUROC (95% CI)", tx("Weight", "Váha")], rows: rows.map((r) => [r.label, r.sub.replace("n ", ""), r.display, A.pct(r.weight, 1)]).concat([[pooled.label, "", pooled.display, "100%"]]) },
      labels: { table: A.t().showTable, chart: A.t().showChart } });
    C.forest(f.body, { rows, pooled, domain: dom, fmt: (v) => num(v, 2), aria: "forest plot" });
    card.appendChild(h("p", { class: "note", text: tx(
      `I² ${num(re.i2 * 100)}% · τ² ${num(re.tau2, 3)} · 95% prediction interval ${num(pooled.piLo, 2)}–${num(pooled.piHi, 2)}: the range of AUROC to expect at a new site like these. FNO eras come from leave-one-era-out validation; they stand in for partner centres until real results arrive.`,
      `I² ${num(re.i2 * 100)} % · τ² ${num(re.tau2, 3)} · 95% predikční interval ${num(pooled.piLo, 2)}–${num(pooled.piHi, 2)}: rozsah AUROC, který lze čekat v novém podobném centru. Období FNO pocházejí z validace s vynecháním jednoho období; zastupují partnerská centra, dokud nepřijdou skutečné výsledky.`) }));
    if (NS.imports.length) {
      const clr = h("button", { class: "btn btn-ghost", type: "button" }, tx("Remove added sites", "Odebrat přidaná centra"));
      clr.addEventListener("click", () => { NS.imports = []; render(root); });
      card.appendChild(clr);
    }
    return card;
  }

  function nextStepCard() {
    const card = h("div", { class: "card card-pad block next-step" });
    card.appendChild(h("div", null,
      h("div", { class: "chart-title", text: tx("Next step: learn together", "Další krok: učit se společně") }),
      h("p", { class: "note", text: tx(
        "Validation shows whether MTOP transfers to your centre. The federated network goes further: centres train a shared model together, still without moving patient data.",
        "Validace ukáže, zda se MTOP přenáší do vašeho centra. Federovaná síť jde dál: centra společně trénují sdílený model, opět bez přesunu dat pacientů.") })));
    const b = h("button", { class: "btn btn-primary", type: "button" }, tx("See the federated network", "Ukázka federované sítě"));
    b.addEventListener("click", () => A.goTo("federated"));
    card.appendChild(b);
    if (A.SITE.contact) card.appendChild(h("p", { class: "note", text: tx(`Centres interested in validating MTOP or joining the network: ${A.SITE.contact}`, `Centra se zájmem o validaci MTOP nebo zapojení do sítě: ${A.SITE.contact}`) }));
    return card;
  }

  A.registerView("network", () => tx("Multicentre", "Multicentrum"), render);
})();
