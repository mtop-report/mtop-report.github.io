/* MTOP application: staged calculator, registry explorer, validation, about. */
(function () {
  "use strict";
  const DATA = JSON.parse(document.getElementById("mtop-data").textContent);
  const B = DATA.bundle;
  const V = DATA.validation;
  const E = DATA.explorer;
  const SITE = DATA.site || {};
  const X = DATA.extras || {};
  const M = window.MTOPModel;
  const XE = window.MTOPExtra;
  const C = window.MTOPCharts;
  const I18N = window.MTOP_I18N;
  const { h } = C;

  // ------------------------------------------------------------------ state
  const store = {
    get(k, d) { try { const v = localStorage.getItem(k); return v === null ? d : v; } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* storage unavailable */ } },
  };
  const navLang = (navigator.language || "").toLowerCase();
  let lang = store.get("mtop.lang", navLang.startsWith("cs") || navLang.startsWith("sk") ? "cs" : "en");
  if (!I18N[lang]) lang = "en";
  const t = () => I18N[lang];
  const L = (o) => (o && typeof o === "object" ? o[lang] || o.en : o);
  const tx = (en, cs) => (lang === "cs" ? cs : en);

  const STAGES = B.stages.map((s) => s.id);
  const STAGE = Object.fromEntries(B.stages.map((s) => [s.id, s]));
  const FEAT = Object.fromEntries(B.features.map((f) => [f.key, f]));
  const POSTERIOR = new Set(["basilar", "other_posterior"]);
  const TARGETS = ["good_outcome", "death90"];
  const REF = B.reference || "Logistic regression (splines)";
  // the temporal splits: primary (as in the 2026 analysis) and sensitivity
  const SPLITS = (V.meta && V.meta.splits) || { primary: 2023 };
  const LAST_YEAR = (E.period || [2005, 2025])[1];
  const yearsOf = (split) => `${SPLITS[split || "primary"]}–${LAST_YEAR}`;
  const devYears = (split) => `${(E.period || [2005])[0]}–${SPLITS[split || "primary"] - 1}`;
  const RECENT = `${B.recent_era_from_year}–${LAST_YEAR}`;

  const EXAMPLE = {
    age: 72, sex_male: 1, mrs_pre: 0, nihss_adm: 16, aspects: 9, occlusion: "M1", side_left: 1,
    ivt: 1, ivt_external: 0, onset_to_ivt: 110, transfer: 0, onset_to_puncture: 185, onset_hour: 9, weekend: 0,
    glucose: 7.2, sbp: 155, dbp: 85, bmi: 27,
    htn: 1, dm: 0, hld: 1, arrhythmia: 1, stroke_prior: 0, tia_prior: 0, smoking: 0, alcohol: 0, statin: 1,
    heart3: "none", antithrombotic: "none",
    tici: 4, technique: "aspiration", stenting: 0, angioplasty: 0, ia_lysis: 0, procedure_min: 40, onset_to_recan: 225, anesthesia: "general",
    nihss_24h: 7, sich: 0, ich_other: 0, nihss_7d: 4, edema: 0, decompression: 0,
  };
  const S = {
    view: "calc",
    current: "h24",
    viewed: "h24",
    inputs: Object.assign({}, EXAMPLE),
    isExample: true,
    explainTarget: "good_outcome",
    whatif: {},
    calcModel: null,          // null: the model chosen by the pre-specified rule for each stage and outcome
    valTarget: "good_outcome",
    valStage: "admission",
    valSplit: "primary",
    valModel: null,           // model shown in the per-model cards (default: the chosen one)
    regView: "outcomes",
    regDim: "era",
    distView: "bar",
  };
  const idx = (s) => STAGES.indexOf(s);

  // ------------------------------------------------------------------ formatting
  const locale = () => (lang === "cs" ? "cs-CZ" : "en-GB");
  const num = (v, d = 0) => (v === null || v === undefined || Number.isNaN(v) ? "–" : new Intl.NumberFormat(locale(), { minimumFractionDigits: d, maximumFractionDigits: d }).format(v));
  const pct = (p, d = 0) => (p === null || p === undefined ? "–" : lang === "cs" ? `${num(p * 100, d)} %` : `${num(p * 100, d)}%`);
  const pct100 = (v, d = 0) => (v === null || v === undefined ? "–" : lang === "cs" ? `${num(v, d)} %` : `${num(v, d)}%`);
  const stageLabel = (s) => L(STAGE[s].label);
  const clock = (h) => `${String(Math.floor(h)).padStart(2, "0")}:${h % 1 ? "30" : "00"}`;

  function displayValue(key, v) {
    const f = FEAT[key];
    if (!f) return "";
    const x = S.inputs;
    if (f.type === "derived") {
      const val = (k) => (x[k] === null || x[k] === undefined ? NaN : Number(x[k]));
      const d = M.derivedValue(key, (k) => (x.ivt === 0 && k === "onset_to_ivt" ? NaN : val(k)));
      if (Number.isNaN(d)) return t().unknown.toLowerCase();
      if (key === "nihss_change_24h") return d > 0 ? `+${d}` : d < 0 ? `−${Math.abs(d)}` : "0";
      if (key === "ivt_to_puncture") return t().hm(d);
      if (key === "pulse_pressure") return `${num(d)} mmHg`;
      return d ? tx("yes", "ano") : tx("no", "ne");
    }
    if (f.requires_ivt && x.ivt === 0) return tx("no IVT", "bez IVT");
    if (v === null || v === undefined || v === "") return t().unknown.toLowerCase();
    if (["onset_to_puncture", "onset_to_ivt", "onset_to_recan"].includes(key)) return t().hm(v);
    if (key === "onset_hour") return clock(v);
    if (f.options) {
      const o = f.options.find((o2) => String(o2.value) === String(v));
      if (o) return L(o.label).replace(/ · .*/, "");
    }
    const unit = f.unit ? ` ${L(f.unit)}` : "";
    return `${num(v, f.step && f.step < 1 ? 1 : 0)}${unit}`;
  }

  // ------------------------------------------------------------------ prediction
  function modelInputs() {
    const x = Object.assign({}, S.inputs);
    if (POSTERIOR.has(x.occlusion)) x.side_left = 0;
    if (x.ivt === 0) { x.ivt_external = 0; x.onset_to_ivt = null; }
    return x;
  }
  const entryOf = (stage, target) => B.models[stage][target];
  const chosenOf = (stage, target) => entryOf(stage, target).chosen;
  const learnerName = (stage, target) => {
    const e = entryOf(stage, target);
    return S.calcModel && e.learners[S.calcModel] ? S.calcModel : e.chosen;
  };
  // Unknown optional inputs are averaged over registry values (quantiles / category mix),
  // instead of using the models' missing-value handling, which encodes *who* had missing data.
  // Inputs that do not apply (onset-to-IVT without IVT) stay missing, as in the registry.
  const IMPUTE = B.impute || {};
  const PERMS = {};
  function perm(n, key) {
    let s = 7;
    for (const ch of key) s = (s * 31 + ch.charCodeAt(0)) % 2147483647;
    const a = Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i > 0; i--) {
      s = (s * 16807) % 2147483647;
      const j = s % (i + 1);
      [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
  }
  function unknownKeys(entry, inputs) {
    const used = new Set(entry.column_feature);
    return Object.keys(IMPUTE).filter((k) => used.has(k) && (inputs[k] === null || inputs[k] === undefined)
      && !(FEAT[k] && FEAT[k].requires_ivt && inputs.ivt === 0));
  }
  function samples(entry, inputs) {
    const unk = unknownKeys(entry, inputs);
    if (!unk.length) return [inputs];
    const n = Math.max(...unk.map((k) => IMPUTE[k].values.length));
    const out = [];
    for (let i = 0; i < n; i++) {
      const x = Object.assign({}, inputs);
      for (const k of unk) {
        const vals = IMPUTE[k].values;
        const p = PERMS[k] || (PERMS[k] = perm(vals.length, k));
        x[k] = vals[p[i % vals.length]];
      }
      out.push(x);
    }
    return out;
  }
  function predictStage(stage, inputs, target, name) {
    const entry = entryOf(stage, target);
    const nm = name || learnerName(stage, target);
    const spec = entry.learners[nm];
    const x0 = inputs || modelInputs();
    const xs = samples(entry, x0);
    let p = 0, lg = 0;
    const rows = xs.map((x) => M.buildRow(entry.columns, B.features, x));
    for (const row of rows) { const r = M.predict(spec, row, { recent: true }); p += r.p; lg += r.logit; }
    return { entry, spec, name: nm, chosen: nm === entry.chosen, rows, p: p / rows.length, logit: lg / rows.length, unknown: unknownKeys(entry, x0) };
  }
  function explainStage(pr) {
    const acc = {};
    let base = 0;
    for (const row of pr.rows) {
      const e = M.explain(pr.spec, row, pr.entry.column_feature);
      base = e.base;
      for (const k in e.byFeature) acc[k] = (acc[k] || 0) + e.byFeature[k] / pr.rows.length;
    }
    return { base, byFeature: acc };
  }
  function threeBand(pg, pd) {
    let g = pg, d = pd;
    if (g + d > 0.98) { const k = 0.98 / (g + d); g *= k; d *= k; }
    return [g, 1 - g - d, d];
  }
  function realityBin(stage, p, name) {
    const dep = ((V.deployed || {})[stage] || {}).good_outcome;
    const lr = dep && dep.learners[name || dep.chosen];
    const bins = (lr && lr.reality_bins) || [];
    if (!bins.length) return null;
    let best = bins[0];
    for (const b of bins) if (p >= b.lo - 1e-9) best = b;
    return best;
  }

  // ------------------------------------------------------------------ DOM refs
  const $ = (sel, el) => (el || document).querySelector(sel);
  const app = $("#app");

  const VIEW_ORDER = ["calc", "registry", "validation", "adaptive", "network", "reporting", "roadmap", "about"];
  const VIEWS = {
    calc: { label: () => t().tabCalc, render: () => renderCalc() },
    registry: { label: () => t().tabRegistry, render: () => renderRegistry() },
    validation: { label: () => t().tabValidation, render: () => renderValidation() },
    about: { label: () => t().tabAbout, render: () => renderAbout() },
  };
  function registerView(id, label, renderFn) { VIEWS[id] = { label, render: renderFn }; }
  const viewIds = () => VIEW_ORDER.filter((id) => VIEWS[id]);
  // addresses of the two parts of the Multicentre tab
  const ALIASES = { federated: ["network", { netMode: "federated" }], validate: ["network", { netMode: "validate" }] };
  function goTo(id) {
    if (ALIASES[id]) Object.assign(S, ALIASES[id][1]);
    S.view = ALIASES[id] ? ALIASES[id][0] : id;
    try { history.replaceState(null, "", `#${id}`); } catch (e) { /* sandboxed frame */ }
    render();
    window.scrollTo(0, 0);
  }

  function render() {
    document.documentElement.lang = lang;
    app.replaceChildren(header(), mainEl(), footer());
    fitHeader();
    renderView();
  }

  // One header row when brand, tabs and language switch fit; otherwise drop the subtitle, then move the
  // tabs to their own scrolling row. Measured, because tab labels differ in length between languages.
  function fitHeader() {
    const top = $(".top");
    if (!top) return;
    const inner = top.firstElementChild;
    top.classList.remove("no-sub", "compact");
    const wrapped = () => inner.offsetHeight > 66;
    if (wrapped()) top.classList.add("no-sub");
    if (wrapped()) top.classList.add("compact");
  }
  let fitQueued = false;
  window.addEventListener("resize", () => {
    if (fitQueued) return;
    fitQueued = true;
    requestAnimationFrame(() => { fitQueued = false; fitHeader(); });
  });
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(fitHeader);

  function header() {
    const nav = h("nav", { class: "tabs", role: "tablist", "aria-label": "MTOP" },
      viewIds().map((id) => h("button", {
        class: "tab", role: "tab", type: "button", id: `tab-${id}`, "aria-selected": String(S.view === id), "aria-controls": `view-${id}`,
        onclick: () => goTo(id),
      }, VIEWS[id].label())));
    const langSw = h("div", { class: "lang", role: "group", "aria-label": "Language" },
      [["en", "EN"], ["cs", "CZ"]].map(([code, lbl]) => h("button", {
        type: "button", "aria-pressed": String(lang === code), onclick: () => { lang = code; store.set("mtop.lang", code); render(); },
      }, lbl)));
    const brand = h("a", { class: "brand", href: "#calc", "aria-label": tx("MTOP, go to the calculator", "MTOP, přejít na kalkulačku"),
      onclick: (e) => { e.preventDefault(); goTo("calc"); } },
      h("span", { class: "brand-logo", "aria-hidden": "true", html: DATA.logo_svg || "MTOP" }),
      h("span", { class: "brand-sub", text: t().brandSub }));
    // on narrow screens the tab row scrolls: keep the active tab in view and fade the edge only while more follows
    const edge = () => nav.classList.toggle("at-end", nav.scrollLeft + nav.clientWidth >= nav.scrollWidth - 4);
    nav.addEventListener("scroll", edge, { passive: true });
    requestAnimationFrame(() => {
      const a = nav.querySelector('[aria-selected="true"]');
      if (a && nav.scrollWidth > nav.clientWidth) nav.scrollLeft = Math.max(0, a.offsetLeft - (nav.clientWidth - a.offsetWidth) / 2);
      edge();
    });
    return h("header", { class: "top" }, h("div", { class: "top-inner" }, brand, nav, langSw));
  }

  function mainEl() {
    return h("main", { class: "wrap", id: "main" },
      viewIds().map((id) => h("section", { class: "view", id: `view-${id}`, role: "tabpanel", "aria-labelledby": `tab-${id}`, hidden: S.view !== id })));
  }

  function footer() {
    const left = h("span", { class: "foot-brand" },
      DATA.logo_svg ? h("span", { "aria-hidden": "true", html: DATA.logo_svg }) : null,
      h("span", { text: `${t().footer} · v${B.version} · ${B.generated}` }));
    const right = SITE.repo_url ? h("a", { href: SITE.repo_url, target: "_blank", rel: "noopener", text: tx("Code on GitHub", "Kód na GitHubu") }) : null;
    return h("footer", { class: "foot" }, h("div", { class: "wrap" }, left, right));
  }

  function renderView() {
    const v = VIEWS[S.view] || VIEWS.calc;
    v.render($(`#view-${S.view}`));
  }

  // ================================================================== calculator
  let readoutEl = null;
  let stickyEl = null;

  function renderCalc() {
    const root = $("#view-calc");
    root.replaceChildren();
    root.appendChild(h("div", { class: "view-head calc-head" }, h("div", null,
      h("h2", { text: t().calcTitle }),
      h("p", { text: t().calcLede }),
      h("div", { class: "notice", role: "note" }, h("span", { html: t().notice })))));
    const left = h("div", { class: "inputs sheet" });
    const ex = h("div", { class: "example-bar" },
      S.isExample ? h("span", { class: "pill", text: t().example }) : null,
      h("span", { text: t().exampleNote }),
      h("button", { class: "btn", type: "button", onclick: () => { S.inputs = Object.assign({}, EXAMPLE); S.isExample = true; S.current = "h24"; S.viewed = "h24"; renderCalc(); } }, t().resetExample));
    left.appendChild(ex);
    for (const st of STAGES) left.appendChild(stageCard(st));
    readoutEl = h("div", { class: "readout", "aria-live": "polite" });
    root.appendChild(h("div", { class: "calc" }, left, readoutEl));
    stickyEl = h("div", { class: "stickybar" }, h("div", { class: "inner" }));
    root.appendChild(stickyEl);
    updateReadout();
  }

  // admission inputs are shown in groups; later stages list their inputs in registry order
  const GROUPS = [
    ["patient", ["Patient", "Pacient"]],
    ["stroke", ["The stroke", "CMP"]],
    ["pathway", ["Thrombolysis, route and timing", "Trombolýza, cesta a časy"]],
    ["vitals", ["On admission", "Při přijetí"]],
    ["history", ["History and medication", "Anamnéza a medikace"]],
  ];
  const HISTORY = B.features.filter((f) => f.group === "history" && f.type === "binary").map((f) => f.key);

  function stageCard(st) {
    const i = idx(st);
    const locked = i > idx(S.current);
    const feats = B.features.filter((f) => f.stage === st && f.type !== "derived");
    const card = h("section", { class: `stage${locked ? " locked" : ""}`, id: `stage-${st}`, "aria-labelledby": `stage-${st}-h` });
    const head = h("div", { class: "stage-head" },
      h("span", { class: "stage-num", "aria-hidden": "true", text: String(i + 1) }),
      h("div", { class: "grow" }, h("h3", { id: `stage-${st}-h`, text: L(STAGE[st].title) }), h("div", { class: "when", text: t().stageWhen[st] })));
    if (!locked && i > 0 && st === S.current) {
      head.appendChild(h("button", { class: "btn btn-ghost", type: "button", onclick: () => setCurrent(STAGES[i - 1]) }, lang === "cs" ? "Odebrat" : "Remove"));
    }
    card.appendChild(head);
    const body = h("div", { class: "stage-body" });
    if (st === "admission") {
      for (const [g, lbl] of GROUPS) {
        body.appendChild(h("div", { class: "field-group wide" }, h("span", { text: tx(lbl[0], lbl[1]) })));
        if (g === "history") { body.appendChild(historyField()); }
        for (const f of feats.filter((f2) => f2.group === g && !HISTORY.includes(f2.key))) body.appendChild(field(f));
      }
    } else {
      for (const f of feats) body.appendChild(field(f));
    }
    body.appendChild(previewBlock(st));
    card.appendChild(body);
    card.appendChild(h("div", { class: "stage-locked-note" },
      h("span", { text: t().lockedNote }),
      h("button", { class: "btn btn-primary", type: "button", onclick: () => setCurrent(st) }, `${t().addStage}: ${stageLabel(st)}`)));
    return card;
  }

  const PREVIEW = {
    admission: [
      { label: ["CT perfusion core volume", "Objem jádra infarktu (CT perfuze)"], unit: "mL", status: "candidate" },
      { label: ["CTA collateral score (0–3)", "Skóre kolaterál na CTA (0–3)"], unit: "", status: "candidate" },
    ],
    procedure: [
      { label: ["Number of device passes", "Počet průchodů"], unit: "", status: "candidate" },
      { label: ["eTICI 2c / 3 distinction", "Rozlišení eTICI 2c / 3"], unit: "", status: "candidate" },
    ],
    h24: [
      { label: ["Serum GFAP", "Sérový GFAP"], unit: "pg/mL", status: "collecting" },
      { label: ["Serum neurofilament light (NfL)", "Sérový neurofilament (NfL)"], unit: "pg/mL", status: "candidate" },
    ],
    d7: [
      { label: ["Follow-up infarct volume", "Kontrolní objem infarktu"], unit: "mL", status: "candidate" },
    ],
  };
  const STATUS = {
    live: () => ({ cls: "st-live", text: tx("Live", "Funguje") }),
    beta: () => ({ cls: "st-beta", text: tx("New · beta", "Nové · beta") }),
    progress: () => ({ cls: "st-progress", text: tx("In progress", "Probíhá") }),
    collecting: () => ({ cls: "st-progress", text: tx("Being collected", "Sbírá se") }),
    planned: () => ({ cls: "st-planned", text: tx("Planned", "Plánováno") }),
    candidate: () => ({ cls: "st-planned", text: tx("Candidate", "Kandidát") }),
    proposed: () => ({ cls: "st-planned", text: tx("Proposed", "Navrženo") }),
  };

  function statusChip(kind) {
    const st = STATUS[kind]();
    return h("span", { class: `status ${st.cls}` }, h("i", { "aria-hidden": "true" }), st.text);
  }

  function previewBlock(st) {
    const items = PREVIEW[st] || [];
    const det = h("details", { class: "preview field wide" });
    det.appendChild(h("summary", null, h("span", { text: tx("Preview: predictors for future versions", "Náhled: prediktory budoucích verzí") }), h("span", { class: "preview-note", text: tx("not used by the model", "model je nepoužívá") })));
    const grid = h("div", { class: "preview-grid" });
    items.forEach((it, i) => {
      const id = `pv-${st}-${i}`;
      grid.appendChild(h("div", { class: "preview-item" },
        h("label", { for: id, text: tx(it.label[0], it.label[1]) }),
        h("div", { class: "preview-row" }, h("input", { id, type: "text", disabled: true, placeholder: it.unit || "–" }), statusChip(it.status))));
    });
    det.appendChild(grid);
    det.appendChild(h("p", { class: "help", text: st === "h24"
      ? tx("GFAP is being collected at FNO. It will enter the model once enough patients with known 90-day outcome are available, after its added value has been tested.", "GFAP se ve FNO sbírá. Do modelu vstoupí, až bude dost pacientů se známým výsledkem za 90 dní a jeho přínos bude ověřen.")
      : tx("Not recorded in the registry yet. Shown to illustrate how later versions could use it.", "Registr je zatím neeviduje. Zobrazeno pro ilustraci, jak by je mohly využít další verze.") }));
    return det;
  }

  function setCurrent(st) {
    S.current = st;
    S.viewed = st;
    renderCalc();
    const el = $(`#stage-${st}`);
    if (el && idx(st) > 0) el.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }

  function setInput(key, v) {
    S.inputs[key] = v;
    S.isExample = false;
    const pill = $(".example-bar .pill");
    if (pill) pill.remove();
    updateReadout();
  }

  const WIDE = new Set(["occlusion", "onset_to_puncture", "technique", "antithrombotic", "heart3"]);
  const SHOW_VALUE = new Set(["nihss_adm", "nihss_24h", "nihss_7d", "onset_to_puncture", "onset_to_ivt", "onset_to_recan", "onset_hour"]);
  const TIMES = new Set(["onset_to_puncture", "onset_to_ivt", "onset_to_recan"]);

  function field(f) {
    const wrap = h("div", { class: `field${WIDE.has(f.key) ? " wide" : ""}`, id: `field-${f.key}` });
    const lblText = L(f.label);
    const valSpan = h("span", { class: "val" });
    const lab = h("div", { class: "field-label" }, h("span", { class: "lbl", id: `lbl-${f.key}`, text: lblText }), valSpan);
    wrap.appendChild(lab);
    const refresh = () => {
      const v = S.inputs[f.key];
      if (f.type === "numeric") {
        valSpan.replaceChildren();
        if (v === null || !SHOW_VALUE.has(f.key)) return;
        if (TIMES.has(f.key)) valSpan.textContent = t().hm(v);
        else if (f.key === "onset_hour") valSpan.textContent = clock(v);
        else valSpan.append(h("small", { text: t().nihssBand(v) }));
      }
    };
    if (f.type === "numeric") {
      const optional = !!f.optional;
      const range = h("input", { type: "range", id: `in-${f.key}`, min: f.min, max: f.max, step: f.step, "aria-labelledby": `lbl-${f.key}` });
      const box = h("input", { type: "number", id: `num-${f.key}`, min: f.min, max: f.max, step: f.step, "aria-labelledby": `lbl-${f.key}`, inputmode: f.step < 1 ? "decimal" : "numeric" });
      const unk = optional ? h("input", { type: "checkbox", id: `unk-${f.key}` }) : null;
      const sync = () => {
        const v = S.inputs[f.key];
        range.value = v === null ? f.default : v;
        box.value = v === null ? "" : v;
        range.disabled = v === null; box.disabled = v === null;
        if (unk) unk.checked = v === null;
        fill();
        refresh();
      };
      const clamp = (v) => Math.min(f.max, Math.max(f.min, v));
      const fill = () => range.style.setProperty("--fill", `${((Number(range.value) - f.min) / ((f.max - f.min) || 1)) * 100}%`);
      range.addEventListener("input", () => { setInput(f.key, Number(range.value)); box.value = range.value; fill(); refresh(); });
      box.addEventListener("change", () => {
        const v = Number(String(box.value).replace(",", "."));
        if (box.value === "" || Number.isNaN(v)) { sync(); return; }
        setInput(f.key, clamp(v)); sync();
      });
      if (unk) unk.addEventListener("change", () => { setInput(f.key, unk.checked ? null : (S.inputs[f.key] ?? f.default)); if (!unk.checked && S.inputs[f.key] === null) S.inputs[f.key] = f.default; sync(); updateReadout(); });
      if (unk) lab.insertBefore(h("label", { class: "unk", for: `unk-${f.key}` }, unk, t().unknown), valSpan.nextSibling);
      const row = h("div", { class: "range-row" }, range, box, f.unit ? h("span", { class: "unit", text: L(f.unit) }) : null);
      wrap.appendChild(row);
      sync();
    } else {
      const opts = f.options.slice();
      const isCat = f.type === "categorical";
      const seg = h("div", { class: `seg${isCat && opts.length > 3 ? " cols-2" : ""}${f.key === "technique" ? " cols-1" : ""}${f.key === "mrs_pre" || f.key === "tici" ? " nums" : ""}`, role: "group", "aria-labelledby": `lbl-${f.key}` });
      const buttons = [];
      const mk = (val, label, extraClass) => {
        const b = h("button", { type: "button", class: extraClass || "", "data-v": val === null ? "" : String(val) }, label);
        b.addEventListener("click", () => {
          const v = val === null ? null : isCat ? String(val) : Number(val);
          const before = S.inputs[f.key];
          // ASPECTS scores the MCA territory: for posterior-circulation strokes the registry
          // records 10 in most patients, so switching to a posterior occlusion sets it to 10.
          if (f.key === "occlusion" && POSTERIOR.has(v) && !POSTERIOR.has(before) && S.inputs.aspects !== null && S.inputs.aspects !== undefined) S.inputs.aspects = 10;
          // a diagnostic angiography has no stent, angioplasty or intra-arterial lysis
          if (f.key === "technique" && v === "diagnostic") { S.inputs.stenting = 0; S.inputs.angioplasty = 0; S.inputs.ia_lysis = 0; }
          setInput(f.key, v);
          paint();
          if (f.key === "occlusion" || f.key === "ivt" || f.key === "technique") { renderCalcKeepScroll(); }
        });
        buttons.push(b);
        seg.appendChild(b);
      };
      for (const o of opts) {
        const label = f.key === "mrs_pre" ? String(o.value) : L(o.label);
        mk(o.value, label);
      }
      if (f.optional) mk(null, t().unknown, "unknown");
      const paint = () => {
        const v = S.inputs[f.key];
        for (const b of buttons) b.setAttribute("aria-pressed", String(b.dataset.v === (v === null || v === undefined ? "" : String(v))));
        if (f.key === "mrs_pre") {
          const o = opts.find((x) => x.value === v);
          valSpan.textContent = o ? L(o.label).split(" · ")[1] || "" : "";
        }
      };
      wrap.appendChild(seg);
      paint();
      if ((f.key === "side_left" && POSTERIOR.has(S.inputs.occlusion)) || (f.requires_ivt && S.inputs.ivt === 0)
        || (["stenting", "angioplasty", "ia_lysis"].includes(f.key) && S.inputs.technique === "diagnostic")) {
        wrap.classList.add("disabled");
        buttons.forEach((b) => (b.disabled = true));
      }
    }
    if (f.type === "numeric" && f.requires_ivt && S.inputs.ivt === 0) {
      wrap.classList.add("disabled");
      wrap.querySelectorAll("input").forEach((el) => (el.disabled = true));
      wrap.appendChild(h("div", { class: "help", text: tx("Not applicable without IV thrombolysis.", "Bez IV trombolýzy se neuplatní.") }));
    }
    if (f.key === "aspects" && POSTERIOR.has(S.inputs.occlusion)) {
      wrap.appendChild(h("div", { class: "help", text: tx(
        "Posterior-circulation stroke: ASPECTS scores the middle cerebral artery territory. The registry records 10 for most posterior strokes and has no pc-ASPECTS, so 10 is set here. Change it only if the CT shows changes in the MCA territory.",
        "CMP v zadním povodí: ASPECTS hodnotí povodí arteria cerebri media. Registr u většiny CMP v zadním povodí eviduje 10 a pc-ASPECTS neobsahuje, proto je zde nastaveno 10. Změňte jen tehdy, ukazuje-li CT změny v povodí ACM.") }));
    } else if (f.help) wrap.appendChild(h("div", { class: "help", text: L(f.help) }));
    return wrap;
  }

  function renderCalcKeepScroll() {
    const y = window.scrollY;
    renderCalc();
    window.scrollTo(0, y);
  }

  function historyField() {
    const wrap = h("div", { class: "field wide", id: "field-history" });
    wrap.appendChild(h("div", { class: "field-label" }, h("span", { class: "lbl", id: "lbl-history", text: tx("Conditions and habits", "Onemocnění a návyky") }), h("span", { class: "val" })));
    const box = h("div", { class: "checks", role: "group", "aria-labelledby": "lbl-history" });
    for (const key of HISTORY) {
      const f = FEAT[key];
      const cb = h("input", { type: "checkbox", id: `cb-${key}` });
      cb.checked = S.inputs[key] === 1;
      cb.addEventListener("change", () => setInput(key, cb.checked ? 1 : 0));
      box.appendChild(h("label", { for: `cb-${key}` }, cb, L(f.label)));
    }
    wrap.appendChild(box);
    wrap.appendChild(h("div", { class: "help", text: t().historyHelp }));
    return wrap;
  }

  // ------------------------------------------------------------------ readout
  function updateReadout() {
    if (!readoutEl) return;
    const x = modelInputs();
    const preds = {};
    for (const st of STAGES) {
      if (idx(st) > idx(S.current)) break;
      preds[st] = { good: predictStage(st, x, "good_outcome"), death: predictStage(st, x, "death90") };
    }
    if (idx(S.viewed) > idx(S.current)) S.viewed = S.current;
    const vs = S.viewed;
    const pv = preds[vs];
    readoutEl.replaceChildren(...[miniCard(vs, pv), headlineCard(vs, pv), distributionCard(vs, pv), trajectoryCard(preds), explainCard(vs, pv), bedsideCard(vs, x), othersCard(vs, x, pv), whatifCard(vs, x)].filter(Boolean));
    readoutEl.insertBefore(modelChooser(vs, pv), readoutEl.children[1] || null);
    if (stickyEl) $(".inner", stickyEl).replaceChildren(
      h("span", { html: t().sticky(pct(pv.good.p), pct(pv.death.p)) }),
      h("span", { class: "delta", text: stageLabel(vs) }));
    observeHeadline();
  }

  // Once the full prognosis card has scrolled away, a slim copy stays at the top of the column
  // (desktop) or at the bottom of the screen (phone), so the panels below remain readable.
  function miniCard(st, pv) {
    const g = pv.good.p, d = pv.death.p;
    const unc = XE.uncertainty(((X.uncertainty || {})[st] || {}).good_outcome, g);
    const P = (v) => `${(Math.max(0, Math.min(1, v)) * 100).toFixed(2)}%`;
    const scale = h("div", { class: "pm-scale" }, h("i", { class: "pm-rail" }),
      unc ? h("i", { class: "pm-band", style: `left:${P(unc[0])};width:${P(unc[1] - unc[0])}` }) : null,
      h("i", { class: "pm-mark", style: `left:${P(g)}` }));
    return h("div", { class: "card prog-mini", "aria-hidden": "true" },
      h("div", { class: "pm-main" }, h("span", { class: "pm-stage", text: stageLabel(st) }),
        h("span", { class: "pm-g" }, h("b", { text: num(g * 100, 0) }), h("small", { text: lang === "cs" ? " %" : "%" }),
          h("span", { text: tx(" mRS 0–2", " mRS 0–2") }))),
      scale,
      h("div", { class: "pm-d" }, h("b", { text: pct(d) }), h("span", { text: tx("death", "úmrtí") })));
  }

  let headIO = null;
  function observeHeadline() {
    if (typeof IntersectionObserver === "undefined" || !readoutEl) return;
    if (headIO) headIO.disconnect();
    const head = $(".headline", readoutEl);
    if (!head) return;
    const top = (parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--head-h")) || 60) + 8;
    headIO = new IntersectionObserver((ents) => {
      for (const e of ents) {
        const gone = S.view === "calc" && e.boundingClientRect.top < top && e.intersectionRatio < 0.35;
        readoutEl.classList.toggle("mini-on", gone);
        if (stickyEl) stickyEl.classList.toggle("show", gone);
      }
    }, { rootMargin: `-${top}px 0px 0px 0px`, threshold: [0, 0.15, 0.35, 0.6, 1] });
    headIO.observe(head);
  }

  // The prognosis instrument: estimate on a calibrated 0–100 % scale, with the spread across model
  // refits (band) and what actually happened to similar recent registry patients (caret).
  let lastScale = null;
  // Which model the calculator uses: by default the one the pre-specified rule chose for each stage
  // and outcome; any other published learner can be shown instead.
  function modelChooser(st, pv) {
    const e = entryOf(st, "good_outcome");
    const names = Object.keys(e.learners);
    const sel = h("select", { class: "sel", id: "calc-model", "aria-label": tx("Model", "Model") });
    sel.appendChild(h("option", { value: "", selected: S.calcModel ? null : true }, tx("Chosen by the pre-specified rule", "Zvolený předem stanoveným pravidlem")));
    for (const nm of names) sel.appendChild(h("option", { value: nm, selected: S.calcModel === nm ? true : null }, mname(nm)));
    sel.addEventListener("change", () => { S.calcModel = sel.value || null; updateReadout(); });
    const now = pv.good.name === pv.death.name ? mname(pv.good.name) : `${mname(pv.good.name)} · ${tx("death", "úmrtí")}: ${mname(pv.death.name)}`;
    return h("div", { class: "card card-pad model-chooser" },
      h("label", { class: "controls", for: "calc-model", style: "margin:0" }, h("span", { text: tx("Model", "Model") }), sel),
      h("p", { class: "note", style: "margin:6px 0 0", text: S.calcModel
        ? tx(`Showing ${now}. The uncertainty range and the check against similar recent patients are computed for the chosen model only.`,
          `Zobrazen ${now}. Rozpětí nejistoty a srovnání s podobnými nedávnými pacienty jsou počítány jen pro zvolený model.`)
        : tx(`Using ${now}, chosen on the Validation tab by the rule fixed before testing.`, `Použit ${now}, zvolený na kartě Validace pravidlem stanoveným před testováním.`) }));
  }

  function headlineCard(st, pv) {
    const g = pv.good.p, d = pv.death.p;
    const bin = pv.good.chosen ? realityBin(st, g) : null;
    const unc = pv.good.chosen ? XE.uncertainty(((X.uncertainty || {})[st] || {}).good_outcome, g) : null;
    const card = h("div", { class: "card prognosis headline" });
    card.appendChild(h("div", { class: "prog-top" }, h("span", { class: "prog-stage", text: t().estimateAt(stageLabel(st)) })));
    card.appendChild(h("div", { class: "prog-main" },
      h("div", null,
        h("div", { class: "big" }, num(g * 100, 0), h("small", { text: lang === "cs" ? " %" : "%" })),
        h("div", { class: "big-label", text: t().headline })),
      h("div", { class: "side-stat" }, h("div", { class: "v", text: pct(d) }), h("div", { class: "l", text: t().deathLabel }),
        st === "d7" ? h("div", { class: "l", text: t().lateDeaths }) : null)));

    // scale
    const P = (v) => `${(Math.max(0, Math.min(1, v)) * 100).toFixed(2)}%`;
    const target = { g, lo: unc ? unc[0] : null, hi: unc ? unc[1] : null, obs: bin ? bin.obs : null };
    const from = lastScale || target;
    const scale = h("div", { class: "pscale", role: "img", "aria-label": t().scaleAria(pct(g), unc ? `${pct(unc[0])}–${pct(unc[1])}` : "", bin ? pct(bin.obs) : "") },
      h("div", { class: "rail" }));
    for (let k = 0; k <= 20; k++) scale.appendChild(h("i", { class: `tick${k % 5 === 0 ? " major" : ""}`, style: `left:${k * 5}%` }));
    [0, 25, 50, 75, 100].forEach((v) => scale.appendChild(h("span", { class: `tl${v === 100 ? " end" : ""}`, style: `left:${v}%`, text: v === 100 ? pct100(100) : String(v) })));
    const band = unc ? h("div", { class: "band", style: `left:${P(from.lo ?? target.lo)};width:${P((from.hi ?? target.hi) - (from.lo ?? target.lo))}` }) : null;
    const obs = bin ? h("div", { class: "obs", style: `left:${P(from.obs ?? target.obs)}` }) : null;
    const mark = h("div", { class: "mark", style: `left:${P(from.g)}` });
    if (band) scale.insertBefore(band, scale.children[1]);
    if (obs) scale.appendChild(obs);
    scale.appendChild(mark);
    card.appendChild(scale);
    if (lastScale && !(window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches)) {
      requestAnimationFrame(() => requestAnimationFrame(() => {
        mark.style.left = P(target.g);
        if (band) { band.style.left = P(target.lo); band.style.width = P(target.hi - target.lo); }
        if (obs) obs.style.left = P(target.obs);
      }));
    } else {
      mark.style.left = P(target.g);
      if (band) { band.style.left = P(target.lo); band.style.width = P(target.hi - target.lo); }
      if (obs) obs.style.left = P(target.obs);
    }
    lastScale = target;

    const key = h("div", { class: "pkey" });
    if (unc) key.appendChild(h("div", null, h("span", { class: "sw-band", "aria-hidden": "true" }),
      h("span", null, tx(`Range across ${num(((X.stability || {})[st] || {}).good_outcome ? X.stability[st].good_outcome.n_boot : 100)} refits of the model (95%): `, `Rozpětí z ${num(((X.stability || {})[st] || {}).good_outcome ? X.stability[st].good_outcome.n_boot : 100)} přetrénování modelu (95 %): `), h("b", { text: `${num(unc[0] * 100)}–${pct(unc[1])}` }))));
    if (bin) key.appendChild(h("div", null, h("span", { class: "sw-obs", "aria-hidden": "true" }),
      h("span", { html: t().reality(RECENT, pct(bin.lo), pct(bin.hi), pct(bin.obs), bin.n) })));
    if (key.childNodes.length) card.appendChild(key);
    return card;
  }

  // ------------------------------------------------------------------ full mRS distribution
  const MRS_GROUP_COLORS = [0, 1, 2, 3, 4, 5, 6].map((k) => `var(--mrs${k})`);
  const MRS_TEXT = [
    ["no symptoms", "bez příznaků"], ["no significant disability", "bez významného postižení"], ["slight disability", "lehké postižení"],
    ["moderate disability", "středně těžké postižení"], ["moderately severe", "středně těžké až těžké"], ["severe disability", "těžké postižení"], ["dead", "úmrtí"],
  ];

  function roundTo100(ps) {
    const raw = ps.map((p) => p * 100);
    const base = raw.map(Math.floor);
    let left = 100 - base.reduce((a, b) => a + b, 0);
    const order = raw.map((v, i) => [v - Math.floor(v), i]).sort((a, b) => b[0] - a[0]);
    for (let k = 0; k < left; k++) base[order[k % order.length][1]] += 1;
    return base;
  }

  function patientDistribution(st, pv) {
    const ord = (X.ordinal || {})[st];
    if (!ord) return null;
    const zg = pv.good.logit - (pv.good.spec.intercept_recent || 0);
    const zd = pv.death.logit - (pv.death.spec.intercept_recent || 0);
    return XE.fullDistribution(pv.good.p, pv.death.p, XE.ordinalProbs(zg, zd, ord));
  }

  function distributionCard(st, pv) {
    const dist = patientDistribution(st, pv);
    if (!dist) return null;
    // round the three bands first so the people count matches the headline, then split each band
    const band = roundTo100([dist[0] + dist[1] + dist[2], dist[3] + dist[4] + dist[5], dist[6]]);
    const split = (lo, hi, n) => {
      const part = dist.slice(lo, hi), tot = part.reduce((a, b) => a + b, 0) || 1;
      const raw = part.map((p) => (p / tot) * n), base = raw.map(Math.floor);
      const order = raw.map((v, i) => [v - Math.floor(v), i]).sort((a, b) => b[0] - a[0]);
      for (let k = 0; k < n - base.reduce((a, b) => a + b, 0); k++) base[order[k][1]] += 1;
      return base;
    };
    const counts = split(0, 3, band[0]).concat(split(3, 6, band[1]), [band[2]]);
    const uw = dist.reduce((a, p, k) => a + p * (X.uw_mrs || [])[k], 0);
    const card = h("div", { class: "card card-pad" });
    const seg = h("div", { class: "mini-seg", role: "group", "aria-label": tx("View", "Zobrazení") },
      [["bar", tx("mRS grades", "Stupně mRS")], ["people", tx("100 patients", "100 pacientů")]].map(([k, lbl]) => h("button", {
        type: "button", "aria-pressed": String(S.distView === k), onclick: () => { S.distView = k; updateReadout(); },
      }, lbl)));
    card.appendChild(h("div", { class: "chart-head" },
      h("div", null, h("div", { class: "chart-title", text: tx("Predicted 90-day mRS", "Předpokládané mRS za 90 dní") }),
        h("div", { class: "chart-sub", text: tx(`Full distribution at ${stageLabel(st)}`, `Celé rozložení · ${stageLabel(st)}`) })),
      seg));
    if (S.distView === "people") {
      const body = h("div", { class: "chart-body icon-array" });
      card.appendChild(body);
      C.iconArray(body, { counts, colors: MRS_GROUP_COLORS, labels: [0, 1, 2, 3, 4, 5, 6].map((k) => `mRS ${k} · ${tx(MRS_TEXT[k][0], MRS_TEXT[k][1])}`),
        aria: tx("100 patients by predicted mRS", "100 pacientů podle předpokládaného mRS") });
    } else {
      const bar = h("div", { class: "mrsbar", role: "img", "aria-label": dist.map((p, k) => `mRS ${k}: ${pct(p)}`).join(", ") });
      dist.forEach((p, k) => {
        const segEl = h("span", { style: `width:${(p * 100).toFixed(2)}%; background:${MRS_GROUP_COLORS[k]}` });
        if (p >= 0.07) segEl.appendChild(h("b", { style: `color:var(--mrs${k}-ink)`, text: String(counts[k]) }));
        segEl.addEventListener("pointermove", (ev) => C.tip.show(`mRS ${k}`, [{ label: tx(MRS_TEXT[k][0], MRS_TEXT[k][1]), color: MRS_GROUP_COLORS[k], value: pct(p) }], ev.clientX, ev.clientY));
        segEl.addEventListener("pointerleave", () => C.tip.hide());
        bar.appendChild(segEl);
      });
      card.appendChild(bar);
      card.appendChild(h("div", { class: "mrs-scale" }, [0, 1, 2, 3, 4, 5, 6].map((k) => h("span", { class: "key" }, h("i", { style: `background:${MRS_GROUP_COLORS[k]}` }), `${k}`))));
    }
    const g = counts[0] + counts[1] + counts[2], dd = counts[6], dep = 100 - g - dd;
    card.appendChild(h("p", { class: "plain", html: tx(
      `Of 100 patients like this, about <b>${g}</b> would live independently at 3 months, <b>${dep}</b> would need help in daily life and <b>${dd}</b> would die.`,
      `Ze 100 takových pacientů by přibližně <b>${g}</b> za 3 měsíce žilo samostatně, <b>${dep}</b> by potřebovalo pomoc v běžném životě a <b>${dd}</b> by zemřelo.`) }));
    card.appendChild(h("p", { class: "note", text: tx(
      `Utility-weighted mRS ${num(uw, 2)} (1 = no symptoms, 0 = severe disability or death). The split into single grades comes from an ordinal layer on the two models; its validation is on the Validation tab.`,
      `Užitkově vážené mRS ${num(uw, 2)} (1 = bez příznaků, 0 = těžké postižení nebo úmrtí). Rozdělení na jednotlivé stupně pochází z ordinální vrstvy nad oběma modely; její validace je na kartě Validace.`) }));
    return card;
  }

  // ------------------------------------------------------------------ other models for this patient
  function aucFor(st, name, split) {
    const sp = V.splits[split || "primary"];
    for (const s2 of [st, "admission", "h24"]) {
      const r = ((sp.stages[s2] || {}).targets || {}).good_outcome;
      if (r && r.models[name]) return { auc: r.models[name].auroc, stage: s2 };
    }
    return null;
  }

  function pointsText(name, v) {
    if (v === null || v === undefined || Number.isNaN(v)) return "";
    if (name === "SPAN-100") return v ? tx("positive", "pozitivní") : tx("negative", "negativní");
    if (name === "PRE") return `${v > 0 ? "+" : ""}${num(v)}`;
    if (name.startsWith("NIHSS")) return `NIHSS ${num(v)}`;
    return tx(`${num(v)} points`, `${num(v)} b.`);
  }

  // ------------------------------------------------------------------ FNO bedside score (7 items)
  const BEDSIDE_LABELS = {
    age: ["Age", "Věk"], nihss_adm: ["NIHSS on admission", "NIHSS při přijetí"], glucose: ["Glucose, mmol/L", "Glykémie, mmol/l"],
    mrs_pre: ["Pre-stroke mRS", "mRS před CMP"], aspects: ["ASPECTS", "ASPECTS"],
    onset_to_puncture: ["Onset to puncture, min", "Od vzniku do punkce, min"], stroke_prior: ["Previous stroke", "Předchozí CMP"],
  };

  function bedsideCard(st, x) {
    const comp = ((X.comparators || {})[st] || {})["Bedside score"];
    if (!comp || !XE.BEDSIDE) return null;
    const total = XE.POINTS["Bedside score"](x);
    const card = h("div", { class: "card card-pad bedside" });
    card.appendChild(h("div", { class: "chart-head" },
      h("div", null, h("div", { class: "chart-title", text: tx("Bedside score", "Bedside skóre") }),
        h("div", { class: "chart-sub", text: tx("Seven items from the earlier FNO analysis. More points, worse outlook.", "Sedm položek z dřívější analýzy FNO. Více bodů, horší výhled.") })),
      h("div", { class: "bs-total", "aria-label": tx("Total points", "Celkem bodů") },
        h("b", { text: Number.isNaN(total) ? "–" : num(total) }), h("span", { text: tx("of 20", "z 20") }))));
    const yesNo = (v) => (v === 1 ? tx("yes", "ano") : tx("no", "ne"));
    const list = h("div", { class: "bs-list" });
    for (const item of XE.BEDSIDE) {
      const k = item.key;
      const v = x[k];
      const m = item.match(x);
      const shown = v === null || v === undefined ? tx("not recorded", "nezadáno")
        : k === "stroke_prior" ? yesNo(v) : k === "onset_to_puncture" ? t().hm(v) : num(v, k === "glucose" ? 1 : 0);
      list.appendChild(h("div", { class: `bs-row${m < 0 ? " unknown" : ""}` },
        h("div", { class: "bs-name" }, h("b", { text: tx(BEDSIDE_LABELS[k][0], BEDSIDE_LABELS[k][1]) }), h("span", { text: shown })),
        h("div", { class: "bs-bands" }, item.bands.map((b, i) => h("span", { class: `bs-band${i === m ? " on" : ""}` },
          b.label === "no" ? tx("no", "ne") : b.label === "yes" ? tx("yes", "ano") : b.label, h("i", { text: String(b.pts) })))),
        h("div", { class: "bs-pts", text: m < 0 ? "0" : String(item.bands[m].pts) })));
    }
    card.appendChild(list);
    if (!Number.isNaN(total)) {
      const p = XE.pointsProb(comp.map, total);
      card.appendChild(h("p", { class: "plain", html: tx(
        `In recent registry patients, ${num(total)} points corresponds to a <b>${pct(p)}</b> chance of mRS 0–2.`,
        `U nedávných pacientů registru odpovídá ${num(total)} bodů šanci na mRS 0–2 <b>${pct(p)}</b>.`) }));
    }
    const a = aucFor(st, "Bedside score");
    const am = a ? aucFor(a.stage, chosenOf(a.stage, "good_outcome")) : null;
    const at = a && a.stage !== st;
    card.appendChild(h("p", { class: "note", text: tx(
      `AUROC ${a ? num(a.auc, 2) : "–"} in ${yearsOf()} patients, against ${am ? num(am.auc, 2) : "–"} for MTOP${at ? " (both before treatment)" : " at this stage"}. Items that are not recorded score 0.`,
      `AUROC ${a ? num(a.auc, 2) : "–"} u pacientů ${yearsOf()}, MTOP ${am ? num(am.auc, 2) : "–"}${at ? " (obojí před léčbou)" : " ve stejné fázi"}. Nezadané položky mají 0 bodů.`) }));
    card.appendChild(h("p", { class: "note", text: st === "admission"
      ? tx("Why it can differ from MTOP for one patient: the score uses seven items in coarse bands and leaves out the other 26 admission predictors (occlusion site, IV thrombolysis, transfer, blood pressure, sex, most of the history). It ranks groups of patients almost as well as MTOP, but for an individual the two often differ by 10 percentage points or more.",
        "Proč se může od MTOP u jednoho pacienta lišit: skóre používá sedm položek v hrubých pásmech a vynechává ostatních 26 prediktorů při přijetí (místo okluze, IV trombolýzu, překlad, tlak, pohlaví, většinu anamnézy). Skupiny pacientů seřadí téměř stejně dobře jako MTOP, u jednotlivce se však oba odhady často liší o 10 procentních bodů i více.")
      : tx(`The score uses admission information only. MTOP at ${stageLabel(st).toLowerCase()} also knows ${st === "procedure" ? "the reperfusion result and procedure details" : "the reperfusion result and the NIHSS course since admission"}, so for one patient the two can differ a lot; compare the score with MTOP's admission estimate in the pathway panel.`,
        `Skóre používá jen informace při přijetí. MTOP ve fázi ${stageLabel(st).toLowerCase()} zná navíc ${st === "procedure" ? "výsledek reperfuze a detaily výkonu" : "výsledek reperfuze a vývoj NIHSS od přijetí"}, takže se u jednoho pacienta mohou oba odhady výrazně lišit; skóre srovnávejte s odhadem MTOP při přijetí v panelu průběhu.`) }));
    return card;
  }

  function othersCard(st, x, pv) {
    const comps = (X.comparators || {})[st] || {};
    const e = entryOf(st, "good_outcome");
    const xs = samples(e, x);
    const rows = [];
    for (const nm of Object.keys(e.learners)) {
      const p = nm === pv.good.name ? pv.good.p : predictStage(st, x, "good_outcome", nm).p;
      rows.push({ name: mname(nm), detail: nm === e.chosen ? tx("chosen by the rule", "zvolen pravidlem") : "", p, auc: aucFor(st, nm), main: nm === pv.good.name });
    }
    for (const [name, comp] of Object.entries(comps)) {
      let sum = 0, ok = 0, pts = null;
      for (const xi of name === "Bedside score" ? [x] : xs) {
        let v;
        if (comp.kind === "points") {
          const sc = XE.POINTS[name](xi);
          v = XE.pointsProb(comp.map, sc);
          if (xs.length === 1 || name === "Bedside score") pts = sc;
        } else v = XE.comparatorProb(comp, xi);
        if (v !== null && !Number.isNaN(v)) { sum += v; ok++; }
      }
      rows.push({ name: mname(name), detail: pointsText(name, pts), p: ok ? sum / ok : null, auc: aucFor(st, name) });
    }
    const card = h("div", { class: "card card-pad" });
    card.appendChild(h("div", { class: "chart-head" }, h("div", null,
      h("div", { class: "chart-title", text: tx("All models for this patient", "Všechny modely pro tohoto pacienta") }),
      h("div", { class: "chart-sub", text: tx(`Chance of mRS 0–2 at ${stageLabel(st)}; AUROC from the ${yearsOf()} validation`, `Šance na mRS 0–2 · ${stageLabel(st)}; AUROC z validace ${yearsOf()}`) }))));
    const list = h("div", { class: "others" });
    for (const r of rows) {
      const track = h("div", { class: "track", "aria-hidden": "true" });
      if (r.p !== null) track.appendChild(h("i", { style: `left:${(r.p * 100).toFixed(1)}%` }));
      const aucTxt = r.auc ? `${num(r.auc.auc, 2)}${r.auc.stage !== st ? "*" : ""}` : "–";
      list.appendChild(h("div", { class: `orow${r.main ? " main" : ""}` },
        h("div", { class: "nm" }, h("b", { text: r.name }), r.detail ? h("span", { text: ` · ${r.detail}` }) : null),
        track,
        h("div", { class: "pv", text: r.p === null ? "–" : pct(r.p) }),
        h("div", { class: "au", text: aucTxt })));
    }
    list.appendChild(h("div", { class: "orow ohead" }, h("div"), h("div", { class: "scale" }, h("span", { text: "0%" }), h("span", { text: "50%" }), h("span", { text: "100%" })), h("div"), h("div", { class: "au", text: "AUROC" })));
    card.appendChild(list);
    card.appendChild(h("p", { class: "note", text: tx(
      `The four learners use exactly the same ${e.columns.length} model inputs and are shifted to the ${RECENT} outcome level. Published scores are mapped to probabilities with a logistic recalibration on the FNO registry; THRIVE-c, MR PREDICTS@24H and the bedside-variables regression are their predictor sets refitted on the registry (MR PREDICTS@24H without the collateral score, which the registry does not record). * AUROC measured at an earlier stage.`,
      `Čtyři modely používají přesně stejných ${e.columns.length} vstupů a jsou posunuty na úroveň výsledků ${RECENT}. Publikovaná skóre jsou převedena na pravděpodobnosti logistickou rekalibrací na registru FNO; THRIVE-c, MR PREDICTS@24H a regrese z položek bedside skóre jsou jejich sady prediktorů přepočtené na registru (MR PREDICTS@24H bez skóre kolaterál, které registr neeviduje). * AUROC změřené v dřívější fázi.`) }));
    return card;
  }

  function trajectoryCard(preds) {
    const card = h("div", { class: "card" });
    card.appendChild(h("div", { class: "panel-head" }, h("h3", { text: t().trajTitle })));
    const box = h("div", { class: "traj" });
    let prev = null;
    for (const st of STAGES) {
      const p = preds[st];
      const future = !p;
      const row = h("button", { class: `traj-row${future ? " future" : ""}`, type: "button", "aria-current": st === S.viewed ? "step" : null });
      row.appendChild(h("span", { class: "st" }, stageLabel(st)));
      if (future) {
        row.appendChild(h("span", { class: "bar3", "aria-hidden": "true" }));
        row.appendChild(h("span", { class: "pct", text: t().addData }));
        row.addEventListener("click", () => setCurrent(st));
      } else {
        const [a, b, c] = threeBand(p.good.p, p.death.p);
        const bar = h("span", { class: "bar3", role: "img", "aria-label": `${t().indep} ${pct(a)}, ${t().dep} ${pct(b)}, ${t().dead} ${pct(c)}` },
          h("i", { class: "b0", style: `width:${a * 100}%` }), h("i", { class: "b1", style: `width:${b * 100}%` }), h("i", { class: "b2", style: `width:${c * 100}%` }));
        row.appendChild(bar);
        const pc = h("span", { class: "pct" }, pct(p.good.p));
        if (prev !== null) {
          const dpp = Math.round((p.good.p - prev) * 100);
          pc.appendChild(h("span", { class: "delta", style: "display:block", text: t().vsPrev(dpp) }));
        }
        row.appendChild(pc);
        row.addEventListener("click", () => { S.viewed = st; updateReadout(); });
        row.addEventListener("pointermove", (ev) => C.tip.show(stageLabel(st), [
          { label: t().indep, color: "var(--accent)", value: pct(a) },
          { label: t().dep, color: "var(--dep)", value: pct(b) },
          { label: t().dead, color: "var(--dead)", value: pct(c) }], ev.clientX, ev.clientY));
        row.addEventListener("pointerleave", () => C.tip.hide());
        prev = p.good.p;
      }
      box.appendChild(row);
    }
    card.appendChild(box);
    card.appendChild(h("div", { class: "traj-legend" },
      h("span", { class: "key" }, h("i", { style: "background:var(--accent)" }), t().indep),
      h("span", { class: "key" }, h("i", { style: "background:var(--dep)" }), t().dep),
      h("span", { class: "key" }, h("i", { style: "background:var(--dead)" }), t().dead)));
    return card;
  }

  function explainCard(st, pv) {
    const target = S.explainTarget;
    const pr = target === "good_outcome" ? pv.good : pv.death;
    const ex = explainStage(pr);
    const p = pr.p;
    const logit = pr.logit;
    const rows = Object.entries(ex.byFeature).map(([k, phi]) => ({ k, phi }));
    rows.sort((a, b) => Math.abs(b.phi) - Math.abs(a.phi));
    const top = rows.slice(0, 7);
    const rest = rows.slice(7);
    const restPhi = rest.reduce((s, r) => s + r.phi, 0);
    const maxAbs = Math.max(0.05, ...top.map((r) => Math.abs(r.phi)), Math.abs(restPhi));
    const card = h("div", { class: "card" });
    const seg = h("div", { class: "mini-seg", role: "group", "aria-label": t().outcome },
      [["good_outcome", t().targetGood], ["death90", t().targetDeath]].map(([k, lbl]) => h("button", {
        type: "button", "aria-pressed": String(target === k), onclick: () => { S.explainTarget = k; updateReadout(); },
      }, lbl)));
    card.appendChild(h("div", { class: "panel-head" }, h("h3", { text: t().explainTitle }), seg));
    card.appendChild(h("p", { class: "panel-sub", text: t().explainSub(stageLabel(st), target === "good_outcome" ? t().targetGoodLong : t().targetDeathLong) }));
    const list = h("div", { class: "shap" });
    const drawRow = (name, valueText, phi) => {
      const eff = M.sigmoid(logit) - M.sigmoid(logit - phi);
      const pp = Math.round(eff * 100);
      const w = (Math.abs(phi) / maxAbs) * 50;
      const pos = phi >= 0;
      const bar = h("i", { class: pos ? "pos" : "neg", style: pos ? `left:50%;width:${w}%` : `left:${50 - w}%;width:${w}%` });
      const effText = pp === 0 ? "±0" : `${pp > 0 ? "+" : "−"}${Math.abs(pp)}`;
      return h("div", { class: "shap-row" },
        h("div", { class: "nm", title: `${name}: ${valueText}` }, h("b", { text: name }), valueText ? h("span", { text: ` · ${valueText}` }) : null),
        h("div", { class: "track", "aria-hidden": "true" }, bar),
        h("div", { class: `eff ${pos ? "pos" : "neg"}`, text: `${effText}${lang === "cs" ? " p. b." : " pp"}` }));
    };
    const xin = modelInputs();
    for (const r of top) list.appendChild(drawRow(L(FEAT[r.k].label), displayValue(r.k, xin[r.k]), r.phi));
    if (rest.length) list.appendChild(drawRow(t().others(rest.length), "", restPhi));
    list.appendChild(h("div", { class: "shap-axis" }, h("div"), h("div", null, h("span", { text: t().lowers }), h("span", { text: t().raises })), h("div")));
    list.appendChild(h("p", { class: "shap-base", text: t().baseline(pct(M.sigmoid(ex.base + (pr.spec.intercept_recent || 0)))) }));
    if (pr.unknown.length) card.appendChild(h("p", { class: "panel-sub", style: "padding-bottom:14px", text: t().unknownNote(pr.unknown.map((k) => L(FEAT[k].label)).join(", ")) }));
    card.appendChild(h("p", { class: "panel-sub", style: "padding-bottom:10px", text: pr.spec.kind === "lgbm" || pr.spec.kind === "xgb"
      ? tx(`${mname(pr.name)}: exact TreeSHAP values, computed in this browser.`, `${mname(pr.name)}: přesné hodnoty TreeSHAP spočtené v tomto prohlížeči.`)
      : tx(`${mname(pr.name)}: each factor's term of the regression (coefficient × standardised value), which is its exact Shapley value; spline terms and missing-value indicators are added to their factor.`,
        `${mname(pr.name)}: člen regrese pro každý faktor (koeficient × standardizovaná hodnota), což je jeho přesná Shapleyho hodnota; členy splajnů a indikátory chybění jsou přičteny k jejich faktoru.`) }));
    card.appendChild(list);
    return card;
  }

  const WHATIF_DEFAULT = { admission: "onset_to_puncture", procedure: "tici", h24: "nihss_24h", d7: "nihss_7d" };
  const WHATIF_KEYS = ["onset_to_puncture", "age", "nihss_adm", "aspects", "glucose", "sbp", "mrs_pre", "onset_to_ivt", "bmi", "tici", "procedure_min", "nihss_24h", "nihss_7d"];

  function whatifCard(st, x) {
    const avail = WHATIF_KEYS.filter((k) => FEAT[k] && idx(FEAT[k].stage) <= idx(st) && !(FEAT[k].requires_ivt && S.inputs.ivt === 0));
    let key = S.whatif[st] && avail.includes(S.whatif[st]) ? S.whatif[st] : WHATIF_DEFAULT[st];
    if (!avail.includes(key)) key = avail[0];
    const f = FEAT[key];
    const card = h("div", { class: "card card-pad" });
    const sel = h("select", { class: "sel", id: "whatif-var", "aria-label": t().vary });
    for (const k of avail) sel.appendChild(h("option", { value: k, selected: k === key ? true : null }, L(FEAT[k].label)));
    sel.addEventListener("change", () => { S.whatif[st] = sel.value; updateReadout(); });
    card.appendChild(h("div", { class: "chart-head" },
      h("div", null, h("div", { class: "chart-title", text: t().whatifTitle }), h("div", { class: "chart-sub", text: t().whatifSub(stageLabel(st)) })),
      h("label", { class: "controls", style: "margin:0" }, h("span", { text: t().vary }), sel)));
    // grid
    let grid;
    const ordinal = !!f.options;
    if (ordinal) grid = f.options.map((o) => o.value);
    else {
      const n = 48;
      const stp = f.step || 1;
      grid = [];
      for (let i = 0; i <= n; i++) grid.push(Math.round((f.min + ((f.max - f.min) * i) / n) / stp) * stp);
      grid = Array.from(new Set(grid.map((v) => +v.toFixed(4))));
    }
    const good = [], death = [];
    grid.forEach((gv, i) => {
      const xi = Object.assign({}, x, { [key]: gv });
      const pg = predictStage(st, xi, "good_outcome").p;
      const pd = predictStage(st, xi, "death90").p;
      const xv = ordinal ? i : gv;
      good.push({ x: xv, y: pg * 100 });
      death.push({ x: xv, y: pd * 100 });
    });
    const body = h("div", { class: "chart-body" });
    const fig = h("figure", { class: "chart" },
      h("div", { class: "legend" },
        h("span", { class: "key" }, h("i", { class: "line", style: "background:var(--accent)" }), t().targetGood),
        h("span", { class: "key" }, h("i", { class: "line", style: "background:var(--dead)" }), t().targetDeath)),
      body);
    card.appendChild(fig);
    const cur = x[key];
    const curX = cur === null || cur === undefined ? null : ordinal ? f.options.findIndex((o) => o.value === cur) : cur;
    const fmtX = (v) => (TIMES.has(key) ? `${num(v / 60, v % 60 === 0 ? 0 : 1)} h` : num(v, f.step && f.step < 1 ? 1 : 0));
    C.lineChart(body, {
      height: 190,
      x: ordinal ? { type: "cat", cats: f.options.map((o) => (key === "mrs_pre" ? String(o.value) : L(o.label))) } : { type: "lin", domain: [f.min, f.max], fmt: fmtX, ticks: key === "onset_to_puncture" ? [60, 360, 720, 1080, 1440] : undefined },
      y: { domain: [0, 100], fmt: (v) => `${v}`, ticks: [0, 25, 50, 75, 100] },
      series: [
        { label: t().targetGood, color: "var(--accent)", points: good, dots: ordinal },
        { label: t().targetDeath, color: "var(--dead)", points: death, dots: ordinal },
      ],
      marker: curX !== null && curX >= 0 ? { x: curX, label: t().currentValue } : null,
      tipFmt: (v) => pct100(v),
      tipHead: (xv) => (ordinal ? L(f.options[xv].label) : displayValue(key, xv)),
      aria: `${t().whatifTitle} ${L(f.label)}`,
    });
    return card;
  }

  // ================================================================== registry
  const MRS_COLORS = [0, 1, 2, 3, 4, 5, 6].map((k) => `var(--mrs${k})`);
  const MRS_INK = ["light", "light", "dark", "dark", "light", "light", "light"];

  function kpi(label, value, sub) {
    return h("div", { class: "kpi" }, h("div", { class: "l", text: label }), h("div", { class: "v", text: value }), sub ? h("div", { class: "s", text: sub }) : null);
  }

  // outcome by age band x NIHSS band (cells with fewer than 30 patients are not published)
  S.regMap = S.regMap || "good";
  const lerp = (a, b, f) => a.map((v, i) => Math.round(v + (b[i] - v) * f));
  const hex2rgb = (x) => [1, 3, 5].map((i) => parseInt(x.slice(i, i + 2), 16));
  function ageNihssCard() {
    const M = E.age_nihss;
    const good = S.regMap === "good";
    const card = h("div", { class: "card card-pad block" });
    const seg = h("div", { class: "mini-seg", role: "group", "aria-label": t().outcome },
      [["good", t().targetGood], ["death", t().targetDeath]].map(([k, lbl]) => h("button", {
        type: "button", "aria-pressed": String(S.regMap === k), onclick: () => { S.regMap = k; card.replaceWith(ageNihssCard()); },
      }, lbl)));
    card.appendChild(h("div", { class: "chart-head" }, h("div", null,
      h("div", { class: "chart-title", text: tx("Outcome by age and stroke severity", "Výsledek podle věku a závažnosti CMP") }),
      h("div", { class: "chart-sub", text: good
        ? tx("Share of patients independent at 90 days (mRS 0–2), by age and NIHSS on admission", "Podíl pacientů soběstačných za 90 dní (mRS 0–2) podle věku a NIHSS při přijetí")
        : tx("Share of patients who died within 90 days, by age and NIHSS on admission", "Podíl pacientů zemřelých do 90 dní podle věku a NIHSS při přijetí") })), seg));
    const lo = hex2rgb(good ? "#eef4fd" : "#fdf1ec"), hi = hex2rgb(good ? "#10428f" : "#8f2f16");
    const top = good ? 80 : 60;
    const tbl = h("table", { class: "heatmap" });
    tbl.appendChild(h("caption", { class: "visually-hidden", text: tx("Outcome by age band and NIHSS band", "Výsledek podle věkové skupiny a skupiny NIHSS") }));
    tbl.appendChild(h("thead", null, h("tr", null, h("th", { scope: "col", class: "corner", text: tx("Age ↓ NIHSS →", "Věk ↓ NIHSS →") }),
      M.cols.map((c) => h("th", { scope: "col", text: c })))));
    const tb = h("tbody");
    M.rows.forEach((r, i) => {
      tb.appendChild(h("tr", null, h("th", { scope: "row", text: r }), M.cells[i].map((cell) => {
        if (cell.suppressed) return h("td", { class: "sup", title: tx(`fewer than ${M.min_n} patients`, `méně než ${M.min_n} pacientů`) }, h("span", { text: "–" }));
        const v = good ? cell.good : cell.death;
        const f = Math.max(0, Math.min(1, v / top));
        const rgb = lerp(lo, hi, f);
        return h("td", { style: `background:rgb(${rgb.join(",")});color:${f > 0.52 ? "#ffffff" : "var(--ink)"}` },
          h("b", { text: pct100(v, 0) }), h("span", { text: `n ${num(cell.n)}` }));
      })));
    });
    tbl.appendChild(tb);
    card.appendChild(h("div", { class: "table-scroll" }, tbl));
    card.appendChild(h("p", { class: "note", text: tx(
      `Observed outcomes in the registry, not model estimates. Cells with fewer than ${M.min_n} patients are hidden. Age and severity explain much of the difference between patients; the calculator adds everything else that is known.`,
      `Pozorované výsledky v registru, nikoli odhady modelu. Buňky s méně než ${M.min_n} pacienty jsou skryté. Věk a závažnost vysvětlují velkou část rozdílů mezi pacienty; kalkulačka přidává vše ostatní, co je známo.`) }));
    return card;
  }

  // factors associated with 90-day outcome: unadjusted and adjusted odds ratios (associations.json)
  S.regOr = S.regOr || "good";
  function associationCard() {
    const A = DATA.associations;
    const good = S.regOr === "good";
    const res = A.outcomes[good ? "good_outcome" : "death90"];
    const card = h("div", { class: "card card-pad block" });
    const seg = h("div", { class: "mini-seg", role: "group", "aria-label": t().outcome },
      [["good", t().targetGood], ["death", t().targetDeath]].map(([k, lbl]) => h("button", {
        type: "button", "aria-pressed": String(S.regOr === k), onclick: () => { S.regOr = k; card.replaceWith(associationCard()); },
      }, lbl)));
    const orTxt = (e) => `${num(e.or, 2)} (${num(e.lo, 2)}–${num(e.hi, 2)})`;
    const UN = tx("Unadjusted", "Neadjustovaný"), AD = tx("Adjusted", "Adjustovaný");
    const groups = Object.keys(A.groups).map((g) => ({
      label: L(A.groups[g]),
      rows: A.terms.filter((tm) => tm.group === g).map((tm) => {
        const r = res.terms[tm.key];
        return {
          label: L(tm.label), unit: tm.unit ? L(tm.unit) : "", a: r.unadjusted, b: r.adjusted, display: orTxt(r.adjusted),
          strong: r.adjusted.lo > 1 || r.adjusted.hi < 1,
          tip: [{ label: UN, value: orTxt(r.unadjusted) }, { label: AD, value: orTxt(r.adjusted) },
            { label: tx("Recorded", "Zaznamenáno"), value: `${num(r.unadjusted.n)} / ${num(res.n)}` }],
        };
      }),
    })).filter((g) => g.rows.length);
    const tableRows = [];
    for (const g of groups) for (const r of g.rows) tableRows.push([r.label, r.unit || "–", orTxt(r.a), num(r.a.n), orTxt(r.b)]);
    const f = C.figure(card, {
      title: tx("Factors associated with the 90-day outcome", "Faktory spojené s výsledkem za 90 dní"),
      sub: good
        ? tx(`Odds ratios for independence at 90 days (mRS 0–2) with 95% confidence intervals, ${num(res.n)} patients. Above 1: more often independent.`,
          `Poměry šancí pro soběstačnost za 90 dní (mRS 0–2) s 95% intervaly spolehlivosti, ${num(res.n)} pacientů. Nad 1: častěji soběstační.`)
        : tx(`Odds ratios for death within 90 days with 95% confidence intervals, ${num(res.n)} patients. Above 1: higher mortality.`,
          `Poměry šancí pro úmrtí do 90 dní s 95% intervaly spolehlivosti, ${num(res.n)} pacientů. Nad 1: vyšší mortalita.`),
      legend: [{ label: tx("Unadjusted: one factor at a time", "Neadjustovaný: každý faktor zvlášť"), type: "ring" },
        { label: tx("Adjusted: all factors together", "Adjustovaný: všechny faktory společně"), type: "sq", color: good ? "var(--blue)" : "var(--dead)" }],
      table: { columns: [tx("Factor", "Faktor"), tx("Unit / comparison", "Jednotka / srovnání"), `${UN} OR (95% CI)`, tx("n recorded", "n se záznamem"), `${AD} OR (95% CI)`], rows: tableRows },
      labels: { table: t().showTable, chart: t().showChart },
    });
    const fh = f.fig.querySelector(".chart-head");
    fh.insertBefore(seg, fh.querySelector(".tbl-toggle"));
    C.orForest(f.body, {
      groups, color: good ? "var(--blue)" : "var(--dead)", tickFmt: (v) => num(v, v < 1 ? 1 : 0),
      left: good ? tx("← less often independent", "← méně často soběstační") : tx("← lower mortality", "← nižší mortalita"),
      right: good ? tx("more often independent →", "častěji soběstační →") : tx("higher mortality →", "vyšší mortalita →"),
      leftShort: tx("← lower odds", "← nižší šance"), rightShort: tx("higher odds →", "vyšší šance →"),
      aria: tx("Forest plot of odds ratios", "Lesní graf poměrů šancí"),
    });
    const notes = h("div", { class: "or-notes" });
    const yr = res.terms.year5 && res.terms.year5.adjusted;
    if (yr) {
      const what = good ? tx("independence (mRS 0–2)", "na soběstačnost (mRS 0–2)") : tx("death", "úmrtí");
      const ci = `OR ${num(yr.or, 2)}, ${tx("95% CI", "95% IS")} ${num(yr.lo, 2)}–${num(yr.hi, 2)}`;
      if (yr.hi < 1 || yr.lo > 1) {
        const ch = Math.round(Math.abs(1 - yr.or) * 100);
        const dir = yr.hi < 1 ? tx("lower", "nižší") : tx("higher", "vyšší");
        notes.appendChild(h("p", { class: "callout", text: tx(
          `Outcomes drifted: with all the other factors accounted for, the odds of ${what} were ${ch}% ${dir} for every five years later in the registry (${ci}). This is why MTOP's estimates are shifted to the outcome level of ${RECENT}.`,
          `Výsledky se v čase posunuly: po zohlednění všech ostatních faktorů byla šance ${what} o ${ch} % ${dir} za každých pět let pozdější léčby (${ci}). Proto jsou odhady MTOP posunuty na úroveň výsledků let ${RECENT}.`) }));
      } else {
        notes.appendChild(h("p", { class: "note", text: tx(
          `With all the other factors accounted for, the odds of ${what} showed no clear change over the years (OR ${num(yr.or, 2)} per five years, 95% CI ${num(yr.lo, 2)}–${num(yr.hi, 2)}).`,
          `Po zohlednění ostatních faktorů se šance ${what} v průběhu let jasně neměnila (OR ${num(yr.or, 2)} na pět let, 95% IS ${num(yr.lo, 2)}–${num(yr.hi, 2)}).`) }));
      }
    }
    notes.appendChild(h("p", { class: "note", text: tx(
      `Unadjusted: logistic regression with one factor, in the patients with that value recorded. Adjusted: one logistic model with all factors and the treatment year; missing values were filled in ${A.imputations} times by multiple imputation and the results combined (Rubin's rules), so all ${num(res.n)} patients count. Bold: the adjusted interval excludes 1. Intervals are not corrected for multiple comparisons.`,
      `Neadjustovaný: logistická regrese s jedním faktorem u pacientů, u nichž je hodnota zaznamenána. Adjustovaný: jeden logistický model se všemi faktory a rokem léčby; chybějící hodnoty byly ${A.imputations}× doplněny vícenásobnou imputací a výsledky sloučeny (Rubinova pravidla), takže se započítá všech ${num(res.n)} pacientů. Tučně: adjustovaný interval nezahrnuje 1. Intervaly nejsou korigovány na mnohočetné srovnávání.`) }));
    notes.appendChild(h("p", { class: "note", text: tx(
      "These are associations in one registry, not effects of treatment choices: general anaesthesia, for example, is used more often in restless or more severely affected patients, and successful reperfusion partly reflects favourable anatomy. The calculator does not use these odds ratios; its model also captures non-linear effects and interactions.",
      "Jde o souvislosti v jednom registru, nikoli o účinky léčebných rozhodnutí: například celková anestezie se častěji volí u neklidných nebo závažněji postižených pacientů a úspěšná reperfuze zčásti odráží příznivou anatomii. Kalkulačka tyto poměry šancí nepoužívá; její model zachycuje i nelineární vztahy a interakce.") }));
    card.appendChild(notes);
    return card;
  }

  function renderRegistry() {
    const root = $("#view-registry");
    root.replaceChildren();
    const k = E.kpis;
    const sub = h("div", { class: "mini-seg subtabs", role: "group", "aria-label": tx("Registry sections", "Části registru") },
      [["outcomes", tx("Outcomes and trends", "Výsledky a trendy")], ["data", tx("Data and preprocessing", "Data a předzpracování")]].map(([key, lbl]) => h("button", {
        type: "button", "aria-pressed": String(S.regView === key), onclick: () => { S.regView = key; renderRegistry(); },
      }, lbl)));
    root.appendChild(h("div", { class: "view-head" }, h("div", null,
      h("h2", { text: t().regTitle }), h("p", { text: t().regDesc(num(k.n), E.period[0], E.period[1], E.min_cell) }), sub)));
    if (S.regView === "data" && window.MTOPApp.preprocessingView) { window.MTOPApp.preprocessingView(root); return; }
    root.appendChild(h("div", { class: "kpis" },
      kpi(t().kPatients, num(k.n), lang === "cs" ? `z ${num(E.flow[0].n)} v registru` : `of ${num(E.flow[0].n)} in the registry`),
      kpi(t().kAge, `${num(k.age_median)}`, t().years),
      kpi(t().kNihss, num(k.nihss_median)),
      kpi(t().kOtp, `${num(k.otp_median)}`, t().minutes),
      kpi(t().kTici, pct100(k.tici2b3, 0)),
      kpi(t().kGood, pct100(k.good, 0)),
      kpi(t().kDeath, pct100(k.death, 0)),
      kpi(t().kSich, pct100(k.sich, 0))));

    // Grotta by subgroup
    const gBlock = h("div", { class: "card card-pad block" });
    const dimSel = h("select", { class: "sel", id: "reg-dim" });
    for (const [key, d] of Object.entries(E.dimensions)) dimSel.appendChild(h("option", { value: key, selected: key === S.regDim ? true : null }, L(d.label)));
    dimSel.addEventListener("change", () => { S.regDim = dimSel.value; renderRegistry(); });
    const dimCtl = h("div", { class: "controls" }, h("label", { for: "reg-dim" }, t().groupBy), dimSel);
    const dim = E.dimensions[S.regDim];
    const overallLabel = lang === "cs" ? "Celý soubor" : "Whole cohort";
    const rows = [{ label: overallLabel, n: E.overall.n, segs: E.overall.dist }].concat(dim.groups.map((g) => ({
      label: L(g.label), n: g.n, segs: g.dist, suppressed: g.suppressed || !g.dist,
    })));
    const legend = [0, 1, 2, 3, 4, 5, 6].map((m) => ({ label: `mRS ${m}`, color: MRS_COLORS[m] }));
    const tableRows = rows.map((r) => [r.label, r.suppressed ? t().suppressed : r.n].concat(r.suppressed || !r.segs ? Array(7).fill(null) : r.segs.map((v) => pct100(v, 1))));
    const fg = C.figure(gBlock, {
      title: t().grottaTitle, sub: t().grottaSub, legend,
      table: { columns: [t().variable, "n"].concat([0, 1, 2, 3, 4, 5, 6].map((m) => `mRS ${m}`)), rows: tableRows },
      labels: { table: t().showTable, chart: t().showChart },
    });
    fg.fig.insertBefore(dimCtl, fg.fig.children[1]);
    if (dim.note) gBlock.appendChild(h("p", { class: "note", text: L(dim.note) }));
    C.grotta(fg.body, { rows, colors: MRS_COLORS, inkOn: MRS_INK, segLabels: legend.map((l) => l.label), fmt: (v) => pct100(v, 1), suppressedText: lang === "cs" ? "méně než 10 pacientů nebo méně než 30 pro rozložení" : "too few patients to show", nLabel: "n", aria: t().grottaTitle });
    root.appendChild(gBlock);
    if (E.age_nihss) root.appendChild(ageNihssCard());
    if (DATA.associations) root.appendChild(associationCard());

    // time + outcome by year
    const two = h("div", { class: "grid-2 block" });
    const tc = E.time_curve;
    const tcBins = tc.all.map((b) => b.bin);
    const tSeries = [
      { label: t().allPatients, color: "var(--c1)", offset: -4, dots: true, points: tc.all.map((b, i) => ({ x: i, y: b.suppressed ? null : b.good, lo: b.lo, hi: b.hi, n: b.n })) },
      { label: t().reperfused, color: "var(--c2)", offset: 4, dots: true, points: tc.reperfused.map((b, i) => ({ x: i, y: b.suppressed ? null : b.good, lo: b.lo, hi: b.hi, n: b.n })) },
    ];
    const timeCard = h("div", { class: "card card-pad" });
    const ft = C.figure(timeCard, {
      title: t().timeTitle, sub: t().timeSub,
      legend: tSeries.map((s) => ({ label: s.label, color: s.color, type: "line" })),
      table: { columns: [t().timeTitle, `${t().allPatients} (n)`, t().allPatients, `${t().reperfused} (n)`, t().reperfused],
        rows: tcBins.map((b, i) => [b, tc.all[i].n ?? t().suppressed, tc.all[i].suppressed ? null : `${pct100(tc.all[i].good, 1)} (${num(tc.all[i].lo, 1)}–${num(tc.all[i].hi, 1)})`, tc.reperfused[i].n ?? t().suppressed, tc.reperfused[i].suppressed ? null : `${pct100(tc.reperfused[i].good, 1)} (${num(tc.reperfused[i].lo, 1)}–${num(tc.reperfused[i].hi, 1)})`]) },
      labels: { table: t().showTable, chart: t().showChart },
    });
    C.lineChart(ft.body, { height: 230, x: { type: "cat", cats: tcBins }, y: { domain: [0, 100], ticks: [0, 25, 50, 75, 100] }, series: tSeries,
      tipFmt: (v, p) => `${pct100(v, 0)} · n ${p.n}`, aria: t().timeTitle });
    two.appendChild(timeCard);

    const tr = E.trends;
    const yrs = tr.map((r) => r.label);
    const outCard = h("div", { class: "card card-pad" });
    const oSeries = [
      { label: t().tGood.replace(" (%)", ""), color: "var(--c1)", points: tr.map((r, i) => ({ x: i, y: r.good })) },
      { label: t().tDeath.replace(" (%)", ""), color: "var(--c2)", points: tr.map((r, i) => ({ x: i, y: r.death })) },
    ];
    const fo = C.figure(outCard, {
      title: lang === "cs" ? "Výsledky za 90 dní podle roku" : "90-day outcomes by year", sub: lang === "cs" ? "Podíl pacientů (%)" : "Share of patients (%)",
      legend: oSeries.map((s) => ({ label: s.label, color: s.color, type: "line" })),
      table: { columns: [lang === "cs" ? "Rok" : "Year", "n", oSeries[0].label, oSeries[1].label], rows: tr.map((r) => [r.label, r.n, pct100(r.good, 1), pct100(r.death, 1)]) },
      labels: { table: t().showTable, chart: t().showChart },
    });
    C.lineChart(fo.body, { height: 230, x: { type: "cat", cats: yrs }, y: { domain: [0, 70], ticks: [0, 20, 40, 60] }, series: oSeries, tipFmt: (v) => pct100(v, 0), endLabels: false, aria: "outcomes by year" });
    two.appendChild(outCard);
    root.appendChild(two);

    // small multiples
    root.appendChild(h("div", { class: "view-head", style: "margin-top:8px" }, h("div", null, h("h2", { style: "font-size:19px", text: t().trendTitle }), h("p", { text: t().trendSub }))));
    const sm = h("div", { class: "grid-3 block" });
    const single = (title, key, dom, fmt, ticksArr) => {
      const card = h("div", { class: "card card-pad" });
      const f = C.figure(card, { title, table: { columns: [lang === "cs" ? "Rok" : "Year", title], rows: tr.map((r) => [r.label, r[key] === null ? null : fmt(r[key])]) }, labels: { table: t().showTable, chart: t().showChart } });
      C.lineChart(f.body, { height: 150, x: { type: "cat", cats: yrs }, y: { domain: dom, ticks: ticksArr }, series: [{ label: title, color: "var(--c1)", area: true, endDot: true, points: tr.map((r, i) => ({ x: i, y: r[key] })) }], tipFmt: fmt, aria: title });
      sm.appendChild(card);
    };
    single(t().tPatients, "n", [0, 150], (v) => num(v), [0, 50, 100, 150]);
    single(t().tAge, "age", [55, 80], (v) => num(v), [55, 65, 75]);
    single(tx("Aged 80 or older (%)", "Ve věku 80 a více let (%)"), "age80", [0, 40], (v) => pct100(v), [0, 20, 40]);
    single(tx("IV thrombolysis (%)", "IV trombolýza (%)"), "ivt", [0, 100], (v) => pct100(v), [0, 50, 100]);
    single(tx("Transferred from another hospital (%)", "Překlad z jiné nemocnice (%)"), "transfer", [0, 100], (v) => pct100(v), [0, 50, 100]);
    single(t().tOtp, "otp", [0, 300], (v) => `${num(v)} min`, [0, 100, 200, 300]);
    single(tx("Procedure duration (median, min)", "Délka výkonu (medián, min)"), "proc", [0, 80], (v) => `${num(v)} min`, [0, 40, 80]);
    single(t().tTici, "tici2b3", [0, 100], (v) => pct100(v), [0, 50, 100]);
    single(tx("Symptomatic ICH (%)", "Symptomatické ICH (%)"), "sich", [0, 20], (v) => pct100(v), [0, 10, 20]);
    single(t().tGa, "ga", [0, 100], (v) => pct100(v), [0, 50, 100]);
    single(tx("Acute stenting (%)", "Akutní stenting (%)"), "stenting", [0, 100], (v) => pct100(v), [0, 50, 100]);
    // technique mix
    const techCard = h("div", { class: "card card-pad" });
    const techSeries = [
      { label: t().aspiration, color: "var(--c1)", points: tr.map((r, i) => ({ x: i, y: r.aspiration })) },
      { label: t().stentRetriever, color: "var(--c2)", points: tr.map((r, i) => ({ x: i, y: r.stent_retriever })) },
      { label: t().combined, color: "var(--c3)", points: tr.map((r, i) => ({ x: i, y: r.combined })) },
    ];
    const ftech = C.figure(techCard, { title: t().tTech, legend: techSeries.map((s) => ({ label: s.label, color: s.color, type: "line" })),
      table: { columns: [lang === "cs" ? "Rok" : "Year"].concat(techSeries.map((s) => s.label)), rows: tr.map((r) => [r.label, pct100(r.aspiration), pct100(r.stent_retriever), pct100(r.combined)]) },
      labels: { table: t().showTable, chart: t().showChart } });
    C.lineChart(ftech.body, { height: 150, x: { type: "cat", cats: yrs }, y: { domain: [0, 100], ticks: [0, 50, 100] }, series: techSeries, tipFmt: (v) => pct100(v), aria: t().tTech });
    sm.appendChild(techCard);
    root.appendChild(sm);

    root.appendChild(h("p", { class: "note block", html: tx(
      "Cohort flow, every data check, the variable map, missing data by era, 90-day outcome availability and Table 1 are under <b>Data and preprocessing</b> above.",
      "Tok pacientů, všechny kontroly dat, mapa proměnných, chybějící data podle období, dostupnost výsledku za 90 dní a tabulka 1 jsou v části <b>Data a předzpracování</b> výše.") }));
  }

  // ================================================================== validation
  const MODEL_NAMES = {
    "Logistic regression (splines)": { en: "Regression with splines", cs: "Regrese se splajny" },
    "Logistic regression": { en: "Linear regression", cs: "Lineární regrese" },
    "LightGBM": { en: "LightGBM", cs: "LightGBM" },
    "XGBoost": { en: "XGBoost", cs: "XGBoost" },
    "MLP": { en: "MLP (neural network)", cs: "MLP (neuronová síť)" },
    "THRIVE": { en: "THRIVE score", cs: "Skóre THRIVE" },
    "SPAN-100": { en: "SPAN-100 (cut-off 100)", cs: "SPAN-100 (hranice 100)" },
    "SPAN index": { en: "SPAN index (age + NIHSS)", cs: "Index SPAN (věk + NIHSS)" },
    "NIHSS 24 h": { en: "NIHSS at 24 h alone", cs: "Samotné NIHSS za 24 h" },
    "NIHSS day 7": { en: "NIHSS on day 7 alone", cs: "Samotné NIHSS 7. den" },
    "THRIVE-c": { en: "THRIVE-c predictors (refit)", cs: "Prediktory THRIVE-c (přepočteno)" },
    "HIAT2": { en: "HIAT2 score", cs: "Skóre HIAT2" },
    "PRE": { en: "PRE score", cs: "Skóre PRE" },
    "MR PREDICTS@24H": { en: "MR PREDICTS@24H predictors (refit)", cs: "Prediktory MR PREDICTS@24H (přepočteno)" },
    "Bedside score": { en: "FNO bedside score (7 items)", cs: "Bedside skóre FNO (7 položek)" },
    "Bedside variables (regression)": { en: "Bedside items as a regression", cs: "Položky bedside skóre jako regrese" },
  };
  const LEARNERS = ["Logistic regression (splines)", "Logistic regression", "LightGBM", "XGBoost", "MLP"];
  const MODEL_COLORS = ["var(--c1)", "var(--c2)", "var(--c3)", "var(--c4)", "var(--c5)", "var(--c6)"];
  const mname = (k) => L(MODEL_NAMES[k] || { en: k, cs: k });
  const mcolor = (k) => { const i = LEARNERS.indexOf(k); return i >= 0 ? MODEL_COLORS[i] : "var(--faint)"; };
  const signed = (v, d = 2) => (v === null || v === undefined ? "–" : `${v > 0 ? "+" : v < 0 ? "−" : ""}${num(Math.abs(v), d)}`);
  const ciTxt = (ci, d = 3) => (ci && ci.length === 2 ? `${num(ci[0], d)}–${num(ci[1], d)}` : "");
  const sciTxt = (ci, d = 3) => (ci && ci.length === 2 ? `${signed(ci[0], d)} to ${signed(ci[1], d)}` : "");

  const valSplit = () => V.splits[S.valSplit] || V.splits.primary;
  const valRes = (st, tg) => valSplit().stages[st].targets[tg];

  function renderValidation() {
    const root = $("#view-validation");
    root.replaceChildren();
    const sp = valSplit();
    const s0 = sp.stages.admission;
    const yrs = yearsOf(S.valSplit), dyrs = devYears(S.valSplit);
    root.appendChild(h("div", { class: "view-head" }, h("div", null, h("h2", { text: t().valTitle }),
      h("p", { text: t().valDesc(num(s0.n_dev), dyrs, num(s0.n_test), yrs, S.valSplit === "primary") }))));
    const seg = h("div", { class: "mini-seg", role: "group", "aria-label": t().outcome },
      TARGETS.map((k) => h("button", { type: "button", "aria-pressed": String(S.valTarget === k), onclick: () => { S.valTarget = k; renderValidation(); } }, k === "good_outcome" ? t().targetGood : t().targetDeath)));
    const stSel = h("select", { class: "sel", id: "val-stage" });
    for (const st of STAGES) stSel.appendChild(h("option", { value: st, selected: st === S.valStage ? true : null }, `${idx(st) + 1} · ${L(STAGE[st].title)}`));
    stSel.addEventListener("change", () => { S.valStage = stSel.value; renderValidation(); });
    const spSel = h("select", { class: "sel", id: "val-split" });
    for (const k of Object.keys(V.splits)) spSel.appendChild(h("option", { value: k, selected: k === S.valSplit ? true : null },
      k === "primary" ? tx(`${yearsOf(k)} (primary)`, `${yearsOf(k)} (hlavní)`) : tx(`${yearsOf(k)} (sensitivity)`, `${yearsOf(k)} (citlivostní)`)));
    spSel.addEventListener("change", () => { S.valSplit = spSel.value; renderValidation(); });
    root.appendChild(h("div", { class: "controls" }, h("span", { text: t().outcome }), seg, h("label", { for: "val-stage" }, t().stage), stSel,
      h("label", { for: "val-split" }, tx("Test years", "Testovací roky")), spSel));

    const tg = S.valTarget, st = S.valStage;
    const vt = valRes(st, tg);
    const models = vt.models;
    const learners = LEARNERS.filter((k) => models[k]);
    const scores = Object.keys(models).filter((k) => !LEARNERS.includes(k));
    const chosen = ((V.selection || {})[st] || {})[tg] ? V.selection[st][tg].chosen : REF;
    if (!S.valModel || !models[S.valModel]) S.valModel = chosen;

    root.appendChild(selectionCard(st, tg));

    // comparison table
    const cCard = h("div", { class: "card card-pad block" });
    cCard.appendChild(h("div", { class: "chart-head" }, h("div", null, h("div", { class: "chart-title", text: t().compTitle(L(STAGE[st].title)) }),
      h("div", { class: "chart-sub", text: tx(`${yrs}: n = ${num(vt.n_test)}, ${tg === "good_outcome" ? "mRS 0–2" : "deaths"} ${num(vt.events_test)} · development ${dyrs}: n = ${num(vt.n_dev)}, ${num(vt.events_dev)}`,
        `${yrs}: n = ${num(vt.n_test)}, ${tg === "good_outcome" ? "mRS 0–2" : "úmrtí"} ${num(vt.events_test)} · vývoj ${dyrs}: n = ${num(vt.n_dev)}, ${num(vt.events_dev)}`) }))));
    const nst = nestedFor(st, tg);
    const bestAuc = Math.max(...learners.map((k) => models[k].auroc));
    const headCols = [t().model, "AUROC (95% CI)", t().dAuc(mname(REF)), t().brier, t().citl, t().slope, "O/E"].concat(nst ? [tx("Nested CV AUROC", "Vnořená CV AUROC")] : []);
    const rowOf = (k) => {
      const r = models[k];
      const d = r.ci.delta_auroc_vs_ref;
      const nm = nst && nst.models[k];
      return h("tr", { class: LEARNERS.includes(k) ? (k === chosen ? "chosen" : "") : "score" },
        h("th", { scope: "row" }, h("span", { class: "dot", style: `background:${mcolor(k)}` }), mname(k), k === chosen ? h("span", { class: "tag", text: tx("chosen", "zvolen") }) : null),
        h("td", { class: r.auroc === bestAuc ? "best" : "" }, `${num(r.auroc, 3)} `, h("span", { class: "ci", text: ciTxt(r.ci.auroc) })),
        h("td", null, d ? `${signed(d.mean, 3)} ` : k === REF ? tx("reference", "reference") : "–", d ? h("span", { class: "ci", text: sciTxt(d.ci) }) : null),
        h("td", { text: num(r.brier, 3) }),
        h("td", null, `${signed(r.citl)} `, h("span", { class: "ci", text: sciTxt(r.ci.citl, 2) })),
        Math.abs(r.slope) > 5 ? h("td", { title: tx("Not estimable: one group has no outcome events (separation)", "Nelze odhadnout: jedna skupina nemá žádnou událost (separace)"), text: "n/e" })
          : h("td", null, `${num(r.slope, 2)} `, h("span", { class: "ci", text: ciTxt(r.ci.slope, 2) })),
        h("td", { text: num(r.oe, 2) }),
        ...(nst ? [h("td", { text: nm ? `${num(nm.auroc.mean, 3)} ± ${num(nm.auroc.sd, 3)}` : "–" })] : []));
    };
    const tbl = h("table", { class: "data comp" },
      h("thead", null, h("tr", null, headCols.map((x) => h("th", { scope: "col", text: x })))),
      h("tbody", null, learners.map(rowOf), scores.length ? h("tr", { class: "sep" }, h("th", { colspan: headCols.length, text: tx("Published and bedside scores", "Publikovaná a bedside skóre") })) : null, scores.map(rowOf)));
    cCard.appendChild(h("div", { class: "table-scroll" }, tbl));
    cCard.appendChild(h("p", { class: "note", text: t().compNote(mname(REF)) }));
    const tun = tuningBlock(vt.settings, yrs);
    if (tun) cCard.appendChild(tun);
    root.appendChild(cCard);

    // AUROC across stages + ROC
    const row1 = h("div", { class: "grid-2 block" });
    const aCard = h("div", { class: "card card-pad" });
    const aSeries = learners.map((k, i) => ({
      label: mname(k), color: mcolor(k), dots: true, offset: (i - (learners.length - 1) / 2) * 5,
      points: STAGES.map((s2, j) => {
        const r = sp.stages[s2].targets[tg].models[k];
        return { x: j, y: r.auroc, lo: r.ci.auroc[0], hi: r.ci.auroc[1] };
      }),
    }));
    const fa = C.figure(aCard, {
      title: t().aurocTitle, sub: t().aurocSub(yrs), legend: aSeries.map((s2) => ({ label: s2.label, color: s2.color, type: "line" })),
      table: { columns: [t().model].concat(STAGES.map(stageLabel)), rows: learners.map((k) => [mname(k)].concat(STAGES.map((s2) => { const r = sp.stages[s2].targets[tg].models[k]; return `${num(r.auroc, 3)} (${ciTxt(r.ci.auroc)})`; }))) },
      labels: { table: t().showTable, chart: t().showChart },
    });
    C.lineChart(fa.body, { height: 240, x: { type: "cat", cats: STAGES.map(stageLabel) }, y: { domain: [0.5, 1], ticks: [0.5, 0.6, 0.7, 0.8, 0.9, 1], fmt: (v) => num(v, 1) }, series: aSeries,
      tipFmt: (v, p) => `${num(v, 3)} (${num(p.lo, 2)}–${num(p.hi, 2)})`, aria: t().aurocTitle });
    row1.appendChild(aCard);
    const rCard = h("div", { class: "card card-pad" });
    const rKeys = learners.concat(scores);
    const rSeries = rKeys.map((k) => ({ label: mname(k), color: mcolor(k), width: k === chosen ? 2.6 : LEARNERS.includes(k) ? 1.5 : 1.0, points: vt.curves.roc[k].map((p) => ({ x: p.fpr, y: p.tpr })) }));
    const rLegend = rSeries.filter((s2) => learners.map(mname).includes(s2.label)).map((s2) => ({ label: s2.label, color: s2.color, type: "line" }));
    if (scores.length) rLegend.push({ label: tx("Scores (grey)", "Skóre (šedě)"), color: "var(--faint)", type: "line" });
    const fr = C.figure(rCard, { title: t().rocTitle, sub: `${L(STAGE[st].title)} · ${yrs}`, legend: rLegend,
      table: { columns: [t().fpr].concat(rKeys.map(mname)), rows: vt.curves.roc[REF].filter((_, i) => i % 5 === 0).map((p, j) => [num(p.fpr, 2)].concat(rKeys.map((k) => num(vt.curves.roc[k][j * 5].tpr, 2)))) },
      labels: { table: t().showTable, chart: t().showChart } });
    C.lineChart(fr.body, { height: 240, x: { type: "lin", domain: [0, 1], fmt: (v) => num(v, 1), title: t().fpr }, y: { domain: [0, 1], fmt: (v) => num(v, 1), title: t().tpr, ticks: [0, 0.2, 0.4, 0.6, 0.8, 1] }, series: rSeries.slice().reverse(), diagonal: true,
      tipFmt: (v) => num(v, 2), tipHead: (xv) => `${t().fpr} ${num(xv, 2)}`, aria: t().rocTitle });
    row1.appendChild(rCard);
    root.appendChild(row1);

    // per-model cards
    const mSel = h("select", { class: "sel", id: "val-model" });
    for (const k of learners.concat(scores)) mSel.appendChild(h("option", { value: k, selected: k === S.valModel ? true : null }, mname(k)));
    mSel.addEventListener("change", () => { S.valModel = mSel.value; renderValidation(); });
    root.appendChild(h("div", { class: "view-head", style: "margin-top:6px" }, h("div", null,
      h("h2", { style: "font-size:19px", text: tx("One model in detail", "Jeden model podrobně") }),
      h("p", { text: tx("Calibration, classification at thresholds, subgroups and what the model relies on, for the model chosen here.",
        "Kalibrace, klasifikace při prazích, podskupiny a na čem model stojí, pro zde zvolený model.") }),
      h("div", { class: "controls", style: "margin-top:8px" }, h("label", { for: "val-model" }, t().model), mSel))));
    const vm = S.valModel;
    const row2 = h("div", { class: "grid-2 block" });
    row2.appendChild(calibrationCard(vt, vm, yrs));
    const up = updatingCard(vt, learners, yrs);
    if (up) row2.appendChild(up);
    root.appendChild(row2);
    const row3 = h("div", { class: "grid-2 block" });
    row3.appendChild(dcaCard(vt, learners, scores, chosen, yrs));
    const thr = thresholdCard(vt, tg, vm, yrs);
    if (thr) row3.appendChild(thr);
    root.appendChild(row3);
    if (vt.subgroups && vt.subgroups[vm]) { const sg = subgroupCard(vt, vm, yrs); sg.classList.add("block"); root.appendChild(sg); }
    const row4 = h("div", { class: "grid-2 block" });
    const ic = importanceCard(st, tg, vm);
    if (ic) row4.appendChild(ic);
    const sc = stabilityCard(st, tg);
    if (sc) row4.appendChild(sc);
    if (row4.childNodes.length) root.appendChild(row4);
    const oc = ordinalCard(st);
    if (oc) { oc.classList.add("block"); root.appendChild(oc); }
    const nc = nestedCard();
    if (nc) root.appendChild(nc);
    root.appendChild(driftCard(st, tg, vt, chosen, yrs, dyrs));
  }

  function nestedFor(st, tg) {
    const N = V.nested;
    if (!N || !N.stages || !N.stages[st] || !N.stages[st][tg]) return null;
    return N.stages[st][tg];
  }

  // The pre-specified model choice and how each candidate fared
  function selectionCard(st, tg) {
    const sel = ((V.selection || {})[st] || {})[tg];
    const card = h("div", { class: "card card-pad block selection" });
    if (!sel) return card;
    card.appendChild(h("div", { class: "chart-head" }, h("div", null,
      h("div", { class: "chart-title", text: tx("Which model the calculator uses, and why", "Který model kalkulačka používá a proč") }),
      h("div", { class: "chart-sub", text: tx(`Rule fixed before testing (as in the 2026 analysis); temporal test ${yearsOf("primary")}${sel.nested_cv ? " and nested cross-validation on all years" : ""}`,
        `Pravidlo stanovené před testováním (jako v analýze z roku 2026); časový test ${yearsOf("primary")}${sel.nested_cv ? " a vnořená křížová validace na všech letech" : ""}`) }))));
    card.appendChild(h("p", { class: "callout", html: tx(
      `<b>${mname(sel.chosen)}</b>${sel.chosen === sel.reference ? " (the reference)" : ""} for ${stageLabel(st).toLowerCase()}, ${tg === "good_outcome" ? "mRS 0–2" : "death"}. ${mname(sel.reference)} is the reference; another model replaces it only if its paired 95% CI of the AUROC difference on the temporal test lies above zero, or in nested cross-validation it gains at least 0.01 AUROC on average and is better in at least 80% of the folds, and in both cases its calibration-in-the-large is no more than 0.05 worse.`,
      `<b>${mname(sel.chosen)}</b>${sel.chosen === sel.reference ? " (reference)" : ""} pro fázi ${stageLabel(st).toLowerCase()}, ${tg === "good_outcome" ? "mRS 0–2" : "úmrtí"}. ${mname(sel.reference)} je referencí; jiný model ji nahradí jen tehdy, když párový 95% IS rozdílu AUROC na časovém testu leží nad nulou, nebo když ve vnořené křížové validaci získá v průměru alespoň 0,01 AUROC a je lepší v alespoň 80 % foldů, a v obou případech jeho kalibrace v celku není o více než 0,05 horší.`) }));
    const rows = Object.entries(sel.candidates).map(([k, c]) => h("tr", { class: c.qualifies ? "chosen" : "" },
      h("th", { scope: "row" }, h("span", { class: "dot", style: `background:${mcolor(k)}` }), mname(k)),
      h("td", null, c.delta_test ? `${signed(c.delta_test.mean, 3)} ` : "–", c.delta_test ? h("span", { class: "ci", text: sciTxt(c.delta_test.ci) }) : null),
      h("td", { text: c.test_rule ? tx("yes", "ano") : tx("no", "ne") }),
      h("td", { text: c.cv_delta_mean !== undefined ? `${signed(c.cv_delta_mean, 3)} · ${num(c.cv_share_better * 100)}%` : "–" }),
      h("td", { text: c.cv_rule === undefined ? "–" : c.cv_rule ? tx("yes", "ano") : tx("no", "ne") }),
      h("td", { text: c.calibration_ok ? tx("yes", "ano") : tx("no", "ne") }),
      h("td", { class: c.qualifies ? "best" : "", text: c.qualifies ? tx("replaces", "nahrazuje") : tx("no", "ne") })));
    card.appendChild(h("div", { class: "table-scroll" }, h("table", { class: "data compact" },
      h("thead", null, h("tr", null, [tx("Candidate", "Kandidát"), tx("ΔAUROC on test (95% CI)", "ΔAUROC na testu (95% IS)"), tx("CI above 0", "IS nad 0"),
        tx("Nested CV: ΔAUROC · folds better", "Vnořená CV: ΔAUROC · lepší foldy"), tx("CV rule met", "Pravidlo CV"), tx("Calibration not worse", "Kalibrace nehorší"), tx("Verdict", "Výsledek")].map((x) => h("th", { scope: "col", text: x })))),
      h("tbody", null, rows))));
    if (!sel.nested_cv) card.appendChild(h("p", { class: "note", text: tx("Nested cross-validation was run for the admission stage and mRS 0–2; here the rule uses the temporal test, and calibration is compared on the test years.",
      "Vnořená křížová validace proběhla pro fázi přijetí a mRS 0–2; zde pravidlo vychází z časového testu a kalibrace se srovnává na testovacích letech.") }));
    return card;
  }

  function calibrationCard(vt, k, yrs) {
    const card = h("div", { class: "card card-pad" });
    const cal = vt.curves.calibration[k] || [];
    const r = vt.models[k];
    const fk = C.figure(card, { title: t().calTitle(mname(k)), sub: t().calSub(yrs),
      table: { columns: [t().predicted, t().observed, "n"], rows: cal.map((b) => [pct(b.pred, 1), pct(b.obs, 1), b.n]) },
      labels: { table: t().showTable, chart: t().showChart } });
    const calMax = Math.min(1, Math.ceil(Math.max(0.1, ...cal.map((b) => Math.max(b.pred, b.obs))) * 10 + 0.5) / 10);
    C.lineChart(fk.body, { height: 250, x: { type: "lin", domain: [0, calMax], fmt: (v) => pct(v), title: t().predicted }, y: { domain: [0, calMax], fmt: (v) => pct(v), title: t().observed }, diagonal: true,
      series: [{ label: t().observed, color: mcolor(k), dots: true, points: cal.map((b) => ({ x: b.pred, y: b.obs, n: b.n })) }],
      tipFmt: (v, p) => `${pct(v, 1)} · n ${p.n}`, tipHead: (xv) => `${t().predicted} ${pct(xv, 1)}`, aria: t().calTitle(mname(k)) });
    card.appendChild(h("p", { class: "note", text: tx(
      `Calibration-in-the-large ${signed(r.citl)} (95% CI ${sciTxt(r.ci.citl, 2)}), slope ${num(r.slope, 2)} (${ciTxt(r.ci.slope, 2)}), ICI ${num(r.ici, 3)}, observed/expected ${num(r.oe, 2)}. A negative intercept means the model expected more good outcomes than happened. These are the development model's estimates; the calculator's are shifted to ${RECENT}.`,
      `Kalibrace v celku ${signed(r.citl)} (95% IS ${sciTxt(r.ci.citl, 2)}), sklon ${num(r.slope, 2)} (${ciTxt(r.ci.slope, 2)}), ICI ${num(r.ici, 3)}, pozorováno/očekáváno ${num(r.oe, 2)}. Záporný intercept znamená, že model čekal více dobrých výsledků, než nastalo. Jde o odhady vývojového modelu; kalkulačka je posouvá na úroveň ${RECENT}.`) }));
    return card;
  }

  // Updating: recalibrated on the first test year, evaluated on the later years (2026 analysis's table)
  function updatingCard(vt, learners, yrs) {
    const rows = learners.filter((k) => vt.updating && vt.updating[k]);
    if (!rows.length) return null;
    const u0 = vt.updating[rows[0]];
    const card = h("div", { class: "card card-pad" });
    card.appendChild(h("div", { class: "chart-head" }, h("div", null,
      h("div", { class: "chart-title", text: tx("Recalibration on newer patients", "Přepočet kalibrace na novějších pacientech") }),
      h("div", { class: "chart-sub", text: tx(`Recalibrated on ${u0.recal_year} (n ${num(u0.n_recal)}), evaluated on the later test years (n ${num(u0.n_eval)}). AUROC does not change; calibration does.`,
        `Přepočteno na roce ${u0.recal_year} (n ${num(u0.n_recal)}), hodnoceno na pozdějších testovacích letech (n ${num(u0.n_eval)}). AUROC se nemění, mění se kalibrace.`) }))));
    const V3 = [["original", tx("Original", "Původní")], ["intercept", tx("Intercept updated", "Upravený intercept")], ["logistic", tx("Intercept and slope", "Intercept a sklon")]];
    const body = [];
    for (const k of rows) for (const [v, lbl] of V3) {
      const r = vt.updating[k][v];
      body.push(h("tr", null, h("th", { scope: "row" }, v === "original" ? mname(k) : ""), h("td", { text: lbl }), h("td", { text: num(r.brier, 3) }), h("td", { text: signed(r.citl) }), h("td", { text: num(r.slope, 2) })));
    }
    card.appendChild(h("div", { class: "table-scroll" }, h("table", { class: "data compact" },
      h("thead", null, h("tr", null, [t().model, tx("Variant", "Varianta"), t().brier, t().citl, t().slope].map((x) => h("th", { scope: "col", text: x })))),
      h("tbody", null, body))));
    card.appendChild(h("p", { class: "note", text: tx(
      "Outcome rates swing from year to year, so a single year over- or under-corrects. The published calculator re-estimates its intercept on the pooled last three years instead.",
      "Podíl dobrých výsledků mezi roky kolísá, takže jediný rok koriguje příliš nebo málo. Zveřejněná kalkulačka proto odhaduje intercept ze sloučených posledních tří let.") }));
    return card;
  }

  function dcaCard(vt, learners, scores, chosen, yrs) {
    const dcCard = h("div", { class: "card card-pad" });
    const all = vt.curves.dca_all;
    const keys = learners.concat(scores);
    const dm = vt.curves.dca;
    const modelSeries = keys.map((k) => ({ label: mname(k), color: mcolor(k), width: k === chosen ? 2.6 : LEARNERS.includes(k) ? 1.6 : 1.0, points: dm[k].map((r) => ({ x: r.t, y: r.nb })) }));
    const dSeries = modelSeries.slice().reverse().concat([{ label: t().treatAll, color: "var(--ink-2)", width: 1.5, dash: true, points: all.map((r) => ({ x: r.t, y: r.nb })) }]);
    const lo = Math.min(-0.05, ...all.map((r) => Math.max(r.nb, -0.1)));
    const hi = Math.max(...keys.flatMap((k) => dm[k].map((r) => r.nb)), ...all.map((r) => r.nb)) + 0.03;
    const dLegend = modelSeries.filter((s2) => learners.map(mname).includes(s2.label)).map((s2) => ({ label: s2.label, color: s2.color, type: "line" }));
    if (scores.length) dLegend.push({ label: tx("Scores (grey)", "Skóre (šedě)"), color: "var(--faint)", type: "line" });
    dLegend.push({ label: t().treatAll, color: "var(--ink-2)", type: "line" }, { label: t().treatNone, color: "var(--line-2)", type: "line" });
    const fd = C.figure(dcCard, { title: t().dcaTitle, sub: `${t().dcaSub} · ${yrs}`, legend: dLegend,
      table: { columns: [t().threshold].concat(keys.map(mname), [t().treatAll, t().treatNone]),
        rows: all.map((r, i) => [pct(r.t)].concat(keys.map((k) => num(dm[k][i].nb, 3)), [num(r.nb, 3), "0"])) },
      labels: { table: t().showTable, chart: t().showChart } });
    C.lineChart(fd.body, { height: 240, x: { type: "lin", domain: [0.05, 0.8], fmt: (v) => pct(v), title: t().threshold, ticks: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] }, y: { domain: [lo, hi], fmt: (v) => num(v, 2), title: t().netBenefit },
      series: dSeries.map((s2) => ({ ...s2, points: s2.points.map((p) => ({ x: p.x, y: Math.max(p.y, lo) })) })), zeroLine: true,
      tipFmt: (v) => num(v, 3), tipHead: (xv) => `${t().threshold} ${pct(xv)}`, aria: t().dcaTitle });
    const useful = dm[chosen] ? dm[chosen].filter((r, i) => r.nb > Math.max(all[i].nb, 0) + 1e-4).map((r) => r.t) : [];
    if (useful.length) {
      const tmin = Math.min(...useful), tmax = Math.max(...useful);
      const gaps = all.map((r) => r.t).filter((x) => x >= tmin - 1e-9 && x <= tmax + 1e-9).length > useful.length;
      dcCard.appendChild(h("p", { class: "note", text: tx(
        `${mname(chosen)} adds net benefit over treating everyone and no one at risk thresholds from ${pct(tmin)} to ${pct(tmax)}${gaps ? " (with gaps)" : ""}. A threshold is the estimate at which a clinician would act; the curve shows for which thresholds the model would help that decision.`,
        `${mname(chosen)} přináší čistý přínos oproti léčbě všech i nikoho při prazích rizika od ${pct(tmin)} do ${pct(tmax)}${gaps ? " (s přerušeními)" : ""}. Práh je odhad, při kterém by lékař jednal; křivka ukazuje, pro které prahy by model rozhodnutí pomohl.`) }));
    }
    return dcCard;
  }

  // Classification at fixed risk thresholds (development model, test years)
  function thresholdCard(vt, tg, k, yrs) {
    const rows = (vt.thresholds || {})[k];
    if (!rows || !rows.length) return null;
    const good = tg === "good_outcome";
    const card = h("div", { class: "card card-pad" });
    card.appendChild(h("div", { class: "chart-head" }, h("div", null,
      h("div", { class: "chart-title", text: tx("Sensitivity, specificity and predictive values", "Senzitivita, specificita a prediktivní hodnoty") }),
      h("div", { class: "chart-sub", text: good
        ? tx(`${mname(k)} · “expected independent” when the estimate of mRS 0–2 reaches the threshold · ${yrs}`, `${mname(k)} · „očekávaně soběstačný“, když odhad mRS 0–2 dosáhne prahu · ${yrs}`)
        : tx(`${mname(k)} · “high risk of death” when the estimate reaches the threshold · ${yrs}`, `${mname(k)} · „vysoké riziko úmrtí“, když odhad dosáhne prahu · ${yrs}`) }))));
    const head = [tx("Threshold", "Práh"), tx("At or above", "Na prahu či výše"), tx("Sensitivity", "Senzitivita"), tx("Specificity", "Specificita"), "PPV", "NPV", tx("Accuracy", "Přesnost")];
    const tbl = h("table", { class: "data compact" },
      h("thead", null, h("tr", null, head.map((x) => h("th", { scope: "col", text: x })))),
      h("tbody", null, rows.map((r) => h("tr", null, h("th", { scope: "row", text: pct(r.t) }),
        ...(r.suppressed ? [h("td", { colspan: 6, class: "muted", text: tx(`not shown: a cell of the 2×2 table has fewer than ${E.min_cell} patients`, `nezobrazeno: buňka tabulky 2×2 má méně než ${E.min_cell} pacientů`) })]
          : [pct(r.pos), pct(r.sens), pct(r.spec), pct(r.ppv), pct(r.npv), pct(r.acc)].map((x) => h("td", { text: x })))))));
    card.appendChild(h("div", { class: "table-scroll" }, tbl));
    const citl = vt.models[k].citl;
    card.appendChild(h("p", { class: "note", text: tx(
      `PPV: share at or above the threshold who ${good ? "reached mRS 0–2" : "died"}; NPV: share below it who ${good ? "did not" : "survived"}. Development-model estimates applied unchanged (calibration intercept ${signed(citl)}); the calculator's are shifted to ${RECENT}, so these cut-offs illustrate the trade-off and are not recommended decision thresholds.`,
      `PPV: podíl na prahu či výše, kteří ${good ? "dosáhli mRS 0–2" : "zemřeli"}; NPV: podíl pod prahem, kteří ${good ? "ho nedosáhli" : "přežili"}. Odhady vývojového modelu beze změny (kalibrační intercept ${signed(citl)}); kalkulačka je posouvá na úroveň ${RECENT}, proto tyto hranice ilustrují kompromis a nejsou doporučenými rozhodovacími prahy.`) }));
    return card;
  }

  // How each learner was tuned: the same budget for all, and the settings chosen for this stage and outcome
  const sig2 = (v) => {
    if (v === null || v === undefined) return "–";
    const a = Math.abs(v);
    if (a >= 100) return num(v, 0);
    if (a >= 1) return num(v, a >= 10 ? 0 : 1);
    return num(v, Math.min(6, Math.max(0, 1 - Math.floor(Math.log10(a)))));
  };
  function tuningBlock(st, yrs) {
    if (!st) return null;
    const trials = (st.LightGBM && st.LightGBM.trials) || 50;
    const SPACE = {
      LightGBM: tx("learning rate 0.01–0.08, 3–15 leaves, depth 2–5, 20–80 patients per leaf, column and row sampling, L2 penalty; monotone constraints; trees by early stopping",
        "rychlost učení 0,01–0,08, 3–15 listů, hloubka 2–5, 20–80 pacientů v listu, vzorkování sloupců a řádků, penalizace L2; monotónní omezení; počet stromů časným zastavením"),
      XGBoost: tx("learning rate 0.01–0.08, depth 2–5, minimum child weight 5–20 (≥ 20 patients per leaf), row and column sampling, L2 penalty, minimum split gain; trees by early stopping",
        "rychlost učení 0,01–0,08, hloubka 2–5, minimální váha listu 5–20 (≥ 20 pacientů v listu), vzorkování řádků a sloupců, penalizace L2, minimální zisk dělení; počet stromů časným zastavením"),
      MLP: tx("1–3 layers of 8–128 neurons, ReLU or tanh, weight decay, learning rate, batch size; the best setting averaged over 5 random starts",
        "1–3 vrstvy po 8–128 neuronech, ReLU nebo tanh, útlum vah, rychlost učení, velikost dávky; nejlepší nastavení zprůměrované přes 5 náhodných startů"),
      "Logistic regression": tx("penalty strength, 50 values", "síla penalizace, 50 hodnot"),
      "Logistic regression (splines)": tx("3, 4 or 5 knots × penalty strength (50 values each)", "3, 4 nebo 5 uzlů × síla penalizace (po 50 hodnotách)"),
    };
    const chosen = (k, s) => {
      if (!s) return "–";
      if (k === "LightGBM") return [`${tx("rate", "rychlost")} ${sig2(s.learning_rate)}`, `${s.num_leaves} ${tx("leaves", "listů")}`, `${tx("depth", "hloubka")} ${s.max_depth}`,
        `≥${s.min_child_samples} ${tx("per leaf", "v listu")}`, `${tx("columns", "sloupce")} ${sig2(s.feature_fraction)}`, `${tx("rows", "řádky")} ${sig2(s.bagging_fraction)}`,
        `L2 ${sig2(s.lambda_l2)}`, `${s.trees} ${tx("trees", "stromů")}`].join(" · ");
      if (k === "XGBoost") return [`${tx("rate", "rychlost")} ${sig2(s.learning_rate)}`, `${tx("depth", "hloubka")} ${s.max_depth}`, `${tx("child weight", "váha listu")} ${sig2(s.min_child_weight)}`,
        `${tx("rows", "řádky")} ${sig2(s.subsample)}`, `${tx("columns", "sloupce")} ${sig2(s.colsample_bytree)}`, `L2 ${sig2(s.lambda)}`, `gamma ${sig2(s.gamma)}`, `${s.trees} ${tx("trees", "stromů")}`].join(" · ");
      if (k === "MLP") return [`${tx("layers", "vrstvy")} ${s.hidden_layers.join("→")}`, s.activation, `${tx("weight decay", "útlum vah")} ${sig2(s.alpha)}`,
        `${tx("rate", "rychlost")} ${sig2(s.learning_rate)}`, `${tx("batch", "dávka")} ${s.batch_size}`, `${s.ensemble} ${tx("starts", "startů")}`].join(" · ");
      if (k === "Logistic regression (splines)") return `${s.knots} ${tx("knots", "uzly")} · C ${sig2(s.C)}`;
      return `C ${sig2(s.C)}`;
    };
    const keys = LEARNERS.filter((k) => st[k]);
    const box = h("details", { class: "tuning" });
    box.appendChild(h("summary", null, h("span", { text: tx("How each model was tuned", "Jak byly modely laděny") }),
      h("span", { class: "muted", text: tx(`${trials} candidate settings each, same folds`, `po ${trials} kandidátních nastaveních, stejné foldy`) })));
    box.appendChild(h("p", { class: "note", text: tx(
      `Every learner had the same budget: ${trials} candidate settings (Bayesian search for the tree models and the network, a grid of penalty strengths for the regressions), each scored by 5-fold cross-validated log-loss on the development patients, with the same folds for every model. The ${yrs} patients were never used to choose anything. Every learner gets the same inputs; missing values are filled with the development median plus a missing-value indicator for the regressions and the network, and follow the trees' learned default branch in LightGBM and XGBoost.`,
      `Každý model měl stejný rozpočet: ${trials} kandidátních nastavení (bayesovské hledání u stromových modelů a sítě, mřížka sil penalizace u regresí), každé hodnocené 5násobnou křížovou validací log-loss na vývojových pacientech se stejnými foldy pro všechny modely. Pacienti ${yrs} nebyli k žádné volbě použiti. Všechny modely mají stejné vstupy; chybějící hodnoty jsou u regresí a sítě doplněny mediánem z vývoje s indikátorem chybění a ve stromech LightGBM a XGBoost jdou naučenou výchozí větví.`) }));
    const tbl = h("table", { class: "data compact tuning-table" },
      h("thead", null, h("tr", null, [t().model, tx("Searched", "Prohledáno"), tx("Chosen for this stage and outcome", "Zvoleno pro tuto fázi a výsledek"), tx("CV log-loss", "CV log-loss")].map((x) => h("th", { scope: "col", text: x })))),
      h("tbody", null, keys.map((k) => h("tr", null, h("th", { scope: "row", text: mname(k) }),
        h("td", { class: "txt", "data-l": tx("Searched", "Prohledáno"), text: SPACE[k] || "" }),
        h("td", { class: "txt", "data-l": tx("Chosen", "Zvoleno"), text: chosen(k, st[k]) }),
        h("td", { "data-l": "CV log-loss", text: num(st[k].cv_logloss, 3) })))));
    box.appendChild(h("div", { class: "table-scroll" }, tbl));
    return box;
  }

  function importanceCard(st, tg, k) {
    const dep = (((V.deployed || {})[st] || {})[tg] || {}).learners || {};
    const ex = dep[k] && dep[k].explain;
    if (!ex) return null;
    const card = h("div", { class: "card card-pad" });
    const imp = Object.entries(ex.importance).slice(0, 14);
    const fi = C.figure(card, { title: t().impTitle(mname(k)), sub: `${t().impSub} · ${L(STAGE[st].title)}`,
      table: { columns: [t().variable, tx("mean |contribution|", "průměrný |příspěvek|")], rows: imp.map(([f, v]) => [FEAT[f] ? L(FEAT[f].label) : f, num(v, 3)]) },
      labels: { table: t().showTable, chart: t().showChart } });
    C.hbars(fi.body, { rows: imp.map(([f, v]) => ({ label: FEAT[f] ? L(FEAT[f].label) : f, value: v, display: num(v, 2) })), color: mcolor(k), valueLabel: tx("mean |contribution|", "průměrný |příspěvek|"), aria: t().impTitle(mname(k)) });
    return card;
  }

  function ordinalCard(st) {
    const ov = (X.ordinal_validation || {})[st];
    if (!ov) return null;
    const card = h("div", { class: "card card-pad" });
    const legend = [0, 1, 2, 3, 4, 5, 6].map((m) => ({ label: `mRS ${m}`, color: `var(--mrs${m})` }));
    const rows = [
      { label: tx(`Observed ${yearsOf()}`, `Pozorováno ${yearsOf()}`), n: ov.n, segs: ov.observed.map((v) => v * 100) },
      { label: tx("Predicted", "Predikováno"), n: ov.n, segs: ov.predicted.map((v) => v * 100) },
    ];
    const f = C.figure(card, { title: tx("Full mRS distribution (ordinal)", "Celé rozložení mRS (ordinální)"),
      sub: tx(`Chosen development models applied to ${yearsOf()} · ordinal c-index ${num(ov.c_ordinal, 3)}`, `Zvolené vývojové modely použité na ${yearsOf()} · ordinální c-index ${num(ov.c_ordinal, 3)}`),
      legend, table: { columns: ["", "n"].concat(legend.map((l) => l.label)), rows: rows.map((r) => [r.label, r.n].concat(r.segs.map((v) => pct100(v, 1)))) },
      labels: { table: t().showTable, chart: t().showChart } });
    C.grotta(f.body, { rows, colors: MRS_COLORS, segLabels: legend.map((l) => l.label), fmt: (v) => pct100(v, 1), nLabel: "n", aria: "mRS distribution" });
    card.appendChild(h("p", { class: "note", text: tx(
      `AUROC for each cut point: ${ov.cum_auroc.map((a, k) => `mRS ≤${k} ${a === null ? "–" : num(a, 2)}`).join(", ")}. Mean utility-weighted mRS observed ${num(ov.uw_obs, 2)} vs predicted ${num(ov.uw_pred, 2)} before recalibration.`,
      `AUROC pro jednotlivé hranice: ${ov.cum_auroc.map((a, k) => `mRS ≤${k} ${a === null ? "–" : num(a, 2)}`).join(", ")}. Průměrné užitkově vážené mRS pozorováno ${num(ov.uw_obs, 2)}, predikováno ${num(ov.uw_pred, 2)} před rekalibrací.`) }));
    return card;
  }

  function stabilityCard(st, tg) {
    const bins = ((X.uncertainty || {})[st] || {})[tg];
    const stab = ((X.stability || {})[st] || {})[tg];
    if (!bins || !stab) return null;
    const card = h("div", { class: "card card-pad" });
    const f = C.figure(card, { title: tx("Prediction stability", "Stabilita predikce"),
      sub: tx(`${mname(stab.model)} (chosen) · ${stab.n_boot} bootstrap refits · typical 95% range by predicted risk`, `${mname(stab.model)} (zvolený) · ${stab.n_boot} přetrénování na bootstrap vzorcích · typické 95% rozpětí podle rizika`),
      table: { columns: [tx("Predicted", "Predikováno"), tx("95% range", "95% rozpětí"), "n"], rows: bins.map((b) => [pct(b.p), `${pct(b.p_lo)}–${pct(b.p_hi)}`, b.n]) },
      labels: { table: t().showTable, chart: t().showChart } });
    C.lineChart(f.body, { height: 230, x: { type: "lin", domain: [0, 1], fmt: (v) => pct(v), title: tx("Estimate", "Odhad") }, y: { domain: [0, 1], fmt: (v) => pct(v), title: tx("Range across refits", "Rozpětí přes přetrénování") }, diagonal: true,
      series: [{ label: tx("95% range", "95% rozpětí"), color: "var(--c1)", dots: true, noLine: true, points: bins.map((b) => ({ x: b.p, y: b.p, lo: b.p_lo, hi: b.p_hi, n: b.n })) }],
      tipFmt: (v, p) => `${pct(p.lo)}–${pct(p.hi)} · n ${p.n}`, tipHead: (xv) => `${tx("Estimate", "Odhad")} ${pct(xv)}`, aria: "stability" });
    card.appendChild(h("p", { class: "note", text: tx(
      `Mean absolute difference between a refitted and the published estimate: ${num(stab.mape * 100, 1)} percentage points; ${num((stab.share_wide || 0) * 100)}% of patients have a 95% range wider than 20 points (Riley & Collins 2023). The calculator shows this range under the headline.`,
      `Průměrný absolutní rozdíl mezi odhadem přetrénovaného a publikovaného modelu: ${num(stab.mape * 100, 1)} procentního bodu; ${num((stab.share_wide || 0) * 100)} % pacientů má 95% rozpětí širší než 20 bodů (Riley & Collins 2023). Kalkulačka toto rozpětí ukazuje pod hlavním odhadem.`) }));
    return card;
  }

  const SUBDIM = {
    circulation: ["Circulation", "Povodí"], occlusion: ["Occlusion", "Okluze"], age: ["Age", "Věk"], sex: ["Sex", "Pohlaví"], ivt: ["IV thrombolysis", "IV trombolýza"],
    route: ["Admission route", "Způsob přijetí"], nihss: ["NIHSS on admission", "NIHSS při přijetí"], otp: ["Onset to puncture", "Od vzniku do punkce"],
    premrs: ["Pre-stroke mRS", "mRS před CMP"], procedure: ["Procedure", "Výkon"],
  };
  const SUBGRP = {
    anterior: ["anterior", "přední"], posterior: ["posterior", "zadní"], M1: ["MCA M1", "ACM M1"], ICA: ["ICA / T", "ACI / T"], cervical_ICA: ["cervical ICA", "krční ACI"],
    distal: ["M2–M3 / ACA", "M2–M3 / ACA"], "<80": ["under 80", "pod 80"], "80+": ["80 or older", "80 a více"], men: ["men", "muži"], women: ["women", "ženy"],
    ivt: ["with IVT", "s IVT"], no_ivt: ["without IVT", "bez IVT"], direct: ["direct", "přímo"], transfer: ["transferred", "překlad"],
    "<10": ["below 10", "pod 10"], "10-19": ["10–19", "10–19"], "20+": ["20 or more", "20 a více"], "<=6h": ["≤ 6 h", "≤ 6 h"], ">6h": ["> 6 h", "> 6 h"],
    0: ["0", "0"], "1+": ["1 or more", "1 a více"], mt: ["thrombectomy", "trombektomie"], other: ["other treatment", "jiná léčba"], diagnostic: ["diagnostic only", "pouze diagnostika"],
  };
  function subgroupCard(vt, k, yrs) {
    const card = h("div", { class: "card card-pad" });
    const rows = vt.subgroups[k] || [];
    card.appendChild(h("div", { class: "chart-head" }, h("div", null,
      h("div", { class: "chart-title", text: t().subTitle }), h("div", { class: "chart-sub", text: t().subSub(mname(k), yrs) }))));
    const tbl = h("table", { class: "data compact" },
      h("thead", null, h("tr", null, [t().subGroup, "n", t().observed, t().predicted, "O/E", "AUROC", t().citl].map((x) => h("th", { scope: "col", text: x })))),
      h("tbody", null, rows.map((r) => h("tr", null,
        h("th", { scope: "row", text: `${tx(...(SUBDIM[r.dim] || [r.dim, r.dim]))}: ${tx(...(SUBGRP[r.group] || [r.group, r.group]))}` }),
        h("td", { text: r.suppressed ? `<${E.min_cell}` : num(r.n) }),
        h("td", { text: r.suppressed ? "–" : pct(r.observed) }),
        h("td", { text: r.suppressed ? "–" : pct(r.predicted) }),
        h("td", { text: r.suppressed ? "–" : num(r.oe, 2) }),
        h("td", null, r.auroc === undefined ? "–" : `${num(r.auroc, 3)} `, r.auroc_ci ? h("span", { class: "ci", text: ciTxt(r.auroc_ci, 2) }) : null),
        h("td", { text: r.citl === undefined ? "–" : signed(r.citl) })))));
    card.appendChild(h("div", { class: "table-scroll" }, tbl));
    card.appendChild(h("p", { class: "note", text: t().subNote(E.min_cell) }));
    return card;
  }

  // Nested cross-validation (all years): spread of performance across outer folds
  function nestedCard() {
    const N = V.nested;
    if (!N || !N.stages) return null;
    const out = h("div", { class: "card card-pad block" });
    for (const [st, per] of Object.entries(N.stages)) for (const [tg, r] of Object.entries(per)) {
      out.appendChild(h("div", { class: "chart-head" }, h("div", null,
        h("div", { class: "chart-title", text: tx("Nested cross-validation on all years", "Vnořená křížová validace na všech letech") }),
        h("div", { class: "chart-sub", text: tx(`${stageLabel(st)}, ${tg === "good_outcome" ? "mRS 0–2" : "death"} · ${N.folds} outer folds × ${N.repeats} · every learner re-tuned inside each fold with ${N.trials} candidate settings · n ${num(r.n)}`,
          `${stageLabel(st)}, ${tg === "good_outcome" ? "mRS 0–2" : "úmrtí"} · ${N.folds} vnějších foldů × ${N.repeats} · každý model znovu laděn v každém foldu s ${N.trials} kandidátními nastaveními · n ${num(r.n)}`) }))));
      const rows = LEARNERS.filter((k) => r.models[k]).map((k) => {
        const m = r.models[k];
        return h("tr", { class: k === REF ? "chosen" : "" }, h("th", { scope: "row" }, h("span", { class: "dot", style: `background:${mcolor(k)}` }), mname(k)),
          h("td", { text: `${num(m.auroc.mean, 3)} ± ${num(m.auroc.sd, 3)}` }), h("td", { text: k === REF ? tx("reference", "reference") : `${signed(m.delta_mean, 3)} (${signed(m.delta_lo, 3)} to ${signed(m.delta_hi, 3)})` }),
          h("td", { text: m.folds_better === null ? "–" : `${m.folds_better} / ${m.auroc ? r.models[k].fold_auroc.length : "–"}` }),
          h("td", { text: num(m.abs_citl_mean, 2) }), h("td", { text: num(m.slope.mean, 2) }), h("td", { text: num(m.brier.mean, 3) }));
      });
      out.appendChild(h("div", { class: "table-scroll" }, h("table", { class: "data compact" },
        h("thead", null, h("tr", null, [t().model, tx("AUROC mean ± SD", "AUROC průměr ± SD"), tx("ΔAUROC vs reference (2.5–97.5th pct)", "ΔAUROC vs reference (2,5.–97,5. pct)"), tx("Folds better", "Lepší foldy"), "|CITL|", t().slope, t().brier].map((x) => h("th", { scope: "col", text: x })))),
        h("tbody", null, rows))));
    }
    out.appendChild(h("p", { class: "note", text: tx(
      "Every year contributes to training and testing, so these numbers describe the modelling procedure rather than the published model; the temporal test above is the main evidence. The ranges across folds are spreads, not confidence intervals.",
      "Do tréninku i testu přispívají všechny roky, proto tato čísla popisují postup modelování spíše než zveřejněný model; hlavním důkazem je časový test výše. Rozpětí přes foldy jsou rozptyly, nikoli intervaly spolehlivosti.") }));
    return out;
  }

  function driftCard(st, tg, vt, chosen, yrs, dyrs) {
    const r = vt.models[chosen] || vt.models[REF];
    const dep = (((V.deployed || {})[st] || {})[tg] || {}).learners || {};
    const shift = dep[chosen] ? dep[chosen].intercept_recent : null;
    const dCard = h("div", { class: "card card-pad block" });
    dCard.appendChild(h("div", { class: "chart-title", text: t().driftTitle(yrs), style: "margin-bottom:8px" }));
    dCard.appendChild(h("p", { class: "callout", text: t().driftText(mname(chosen), signed(r.citl), pct(vt.prevalence_dev), dyrs, pct(vt.prevalence_test), yrs, signed(shift), RECENT) }));
    return dCard;
  }

  // ================================================================== about
  function renderAbout() {
    const root = $("#view-about");
    root.replaceChildren();
    const nAll = E.kpis.n;
    const sp = V.splits.primary;
    const ch = (s2) => chosenOf(s2, "good_outcome");
    const auc = (s2) => num(sp.stages[s2].targets.good_outcome.models[ch(s2)].auroc, 2);
    const en = lang !== "cs";
    const P = (txt) => h("p", { text: txt });
    const flowSteps = (DATA.preprocessing && DATA.preprocessing.flow) ? DATA.preprocessing.flow.steps : E.flow;
    const kept = (flowSteps[flowSteps.length - 1] || {}).kept || {};
    const chosenNames = Array.from(new Set(STAGES.flatMap((s2) => TARGETS.map((t2) => chosenOf(s2, t2))))).map(mname);
    const prose = h("div", { class: "prose" },
      h("h2", { style: "font-size:24px", text: t().aboutTitle }),
      P(en
        ? `MTOP estimates the chance that a patient taken for mechanical thrombectomy for acute ischaemic stroke will be functionally independent (modified Rankin Scale 0–2) 90 days later, and the chance of death. The estimate is updated at four points of care: before the procedure, at its end, at 24 hours and on day 7. Each update uses only what is known at that point.`
        : `MTOP odhaduje šanci, že pacient přijatý k mechanické trombektomii pro akutní ischemickou CMP bude za 90 dní funkčně nezávislý (modifikovaná Rankinova škála 0–2), a riziko úmrtí. Odhad se aktualizuje ve čtyřech okamžicích péče: před výkonem, na jeho konci, po 24 hodinách a 7. den. Každá aktualizace používá jen údaje známé v daném okamžiku.`),
      h("h3", { text: en ? "Data" : "Data" }),
      P(en
        ? `The models were built from the thrombectomy registry of University Hospital Ostrava (FNO). Of ${num(flowSteps[0].n)} registry entries, ${num(nAll)} adults with a recorded 90-day mRS were analysed (${E.period[0]}–${E.period[1]}), whatever the procedure turned out to be: ${num(kept.diagnostic_only || 0)} had a diagnostic angiography only, because before the procedure nobody knows that it will be. Impossible times (IV thrombolysis after puncture, recanalisation before puncture) were set to missing and the ${num(kept.impossible_time || 0)} patients kept. An NIHSS of 43 is read as death and 99 as "not assessable"; these codes are never used as scores. Every cleaning step and check is listed under Registry → Data and preprocessing.`
        : `Modely vycházejí z registru trombektomií Fakultní nemocnice Ostrava (FNO). Z ${num(flowSteps[0].n)} záznamů registru bylo analyzováno ${num(nAll)} dospělých se známým mRS za 90 dní (${E.period[0]}–${E.period[1]}), bez ohledu na to, jaký výkon nakonec proběhl: ${num(kept.diagnostic_only || 0)} mělo jen diagnostickou angiografii, protože před výkonem to nikdo neví. Nemožné časy (IV trombolýza po punkci, rekanalizace před punkcí) byly nastaveny jako chybějící a ${num(kept.impossible_time || 0)} pacientů ponecháno. NIHSS 43 je chápáno jako úmrtí a 99 jako „nehodnotitelné“; tyto kódy se nikdy nepoužívají jako skóre. Všechny kroky čištění a kontroly jsou v části Registr → Data a předzpracování.`),
      h("h3", { text: en ? "Models" : "Modely" }),
      P(en
        ? `Before the procedure the models use the 33 pre-procedural predictors of the 2026 FNO analysis; the end of the procedure adds reperfusion and procedure details (any procedure type, including stenting, angioplasty and intra-arterial lysis), 24 hours the NIHSS course and haemorrhage, and day 7 the day-7 NIHSS, oedema and decompression. Five learners were tuned with the same budget on the same folds: logistic regression with restricted cubic splines (the reference), linear logistic regression, LightGBM with monotone constraints, XGBoost and a multilayer perceptron. A rule fixed before testing picks the model for each stage and outcome; the calculator currently uses ${chosenNames.join(" and ")}. Published scores (THRIVE, SPAN-100, HIAT2, PRE, NIHSS alone), refitted predictor sets (THRIVE-c, MR PREDICTS@24H) and the 7-item FNO bedside score are benchmarks.`
        : `Před výkonem modely používají 33 předprocedurálních prediktorů z analýzy FNO z roku 2026; konec výkonu přidává reperfuzi a detaily výkonu (jakýkoli typ výkonu včetně stentu, angioplastiky a intraarteriální lýzy), 24 hodin vývoj NIHSS a krvácení a 7. den NIHSS 7. dne, edém a dekompresi. Pět modelů bylo laděno se stejným rozpočtem na stejných foldech: logistická regrese s restrikovanými kubickými splajny (reference), lineární logistická regrese, LightGBM s monotónními omezeními, XGBoost a vícevrstvý perceptron. Pravidlo stanovené před testováním vybírá model pro každou fázi a výsledek; kalkulačka nyní používá ${chosenNames.join(" a ")}. Srovnáním jsou publikovaná skóre (THRIVE, SPAN-100, HIAT2, PRE, samotné NIHSS), přepočtené sady prediktorů (THRIVE-c, MR PREDICTS@24H) a 7položkové bedside skóre FNO.`),
      h("h3", { text: en ? "Validation" : "Validace" }),
      P(en
        ? `Temporal validation on patients treated in ${yearsOf()} (models developed on ${devYears()}) gave an AUROC for mRS 0–2 of ${auc("admission")} before the procedure, ${auc("procedure")} at its end, ${auc("h24")} at 24 hours and ${auc("d7")} on day 7; a sensitivity analysis tests on ${yearsOf("sensitivity")}. Outcomes in recent years were worse than the development data predicted, so the published models are refitted on all years and their intercepts re-estimated on the pooled years ${RECENT}.`
        : `Časová validace na pacientech léčených ${yearsOf()} (modely vyvinuté na letech ${devYears()}) ukázala AUROC pro mRS 0–2 ${auc("admission")} před výkonem, ${auc("procedure")} na jeho konci, ${auc("h24")} po 24 hodinách a ${auc("d7")} 7. den; citlivostní analýza testuje na letech ${yearsOf("sensitivity")}. Výsledky v posledních letech byly horší, než vývojová data předpovídala, proto jsou zveřejněné modely přetrénovány na všech letech a jejich intercept odhadnut ze sloučených let ${RECENT}.`),
      h("h3", { text: en ? "Limits" : "Omezení" }),
      h("ul", null, (en ? [
        "Single centre; performance elsewhere is unknown until externally validated.",
        "The temporal test has fewer than 100 good outcomes; its intervals are wide. The sensitivity test years double the events.",
        "ASPECTS was recorded from 2016 and anaesthesia type from about 2012; earlier patients contribute through missing-value handling.",
        "Since 2021 about half of patients lack a 24-hour NIHSS and many lack day-7 data, so the 24-h and day-7 validation rests on a selected subset.",
        "Collateral status, core volume and number of passes are not in the registry and cannot be used.",
      ] : [
        "Jedno centrum; výkon jinde není znám, dokud nebude externě validován.",
        "Časový test má méně než 100 dobrých výsledků; jeho intervaly jsou široké. Citlivostní testovací roky počet událostí zdvojnásobují.",
        "ASPECTS je evidováno od roku 2016 a typ anestezie přibližně od roku 2012; starší pacienti přispívají přes zpracování chybějících hodnot.",
        "Od roku 2021 asi polovině pacientů chybí NIHSS za 24 h a mnohým údaje ze 7. dne, takže validace fází 24 h a 7. den stojí na vybrané podskupině.",
        "Kolaterální oběh, objem jádra infarktu a počet průchodů registr neobsahuje.",
      ]).map((x) => h("li", { text: x }))),
      h("h3", { text: en ? "Privacy" : "Ochrana údajů" }),
      P(en
        ? `This page contains no patient records. It carries the trained models (regression coefficients, and trees in which every leaf summarises at least 20 patients) and aggregate statistics in which groups under ${E.min_cell} patients are hidden. Values typed into the calculator stay in your browser and are not stored or sent.`
        : `Tato stránka neobsahuje žádné záznamy pacientů. Obsahuje natrénované modely (koeficienty regresí a stromy, v nichž každý list shrnuje nejméně 20 pacientů) a souhrnné statistiky, ve kterých jsou skryty skupiny pod ${E.min_cell} pacientů. Hodnoty zadané do kalkulačky zůstávají ve vašem prohlížeči a nikam se neukládají ani neodesílají.`));
    const meta = h("dl", { class: "meta-list" });
    const add = (k, v) => meta.appendChild(h("div", null, h("dt", { text: k }), h("dd", null, v)));
    const authors = SITE.authors || [];
    add(en ? "Authors" : "Autoři", h("span", { text: authors.map((a) => a.name).join(", ") }));
    if (authors.length) add(en ? "Affiliations" : "Pracoviště", h("span", { text: Array.from(new Set(authors.flatMap((a) => (a.affiliations || []).map(L)))).join("; ") }));
    if (SITE.funding) add(en ? "Funding" : "Financování", h("span", { text: L(SITE.funding) }));
    if (SITE.contact) add(en ? "Contact" : "Kontakt", h("span", { class: "copyable", text: SITE.contact }));
    add(en ? "Version" : "Verze", h("span", { text: `${B.version} · ${B.generated}` }));
    if (B.fingerprint) add(en ? "Model fingerprint (SHA-256)" : "Otisk modelu (SHA-256)", h("span", { class: "copyable", title: B.fingerprint, text: `${B.fingerprint.slice(0, 16)}…` }));
    if (SITE.repo_url) add(en ? "Code" : "Kód", h("a", { href: SITE.repo_url, target: "_blank", rel: "noopener", text: SITE.repo_url.replace(/^https?:\/\//, "") }));
    add(en ? "Reporting" : "Reporting", h("a", { href: "#reporting", onclick: (e) => { e.preventDefault(); goTo("reporting"); }, text: en ? "TRIPOD+AI checklist and other standards" : "Checklist TRIPOD+AI a další standardy" }));
    add(en ? "Intended use" : "Určení", h("span", { text: en ? "Research and teaching. Not a medical device; not for individual treatment decisions." : "Výzkum a výuka. Není zdravotnickým prostředkem; není určeno k rozhodování o léčbě jednotlivce." }));
    const aside = h("aside", { class: "about-side" }, h("div", { class: "card card-pad" }, meta));
    if (DATA.qr_svg && SITE.site_url) {
      const qr = h("div", { class: "qr", html: DATA.qr_svg, role: "img", "aria-label": `QR: ${SITE.site_url}` });
      aside.appendChild(h("div", { class: "card card-pad qr-card" }, qr,
        h("div", null, h("div", { class: "chart-title", text: en ? "Open on your phone" : "Otevřít v telefonu" }), h("div", { class: "copyable", text: SITE.site_url.replace(/^https?:\/\//, "") }))));
    }
    root.appendChild(h("div", { class: "about-grid" }, prose, aside));
  }

  // ------------------------------------------------------------------ shared API for the other views
  window.MTOPApp = {
    DATA, B, V, E, X, SITE, M, XE, C, h, S, STAGES, STAGE, FEAT, TARGETS, MODEL_NAMES, LEARNERS, MRS_COLORS, REF, SPLITS, RECENT,
    t, L, tx, lang: () => lang, num, pct, pct100, signed, stageLabel, idx, mname, mcolor, statusChip, yearsOf, devYears, chosenOf, entryOf,
    predictStage, modelInputs, samples, threeBand, registerView, render, renderView, goTo, store,
    viewLabel: (id) => { const v = VIEWS[ALIASES[id] ? ALIASES[id][0] : id]; return v ? v.label() : id; },
    boot() {
      const hash = (location.hash || "").replace("#", "");
      if (ALIASES[hash]) { Object.assign(S, ALIASES[hash][1]); S.view = ALIASES[hash][0]; } else if (VIEWS[hash]) S.view = hash;
      render();
      window.addEventListener("hashchange", () => {
        const id = (location.hash || "").replace("#", "");
        if (ALIASES[id] || VIEWS[id]) goTo(id);
      });
    },
  };
})();
