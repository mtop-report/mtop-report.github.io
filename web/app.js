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
  const HISTORY = ["htn", "dm", "arrhythmia", "stroke_prior", "anticoag"];
  const TARGETS = ["good_outcome", "death90"];

  const EXAMPLE = {
    age: 72, sex_male: 1, mrs_pre: 0, nihss_adm: 16, aspects: 9, occlusion: "M1", side_left: 1,
    onset_to_puncture: 185, ivt: 1, transfer: 0, glucose: 7.2, sbp: 155,
    htn: 1, dm: 0, arrhythmia: 1, stroke_prior: 0, anticoag: 0,
    tici: 4, procedure_min: 40, technique: "aspiration", stenting: 0, anesthesia: "general",
    nihss_24h: 7, sich: 0, nihss_7d: 4, edema: 0,
  };
  const S = {
    view: "calc",
    current: "h24",
    viewed: "h24",
    inputs: Object.assign({}, EXAMPLE),
    isExample: true,
    explainTarget: "good_outcome",
    whatif: {},
    valTarget: "good_outcome",
    valStage: "admission",
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

  function displayValue(key, v) {
    const f = FEAT[key];
    if (!f) return "";
    if (key === "nihss_change_24h") {
      const a = S.inputs.nihss_24h, b = S.inputs.nihss_adm;
      if (a === null || b === null) return t().unknown;
      const d = a - b;
      return d > 0 ? `+${d}` : d < 0 ? `−${Math.abs(d)}` : "0";
    }
    if (v === null || v === undefined || v === "") return t().unknown.toLowerCase();
    if (key === "onset_to_puncture") return t().hm(v);
    if (f.options) {
      const o = f.options.find((x) => String(x.value) === String(v));
      if (o) return L(o.label).replace(/ · .*/, "");
    }
    const unit = f.unit ? ` ${L(f.unit)}` : "";
    return `${num(v, f.step && f.step < 1 ? 1 : 0)}${unit}`;
  }

  // ------------------------------------------------------------------ prediction
  function modelInputs() {
    const x = Object.assign({}, S.inputs);
    if (POSTERIOR.has(x.occlusion)) x.side_left = 0;
    return x;
  }
  // Unknown optional inputs are averaged over registry values (quantiles / category mix),
  // instead of using the trees' missing-value branches, which encode *who* had missing data.
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
  function unknownKeys(spec, inputs) {
    const used = new Set(spec.column_feature);
    return Object.keys(IMPUTE).filter((k) => used.has(k) && (inputs[k] === null || inputs[k] === undefined));
  }
  function samples(spec, inputs) {
    const unk = unknownKeys(spec, inputs);
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
  function predictStage(stage, inputs, target) {
    const spec = B.models[stage][target];
    const xs = samples(spec, inputs || modelInputs());
    let p = 0, lg = 0;
    const rows = xs.map((x) => M.buildRow(spec, B.features, x));
    for (const row of rows) { const r = M.predict(spec, row, { recent: true }); p += r.p; lg += r.logit; }
    return { spec, rows, p: p / rows.length, logit: lg / rows.length, unknown: unknownKeys(spec, inputs || modelInputs()) };
  }
  function explainStage(pr) {
    const acc = {};
    let base = 0;
    for (const row of pr.rows) {
      const e = M.explain(pr.spec, row);
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
  function realityBin(stage, p) {
    const bins = (V.stages[stage].targets.good_outcome.deployed || {}).reality_bins || [];
    if (!bins.length) return null;
    let best = bins[0];
    for (const b of bins) if (p >= b.lo - 1e-9) best = b;
    return best;
  }

  // ------------------------------------------------------------------ DOM refs
  const $ = (sel, el) => (el || document).querySelector(sel);
  const app = $("#app");

  const VIEW_ORDER = ["calc", "registry", "validation", "adaptive", "network", "roadmap", "about"];
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
    const order = st === "admission"
      ? ["age", "sex_male", "nihss_adm", "mrs_pre", "occlusion", "side_left", "ivt", "aspects", "transfer", "onset_to_puncture", "glucose", "sbp"]
      : feats.map((f) => f.key);
    for (const key of order) {
      if (HISTORY.includes(key)) continue;
      if (FEAT[key]) body.appendChild(field(FEAT[key]));
    }
    if (st === "admission") body.appendChild(historyField());
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
      ? tx("GFAP is being collected prospectively at FNO. It will enter the model once enough patients with known 90-day outcome are available, after its added value has been tested.", "GFAP se ve FNO prospektivně sbírá. Do modelu vstoupí, až bude dost pacientů se známým výsledkem za 90 dní a jeho přínos bude ověřen.")
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

  const WIDE = new Set(["occlusion", "onset_to_puncture", "technique"]);
  const SHOW_VALUE = new Set(["nihss_adm", "nihss_24h", "nihss_7d", "onset_to_puncture"]);

  function field(f) {
    const wrap = h("div", { class: `field${WIDE.has(f.key) ? " wide" : ""}`, id: `field-${f.key}` });
    const lblText = f.key === "aspects" && POSTERIOR.has(S.inputs.occlusion) ? t().pcAspects : L(f.label);
    const valSpan = h("span", { class: "val" });
    const lab = h("div", { class: "field-label" }, h("span", { class: "lbl", id: `lbl-${f.key}`, text: lblText }), valSpan);
    wrap.appendChild(lab);
    const refresh = () => {
      const v = S.inputs[f.key];
      if (f.type === "numeric") {
        valSpan.replaceChildren();
        if (v === null || !SHOW_VALUE.has(f.key)) return;
        if (f.key === "onset_to_puncture") valSpan.textContent = t().hm(v);
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
      const seg = h("div", { class: `seg${isCat && opts.length > 3 ? " cols-2" : ""}${f.key === "mrs_pre" || f.key === "tici" ? " nums" : ""}`, role: "group", "aria-labelledby": `lbl-${f.key}` });
      const buttons = [];
      const mk = (val, label, extraClass) => {
        const b = h("button", { type: "button", class: extraClass || "", "data-v": val === null ? "" : String(val) }, label);
        b.addEventListener("click", () => {
          const v = val === null ? null : isCat ? String(val) : Number(val);
          setInput(f.key, v);
          paint();
          if (f.key === "occlusion") { renderCalcKeepScroll(); }
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
      if (f.key === "side_left" && POSTERIOR.has(S.inputs.occlusion)) {
        wrap.classList.add("disabled");
        buttons.forEach((b) => (b.disabled = true));
      }
    }
    if (f.help) wrap.appendChild(h("div", { class: "help", text: L(f.help) }));
    return wrap;
  }

  function renderCalcKeepScroll() {
    const y = window.scrollY;
    renderCalc();
    window.scrollTo(0, y);
  }

  function historyField() {
    const wrap = h("div", { class: "field wide", id: "field-history" });
    wrap.appendChild(h("div", { class: "field-label" }, h("span", { class: "lbl", id: "lbl-history", text: t().history }), h("span", { class: "val" })));
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
  function headlineCard(st, pv) {
    const g = pv.good.p, d = pv.death.p;
    const bin = realityBin(st, g);
    const unc = XE.uncertainty(((X.uncertainty || {})[st] || {}).good_outcome, g);
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
      h("span", null, tx("Range across 100 refits of the model (95%): ", "Rozpětí ze 100 přetrénování modelu (95 %): "), h("b", { text: `${num(unc[0] * 100)}–${pct(unc[1])}` }))));
    if (bin) key.appendChild(h("div", null, h("span", { class: "sw-obs", "aria-hidden": "true" }),
      h("span", { html: t().reality(pct(bin.lo), pct(bin.hi), pct(bin.obs), bin.n) })));
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
  function aucFor(st, name) {
    for (const s2 of [st, "admission", "h24"]) {
      const r = ((V.stages[s2] || {}).targets || {}).good_outcome;
      if (r && r.temporal[name]) return { auc: r.temporal[name].auroc, stage: s2 };
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
    const am = a ? aucFor(a.stage, "LightGBM") : null;
    const at = a && a.stage !== st;
    card.appendChild(h("p", { class: "note", text: tx(
      `AUROC ${a ? num(a.auc, 2) : "–"} in 2021–2025 patients, against ${am ? num(am.auc, 2) : "–"} for MTOP${at ? " (both before thrombectomy)" : " at this stage"}; 0.774 on 2023–2025 patients in the earlier analysis. Items that are not recorded score 0.`,
      `AUROC ${a ? num(a.auc, 2) : "–"} u pacientů 2021–2025, MTOP ${am ? num(am.auc, 2) : "–"}${at ? " (obojí před trombektomií)" : " ve stejné fázi"}; v dřívější analýze 0,774 u pacientů 2023–2025. Nezadané položky mají 0 bodů.`) }));
    card.appendChild(h("p", { class: "note", text: st === "admission"
      ? tx("Why it can differ from MTOP for one patient: the score uses seven items in coarse bands and ignores occlusion site, IV thrombolysis, transfer, blood pressure, sex and most of the history. It ranks groups of patients almost as well as MTOP, but for an individual the two often differ by 10 percentage points or more.",
        "Proč se může od MTOP u jednoho pacienta lišit: skóre používá sedm položek v hrubých pásmech a nebere v úvahu místo okluze, IV trombolýzu, překlad, tlak, pohlaví ani většinu anamnézy. Skupiny pacientů seřadí téměř stejně dobře jako MTOP, u jednotlivce se však oba odhady často liší o 10 procentních bodů i více.")
      : tx(`The score uses admission information only. MTOP at ${stageLabel(st).toLowerCase()} also knows ${st === "procedure" ? "the reperfusion result and procedure details" : "the reperfusion result and the NIHSS course since admission"}, so for one patient the two can differ a lot; compare the score with MTOP's admission estimate in the pathway panel.`,
        `Skóre používá jen informace při přijetí. MTOP ve fázi ${stageLabel(st).toLowerCase()} zná navíc ${st === "procedure" ? "výsledek reperfuze a detaily výkonu" : "výsledek reperfuze a vývoj NIHSS od přijetí"}, takže se u jednoho pacienta mohou oba odhady výrazně lišit; skóre srovnávejte s odhadem MTOP při přijetí v panelu průběhu.`) }));
    return card;
  }

  function othersCard(st, x, pv) {
    const comps = (X.comparators || {})[st];
    if (!comps) return null;
    const xs = samples(pv.good.spec, x);
    const rows = [{ name: tx("MTOP (LightGBM)", "MTOP (LightGBM)"), p: pv.good.p, auc: aucFor(st, "LightGBM"), main: true }];
    for (const [name, comp] of Object.entries(comps)) {
      let sum = 0, ok = 0, pts = null;
      for (const xi of name === "Bedside score" ? [x] : xs) {
        let v;
        if (comp.kind === "points") {
          const sc = XE.POINTS[name](xi);
          v = XE.pointsProb(comp.map, sc);
          if (xs.length === 1 || name === "Bedside score") pts = sc;
        } else v = XE.comparatorProb(comp, xi, B.features, M.buildRow);
        if (v !== null && !Number.isNaN(v)) { sum += v; ok++; }
      }
      rows.push({ name: mname(name), detail: pointsText(name, pts), p: ok ? sum / ok : null, auc: aucFor(st, name) });
    }
    const card = h("div", { class: "card card-pad" });
    card.appendChild(h("div", { class: "chart-head" }, h("div", null,
      h("div", { class: "chart-title", text: tx("Other models for this patient", "Další modely pro tohoto pacienta") }),
      h("div", { class: "chart-sub", text: tx(`Chance of mRS 0–2 at ${stageLabel(st)}; AUROC from 2021–2025 validation`, `Šance na mRS 0–2 · ${stageLabel(st)}; AUROC z validace 2021–2025`) }))));
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
      "The two logistic regressions use exactly MTOP's inputs (one linear, one with restricted cubic splines for age, NIHSS, times and glucose) and are shifted to recent outcomes like MTOP. Published scores are mapped to probabilities with a logistic recalibration on the FNO registry; THRIVE-c and MR PREDICTS@24H are their predictor sets refitted on the registry (MR PREDICTS@24H without collateral score, which the registry does not record). * AUROC measured at an earlier stage.",
      "Obě logistické regrese používají přesně vstupy MTOP (jedna lineární, druhá s restrikovanými kubickými splajny pro věk, NIHSS, časy a glykémii) a jsou posunuté k nedávným výsledkům stejně jako MTOP. Publikovaná skóre jsou převedena na pravděpodobnosti logistickou rekalibrací na registru FNO; THRIVE-c a MR PREDICTS@24H jsou jejich sady prediktorů přepočtené na registru (MR PREDICTS@24H bez skóre kolaterál, které registr neeviduje). * AUROC změřené v dřívější fázi.") }));
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
    card.appendChild(list);
    return card;
  }

  const WHATIF_DEFAULT = { admission: "onset_to_puncture", procedure: "tici", h24: "nihss_24h", d7: "nihss_7d" };
  const WHATIF_KEYS = ["onset_to_puncture", "age", "nihss_adm", "aspects", "glucose", "sbp", "mrs_pre", "tici", "procedure_min", "nihss_24h", "nihss_7d"];

  function whatifCard(st, x) {
    const avail = WHATIF_KEYS.filter((k) => FEAT[k] && idx(FEAT[k].stage) <= idx(st));
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
    const fmtX = (v) => (key === "onset_to_puncture" ? `${num(v / 60, v % 60 === 0 ? 0 : 1)} h` : num(v, f.step && f.step < 1 ? 1 : 0));
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
          `Outcomes drifted: with all the other factors accounted for, the odds of ${what} were ${ch}% ${dir} for every five years later in the registry (${ci}). This is why MTOP's estimates are shifted to the outcome level of 2021–2025.`,
          `Výsledky se v čase posunuly: po zohlednění všech ostatních faktorů byla šance ${what} o ${ch} % ${dir} za každých pět let pozdější léčby (${ci}). Proto jsou odhady MTOP posunuty na úroveň výsledků let 2021–2025.`) }));
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
    root.appendChild(h("div", { class: "view-head" }, h("div", null,
      h("h2", { text: t().regTitle }), h("p", { text: t().regDesc(num(k.n), E.period[0], E.period[1], E.min_cell) }))));
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

    // completeness (all adults in registry)
    const ry = E.registry_years;
    const compCard = h("div", { class: "card card-pad block" });
    const cSeries = [
      { label: t().mrsMissing, color: "var(--c1)", points: ry.map((r, i) => ({ x: i, y: r.outcome_missing })) },
      { label: t().nihss24Missing, color: "var(--c2)", points: ry.map((r, i) => ({ x: i, y: r.nihss24_missing })) },
    ];
    const fc = C.figure(compCard, { title: t().tMissing, sub: lang === "cs" ? "Všichni dospělí v registru včetně pacientů bez známého výsledku. Chybějící údaje v posledních letech oslabují časnou validaci fází 24 h a 7. den." : "All adults in the registry, including those without a known outcome. Recent gaps weaken the temporal validation of the 24-h and day-7 stages.",
      legend: cSeries.map((s) => ({ label: s.label, color: s.color, type: "line" })),
      table: { columns: [lang === "cs" ? "Rok" : "Year", "n", t().mrsMissing, t().nihss24Missing], rows: ry.map((r) => [r.label, r.n, pct100(r.outcome_missing, 1), pct100(r.nihss24_missing, 1)]) },
      labels: { table: t().showTable, chart: t().showChart } });
    C.lineChart(fc.body, { height: 170, x: { type: "cat", cats: ry.map((r) => r.label) }, y: { domain: [0, 60], ticks: [0, 20, 40, 60] }, series: cSeries, tipFmt: (v) => pct100(v, 1), aria: t().tMissing });
    root.appendChild(compCard);

    // Table 1
    const t1 = E.table1;
    const t1Card = h("div", { class: "card card-pad block" });
    t1Card.appendChild(h("div", { class: "chart-head" }, h("div", null, h("div", { class: "chart-title", text: t().table1Title }), h("div", { class: "chart-sub", text: t().t1Note }))));
    const cols = ["all", "dev", "test", "good", "poor"];
    const head = [t().variable, `${t().t1All} (n ${num(t1.n.all)})`, `${t().t1Dev} (n ${num(t1.n.dev)})`, `${t().t1Test} (n ${num(t1.n.test)})`, `${t().t1Good} (n ${num(t1.n.good)})`, `${t().t1Poor} (n ${num(t1.n.poor)})`];
    const cell = (r, c) => {
      if (r.var === "good" && (c === "good" || c === "poor")) return "–";
      const v = r[c];
      if (!v) return "–";
      if (r.kind === "median") return v.median === null ? "–" : `${num(v.median)} (${num(v.iqr[0])}–${num(v.iqr[1])})`;
      if (v.small) return `<${E.min_cell}`;
      return pct100(v.pct, 1);
    };
    const table = h("table", { class: "data" },
      h("thead", null, h("tr", null, head.map((x) => h("th", { scope: "col", text: x })))),
      h("tbody", null, t1.rows.map((r) => h("tr", null, h("th", { scope: "row", text: L(r.label) }), cols.map((c) => h("td", { text: cell(r, c) }))))));
    t1Card.appendChild(h("div", { class: "table-scroll" }, table));
    root.appendChild(t1Card);
  }

  // ================================================================== validation
  const MODEL_NAMES = {
    "LightGBM": { en: "LightGBM", cs: "LightGBM" },
    "Logistic regression": { en: "Logistic regression", cs: "Logistická regrese" },
    "XGBoost": { en: "XGBoost", cs: "XGBoost" },
    "MLP": { en: "MLP (neural network)", cs: "MLP (neuronová síť)" },
    "THRIVE": { en: "THRIVE score", cs: "Skóre THRIVE" },
    "SPAN-100": { en: "SPAN-100", cs: "SPAN-100" },
    "NIHSS 24 h": { en: "NIHSS at 24 h alone", cs: "Samotné NIHSS za 24 h" },
    "NIHSS day 7": { en: "NIHSS on day 7 alone", cs: "Samotné NIHSS 7. den" },
    "THRIVE-c": { en: "THRIVE-c predictors (refit)", cs: "Prediktory THRIVE-c (přepočteno)" },
    "HIAT2": { en: "HIAT2 score", cs: "Skóre HIAT2" },
    "PRE": { en: "PRE score", cs: "Skóre PRE" },
    "MR PREDICTS@24H": { en: "MR PREDICTS@24H predictors (refit)", cs: "Prediktory MR PREDICTS@24H (přepočteno)" },
    "Logistic regression (splines)": { en: "Logistic regression with splines", cs: "Logistická regrese se splajny" },
    "Bedside score": { en: "FNO bedside score (7 items)", cs: "Bedside skóre FNO (7 položek)" },
  };
  const LEARNERS = ["LightGBM", "Logistic regression", "XGBoost", "MLP", "Logistic regression (splines)"];
  const MODEL_COLORS = ["var(--c1)", "var(--c2)", "var(--c3)", "var(--c4)", "var(--c5)", "var(--c6)"];
  const mname = (k) => L(MODEL_NAMES[k] || { en: k, cs: k });
  const signed = (v, d = 2) => (v === null || v === undefined ? "–" : `${v > 0 ? "+" : v < 0 ? "−" : ""}${num(Math.abs(v), d)}`);

  function renderValidation() {
    const root = $("#view-validation");
    root.replaceChildren();
    const s0 = V.stages.admission;
    root.appendChild(h("div", { class: "view-head" }, h("div", null, h("h2", { text: t().valTitle }), h("p", { text: t().valDesc(num(s0.n_dev), num(s0.n_test)) }))));
    const seg = h("div", { class: "mini-seg", role: "group", "aria-label": t().outcome },
      TARGETS.map((k) => h("button", { type: "button", "aria-pressed": String(S.valTarget === k), onclick: () => { S.valTarget = k; renderValidation(); } }, k === "good_outcome" ? t().targetGood : t().targetDeath)));
    const stSel = h("select", { class: "sel", id: "val-stage" });
    for (const st of STAGES) stSel.appendChild(h("option", { value: st, selected: st === S.valStage ? true : null }, `${idx(st) + 1} · ${L(STAGE[st].title)}`));
    stSel.addEventListener("change", () => { S.valStage = stSel.value; renderValidation(); });
    root.appendChild(h("div", { class: "controls" }, h("span", { text: t().outcome }), seg, h("label", { for: "val-stage" }, t().stage), stSel));

    const tg = S.valTarget;
    const vt = V.stages[S.valStage].targets[tg];
    const temporal = vt.temporal;
    const modelKeys = Object.keys(temporal);

    // AUROC across stages
    const row1 = h("div", { class: "grid-2 block" });
    const rowTable = h("div", { class: "block" });
    const aCard = h("div", { class: "card card-pad" });
    const learners = LEARNERS.filter((k) => V.stages[S.valStage].targets[tg].temporal[k]);
    const aSeries = learners.map((k, i) => ({
      label: mname(k), color: MODEL_COLORS[LEARNERS.indexOf(k)], dots: true, offset: (i - (learners.length - 1) / 2) * 5,
      points: STAGES.map((st, j) => {
        const r = V.stages[st].targets[tg].temporal[k];
        return { x: j, y: r.auroc, lo: r.ci.auroc[0], hi: r.ci.auroc[1] };
      }),
    }));
    const fa = C.figure(aCard, {
      title: t().aurocTitle, sub: t().aurocSub, legend: aSeries.map((s) => ({ label: s.label, color: s.color, type: "line" })),
      table: { columns: [t().model].concat(STAGES.map(stageLabel)), rows: learners.map((k) => [mname(k)].concat(STAGES.map((st) => { const r = V.stages[st].targets[tg].temporal[k]; return `${num(r.auroc, 3)} (${num(r.ci.auroc[0], 3)}–${num(r.ci.auroc[1], 3)})`; }))) },
      labels: { table: t().showTable, chart: t().showChart },
    });
    C.lineChart(fa.body, { height: 240, x: { type: "cat", cats: STAGES.map(stageLabel) }, y: { domain: [0.5, 1], ticks: [0.5, 0.6, 0.7, 0.8, 0.9, 1], fmt: (v) => num(v, 1) }, series: aSeries,
      tipFmt: (v, p) => `${num(v, 3)} (${num(p.lo, 2)}–${num(p.hi, 2)})`, aria: t().aurocTitle });
    row1.appendChild(aCard);

    // comparison table
    const cCard = h("div", { class: "card card-pad" });
    cCard.appendChild(h("div", { class: "chart-head" }, h("div", null, h("div", { class: "chart-title", text: t().compTitle(L(STAGE[S.valStage].title)) }),
      h("div", { class: "chart-sub", text: `n = ${num(V.stages[S.valStage].n_test)} · ${lang === "cs" ? "události" : "events"} ${num(temporal.LightGBM.events)} · ${lang === "cs" ? "vývoj" : "development"} n = ${num(V.stages[S.valStage].n_dev)}` }))));
    const bestAuc = Math.max(...modelKeys.map((k) => temporal[k].auroc));
    const tbl = h("table", { class: "data" },
      h("thead", null, h("tr", null, [t().model, "AUROC (95% CI)", t().brier, t().citl, t().slope, t().dAuc].map((x) => h("th", { scope: "col", text: x })))),
      h("tbody", null, modelKeys.map((k) => {
        const r = temporal[k];
        const d = r.ci.delta_auroc_vs_ref;
        return h("tr", null,
          h("th", { scope: "row", text: mname(k) }),
          h("td", { class: r.auroc === bestAuc ? "best" : "" }, `${num(r.auroc, 3)} `, h("span", { class: "ci", text: `${num(r.ci.auroc[0], 3)}–${num(r.ci.auroc[1], 3)}` })),
          h("td", { text: num(r.brier, 3) }),
          h("td", { text: signed(r.citl) }),
          h("td", { text: num(r.slope, 2) }),
          h("td", null, d ? `${signed(d.mean, 3)} ` : "–", d ? h("span", { class: "ci", text: `${signed(d.ci[0], 3)} to ${signed(d.ci[1], 3)}` }) : null));
      })));
    cCard.appendChild(h("div", { class: "table-scroll" }, tbl));
    const intr = vt.internal;
    cCard.appendChild(h("p", { class: "note", text: `${t().compNote} ${t().internalCv}: AUROC ${num(intr.auroc, 3)} (${lang === "cs" ? "průměr přes foldy" : "fold mean"} ${num(intr.fold_auroc_mean, 3)} ± ${num(intr.fold_auroc_sd, 3)}).` }));
    rowTable.appendChild(cCard);
    root.appendChild(rowTable);
    root.appendChild(row1);

    // ROC + calibration
    const row2 = h("div", { class: "grid-2 block" });
    const rCard = h("div", { class: "card card-pad" });
    const rSeries = modelKeys.map((k) => {
      const li = LEARNERS.indexOf(k);
      return { label: mname(k), color: li >= 0 ? MODEL_COLORS[li] : "var(--muted)", width: k === "LightGBM" ? 2.5 : li >= 0 ? 1.5 : 1.1, points: vt.curves.roc[k].map((p) => ({ x: p.fpr, y: p.tpr })) };
    });
    const rLegend = rSeries.filter((s2) => LEARNERS.map(mname).includes(s2.label)).map((s2) => ({ label: s2.label, color: s2.color, type: "line" }));
    if (rSeries.length > rLegend.length) rLegend.push({ label: tx("Published scores (grey)", "Publikovaná skóre (šedě)"), color: "var(--muted)", type: "line" });
    const fr = C.figure(rCard, { title: t().rocTitle, sub: L(STAGE[S.valStage].title), legend: rLegend,
      table: { columns: [t().fpr].concat(rSeries.map((s) => s.label)), rows: vt.curves.roc.LightGBM.filter((_, i) => i % 5 === 0).map((p, j) => [num(p.fpr, 2)].concat(modelKeys.map((k) => num(vt.curves.roc[k][j * 5].tpr, 2)))) },
      labels: { table: t().showTable, chart: t().showChart } });
    C.lineChart(fr.body, { height: 240, x: { type: "lin", domain: [0, 1], fmt: (v) => num(v, 1), title: t().fpr }, y: { domain: [0, 1], fmt: (v) => num(v, 1), title: t().tpr, ticks: [0, 0.25, 0.5, 0.75, 1] }, series: rSeries, diagonal: true,
      tipFmt: (v) => num(v, 2), tipHead: (xv) => `${t().fpr} ${num(xv, 2)}`, aria: t().rocTitle });
    row1.appendChild(rCard);

    const kCard = h("div", { class: "card card-pad" });
    const cal = vt.curves.calibration;
    const fk = C.figure(kCard, { title: t().calTitle, sub: t().calSub,
      table: { columns: [t().predicted, t().observed, "n"], rows: cal.map((b) => [pct(b.pred, 1), pct(b.obs, 1), b.n]) },
      labels: { table: t().showTable, chart: t().showChart } });
    const calMax = Math.min(1, Math.ceil(Math.max(...cal.map((b) => Math.max(b.pred, b.obs))) * 10 + 0.5) / 10);
    C.lineChart(fk.body, { height: 260, x: { type: "lin", domain: [0, calMax], fmt: (v) => pct(v), title: t().predicted }, y: { domain: [0, calMax], fmt: (v) => pct(v), title: t().observed }, diagonal: true,
      series: [{ label: t().observed, color: "var(--c1)", dots: true, points: cal.map((b) => ({ x: b.pred, y: b.obs, n: b.n })) }],
      tipFmt: (v, p) => `${pct(v, 1)} · n ${p.n}`, tipHead: (xv) => `${t().predicted} ${pct(xv, 1)}`, aria: t().calTitle });
    row2.appendChild(kCard);
    const sc = stabilityCard(S.valStage, tg);
    if (sc) row2.appendChild(sc);
    root.appendChild(row2);
    const oc = ordinalCard(S.valStage);
    if (oc) { oc.classList.add("block"); root.appendChild(oc); }
    const sg = subgroupCard(tg, S.valStage);
    sg.classList.add("block");
    root.appendChild(sg);

    // drift
    const lg = temporal.LightGBM;
    const rc = vt.recalibration;
    const dep = vt.deployed;
    const dCard = h("div", { class: "card card-pad block" });
    dCard.appendChild(h("div", { class: "chart-title", text: t().driftTitle, style: "margin-bottom:8px" }));
    const devPrev = V.stages[S.valStage].targets[tg].internal.prevalence;
    dCard.appendChild(h("p", { class: "callout", text: t().driftText(signed(lg.citl), pct(devPrev), pct(lg.prevalence), signed(dep.intercept_recent), rc ? signed(rc.before.citl) : "–", rc ? signed(rc.after.citl) : "–") }));
    root.appendChild(dCard);

    // importance + DCA
    const row3 = h("div", { class: "grid-2 block" });
    const iCard = h("div", { class: "card card-pad" });
    const imp = Object.entries(vt.explain.importance).slice(0, 12);
    const fi = C.figure(iCard, { title: t().impTitle, sub: `${t().impSub} · ${L(STAGE[S.valStage].title)}`,
      table: { columns: [t().variable, "mean |SHAP|"], rows: imp.map(([k, v]) => [L(FEAT[k].label), num(v, 3)]) },
      labels: { table: t().showTable, chart: t().showChart } });
    C.hbars(fi.body, { rows: imp.map(([k, v]) => ({ label: L(FEAT[k].label), value: v, display: num(v, 2) })), color: "var(--c1)", valueLabel: "mean |SHAP|", aria: t().impTitle });
    row3.appendChild(iCard);

    const dcCard = h("div", { class: "card card-pad" });
    const dca = vt.curves.dca;
    const dm = vt.curves.dca_models || { LightGBM: dca.map((r) => ({ t: r.t, nb: r.model })) };
    const dKeys = Object.keys(dm);
    const modelSeries = dKeys.map((k) => {
      const li = LEARNERS.indexOf(k);
      return { label: mname(k), color: li >= 0 ? MODEL_COLORS[li] : "var(--faint)", width: k === "LightGBM" ? 2.5 : li >= 0 ? 1.6 : 1.1,
        points: dm[k].map((r) => ({ x: r.t, y: r.nb })) };
    });
    const dSeries = modelSeries.slice().reverse().concat([
      { label: t().treatAll, color: "var(--ink-2)", width: 1.5, dash: true, points: dca.map((r) => ({ x: r.t, y: r.all })) },
    ]);
    const lo = Math.min(-0.05, ...dca.map((r) => Math.max(r.all, -0.1)));
    const hi = Math.max(...dKeys.flatMap((k) => dm[k].map((r) => r.nb)), ...dca.map((r) => r.all)) + 0.03;
    const dLegend = modelSeries.filter((s2) => LEARNERS.map(mname).includes(s2.label)).map((s2) => ({ label: s2.label, color: s2.color, type: "line" }));
    if (modelSeries.length > dLegend.length) dLegend.push({ label: tx("Published and bedside scores (grey)", "Publikovaná a bedside skóre (šedě)"), color: "var(--faint)", type: "line" });
    dLegend.push({ label: t().treatAll, color: "var(--ink-2)", type: "line" }, { label: t().treatNone, color: "var(--line-2)", type: "line" });
    const fd = C.figure(dcCard, { title: t().dcaTitle, sub: t().dcaSub, legend: dLegend,
      table: { columns: [t().threshold].concat(dKeys.map(mname), [t().treatAll, t().treatNone]),
        rows: dca.map((r, i) => [pct(r.t)].concat(dKeys.map((k) => num(dm[k][i].nb, 3)), [num(r.all, 3), "0"])) },
      labels: { table: t().showTable, chart: t().showChart } });
    C.lineChart(fd.body, { height: 240, x: { type: "lin", domain: [0.05, 0.8], fmt: (v) => pct(v), title: t().threshold, ticks: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] }, y: { domain: [lo, hi], fmt: (v) => num(v, 2), title: t().netBenefit },
      series: dSeries.map((s) => ({ ...s, points: s.points.map((p) => ({ x: p.x, y: Math.max(p.y, lo) })) })), zeroLine: true,
      tipFmt: (v) => num(v, 3), tipHead: (xv) => `${t().threshold} ${pct(xv)}`, aria: t().dcaTitle });
    row3.appendChild(dcCard);
    root.appendChild(row3);
    const recon = reconciliationCard();
    if (recon) root.appendChild(recon);
  }

  // Same registry, earlier analysis: the HTML report rebuilt from the raw data
  function reconciliationCard() {
    const R = DATA.reconciliation;
    if (!R) return null;
    const rep = R.report_numbers;
    const steps = Object.fromEntries(R.report_cohort_rebuilt.map((s2) => [s2.step, s2.n]));
    const S1 = R.setups["report cohort, report split (test 2023-2025)"];
    const S2 = R.setups["report cohort, report split, + 12 extra predictors"];
    const S3 = R.setups["MTOP cohort, MTOP split (test 2021-2025)"];
    if (!S1 || !S3) return null;
    const A = (s2, k) => (s2 && s2.models[k] ? num(s2.models[k].auroc, 3) : "–");
    const card = h("div", { class: "card card-pad block recon" });
    card.appendChild(h("div", { class: "chart-head" }, h("div", null,
      h("div", { class: "chart-title", text: tx("Same registry, earlier analysis", "Stejný registr, dřívější analýza") }),
      h("div", { class: "chart-sub", text: tx(
        "An earlier FNO report (2026) analysed this registry with a different cohort, test period and predictor set. Rebuilt from the raw data, its set-up gives the same numbers.",
        "Dřívější zpráva FNO (2026) analyzovala tento registr s jinou kohortou, testovacím obdobím a sadou prediktorů. Po rekonstrukci ze surových dat dává stejná čísla.") }))));
    const agree = [
      [tx("Patients with known 90-day mRS", "Pacienti se známým mRS za 90 dní"), num(rep.known_mrs), num(steps["90-day mRS known"])],
      [tx("Test set 2023–2025", "Testovací soubor 2023–2025"), `${num(rep.test)} · ${pct(rep.test_good_rate, 1)} mRS 0–2`, `${num(S1.n_test)} · ${pct(S1.test_good_rate, 1)} mRS 0–2`],
      ["THRIVE", num(rep.auc_thrive, 3), A(S1, "THRIVE")],
      [tx("SPAN as age + NIHSS", "SPAN jako věk + NIHSS"), num(rep.auc_span100, 3), A(S1, "SPAN index (age + NIHSS)")],
      [tx("Bedside score", "Bedside skóre"), num(rep.auc_bedside, 3), A(S1, "Bedside score")],
      [tx("Regression with splines", "Regrese se splajny"), `${num(rep.auc_spline_lr, 3)} · ${tx("33 predictors", "33 prediktorů")}`, `${A(S2, "Logistic regression (splines)")} · ${S2 ? num(S2.predictors) : "–"} ${tx("predictors", "prediktorů")}`],
    ];
    card.appendChild(h("div", { class: "table-scroll" }, h("table", { class: "data" },
      h("thead", null, h("tr", null, [tx("Quantity (AUROC unless stated)", "Veličina (AUROC, není-li uvedeno jinak)"), tx("Earlier report", "Dřívější zpráva"), tx("Rebuilt here", "Rekonstrukce")].map((x) => h("th", { scope: "col", text: x })))),
      h("tbody", null, agree.map((r) => h("tr", null, h("th", { scope: "row", text: r[0] }), h("td", { text: r[1] }), h("td", { text: r[2] })))))));
    const why = h("div", { class: "recon-grid" });
    why.appendChild(h("div", null, h("h4", { text: tx("Why MTOP's own numbers differ", "Proč se čísla MTOP liší") }), h("ul", null, [
      tx(`MTOP leaves out the ${num(steps["of which diagnostic angiography only"])} diagnostic angiographies without thrombectomy and keeps the 29 patients with impossible times, whose times become missing.`,
        `MTOP vynechává ${num(steps["of which diagnostic angiography only"])} diagnostických angiografií bez trombektomie a ponechává 29 pacientů s nemožnými časy, jejichž časy se berou jako chybějící.`),
      tx(`MTOP tests on 2021–2025 (${num(S3.n_test)} patients) instead of 2023–2025 (${num(S1.n_test)}).`, `MTOP testuje na letech 2021–2025 (${num(S3.n_test)} pacientů) místo 2023–2025 (${num(S1.n_test)}).`),
      tx("MTOP uses 17 admission predictors; the report used 33.", "MTOP používá 17 prediktorů při přijetí; zpráva jich měla 33."),
      tx("SPAN-100 is scored here with its published cut-off of 100, which gives a lower AUROC than the continuous age + NIHSS.", "SPAN-100 se zde hodnotí s publikovanou hranicí 100, což dává nižší AUROC než spojitý součet věku a NIHSS."),
    ].map((x) => h("li", { text: x })))));
    const rk = [[tx("Report set-up (test 2023–2025)", "Nastavení zprávy (test 2023–2025)"), S1], [tx("MTOP set-up (test 2021–2025)", "Nastavení MTOP (test 2021–2025)"), S3]];
    why.appendChild(h("div", null, h("h4", { text: tx("Which model ranks first depends on the test years", "Pořadí modelů závisí na testovacích letech") }),
      h("div", { class: "table-scroll" }, h("table", { class: "data compact" },
        h("thead", null, h("tr", null, ["", "LightGBM", tx("Regression", "Regrese"), tx("Splines", "Splajny"), tx("Bedside", "Bedside")].map((x) => h("th", { scope: "col", text: x })))),
        h("tbody", null, rk.map(([lbl, s2]) => h("tr", null, h("th", { scope: "row", text: lbl }),
          h("td", { text: A(s2, "LightGBM (MTOP settings)") }), h("td", { text: A(s2, "Logistic regression (linear)") }),
          h("td", { text: A(s2, "Logistic regression (splines)") }), h("td", { text: A(s2, "Bedside score") })))))),
      h("p", { class: "note", text: (() => {
        const d = S3.models["LightGBM (MTOP settings)"].delta_vs_spline_lr;
        return tx(`On the report's test years all regression and tree models lie within 0.02 of each other with overlapping intervals, as the report found. On MTOP's larger 2021–2025 test set LightGBM leads regression with splines by ${num(-d.mean, 3)} (95% CI ${num(-d.ci[1], 3)} to ${num(-d.ci[0], 3)}).`,
          `Na testovacích letech zprávy leží všechny regresní i stromové modely do 0,02 od sebe s překrývajícími se intervaly, jak uvádí zpráva. Na větším testovacím souboru MTOP 2021–2025 vede LightGBM před regresí se splajny o ${num(-d.mean, 3)} (95% IS ${num(-d.ci[1], 3)} až ${num(-d.ci[0], 3)}).`);
      })() })));
    card.appendChild(why);
    return card;
  }

  function ordinalCard(st) {
    const ov = (X.ordinal_validation || {})[st];
    if (!ov) return null;
    const card = h("div", { class: "card card-pad" });
    const legend = [0, 1, 2, 3, 4, 5, 6].map((m) => ({ label: `mRS ${m}`, color: `var(--mrs${m})` }));
    const rows = [
      { label: tx("Observed 2021–2025", "Pozorováno 2021–2025"), n: ov.n, segs: ov.observed.map((v) => v * 100) },
      { label: tx("Predicted", "Predikováno"), n: ov.n, segs: ov.predicted.map((v) => v * 100) },
    ];
    const f = C.figure(card, { title: tx("Full mRS distribution (ordinal)", "Celé rozložení mRS (ordinální)"),
      sub: tx(`Development models applied to 2021–2025 · ordinal c-index ${num(ov.c_ordinal, 3)}`, `Vývojové modely použité na 2021–2025 · ordinální c-index ${num(ov.c_ordinal, 3)}`),
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
      sub: tx(`${stab.n_boot} bootstrap refits of the published model · typical 95% range by predicted risk`, `${stab.n_boot} přetrénování publikovaného modelu na bootstrap vzorcích · typické 95% rozpětí podle predikovaného rizika`),
      table: { columns: [tx("Predicted", "Predikováno"), tx("95% range", "95% rozpětí"), "n"], rows: bins.map((b) => [pct(b.p), `${pct(b.p_lo)}–${pct(b.p_hi)}`, b.n]) },
      labels: { table: t().showTable, chart: t().showChart } });
    C.lineChart(f.body, { height: 230, x: { type: "lin", domain: [0, 1], fmt: (v) => pct(v), title: tx("Estimate", "Odhad") }, y: { domain: [0, 1], fmt: (v) => pct(v), title: tx("Range across refits", "Rozpětí přes přetrénování") }, diagonal: true,
      series: [{ label: tx("95% range", "95% rozpětí"), color: "var(--c1)", dots: true, noLine: true, points: bins.map((b) => ({ x: b.p, y: b.p, lo: b.p_lo, hi: b.p_hi, n: b.n })) }],
      tipFmt: (v, p) => `${pct(p.lo)}–${pct(p.hi)} · n ${p.n}`, tipHead: (xv) => `${tx("Estimate", "Odhad")} ${pct(xv)}`, aria: "stability" });
    card.appendChild(h("p", { class: "note", text: tx(
      `Mean absolute difference between a refitted and the published estimate: ${num(stab.mape * 100, 1)} percentage points (Riley & Collins 2023). The calculator shows this range under the headline.`,
      `Průměrný absolutní rozdíl mezi odhadem přetrénovaného a publikovaného modelu: ${num(stab.mape * 100, 1)} procentního bodu (Riley & Collins 2023). Kalkulačka toto rozpětí ukazuje pod hlavním odhadem.`) }));
    return card;
  }

  function subgroupCard(tg, st) {
    const card = h("div", { class: "card card-pad" });
    const rows = (DATA.subgroups || []).filter((r) => r.target === tg && r.stage === st);
    card.appendChild(h("div", { class: "chart-head" }, h("div", null,
      h("div", { class: "chart-title", text: t().subTitle }), h("div", { class: "chart-sub", text: t().subSub }))));
    if (!rows.length) { card.appendChild(h("p", { class: "note", text: "–" })); return card; }
    const dimName = { sex: t().subSex, age: t().subAge, circulation: t().subCirc, ivt: t().kIvt, route: t().subRoute };
    const grpName = (g) => (lang === "cs" ? ({ Women: "Ženy", Men: "Muži", Anterior: "Přední povodí", Posterior: "Zadní povodí", "No IVT": "Bez IVT", Direct: "Přímo do FNO" }[g] || g) : g);
    const tbl = h("table", { class: "data" },
      h("thead", null, h("tr", null, [t().subGroup, "n", lang === "cs" ? "Události" : "Events", "AUROC", t().observed, t().predicted].map((x) => h("th", { scope: "col", text: x })))),
      h("tbody", null, rows.map((r) => h("tr", null,
        h("th", { scope: "row", text: `${dimName[r.dim] || r.dim}: ${grpName(r.group)}` }),
        h("td", { text: r.suppressed ? t().suppressed : num(r.n) }),
        h("td", { text: r.suppressed ? "–" : num(r.events) }),
        h("td", { text: r.suppressed ? "–" : num(r.auroc, 3) }),
        h("td", { text: r.suppressed ? "–" : pct(r.observed) }),
        h("td", { text: r.suppressed ? "–" : pct(r.predicted) })))));
    card.appendChild(h("div", { class: "table-scroll" }, tbl));
    card.appendChild(h("p", { class: "note", text: t().subNote }));
    return card;
  }

  // ================================================================== about
  function renderAbout() {
    const root = $("#view-about");
    root.replaceChildren();
    const nAll = E.kpis.n;
    const st = V.stages;
    const auc = (s) => num(st[s].targets.good_outcome.temporal.LightGBM.auroc, 2);
    const en = lang !== "cs";
    const P = (txt) => h("p", { text: txt });
    const prose = h("div", { class: "prose" },
      h("h2", { style: "font-size:24px", text: t().aboutTitle }),
      P(en
        ? `MTOP estimates the chance that a patient treated with mechanical thrombectomy for acute ischaemic stroke will be functionally independent (modified Rankin Scale 0–2) 90 days later, and the chance of death. The estimate is updated at four points of care: before the procedure, at its end, at 24 hours and on day 7. Each update uses only what is known at that point.`
        : `MTOP odhaduje šanci, že pacient po mechanické trombektomii pro akutní ischemickou CMP bude za 90 dní funkčně nezávislý (modifikovaná Rankinova škála 0–2), a riziko úmrtí. Odhad se aktualizuje ve čtyřech okamžicích péče: před výkonem, na jeho konci, po 24 hodinách a 7. den. Každá aktualizace používá jen údaje známé v daném okamžiku.`),
      h("h3", { text: en ? "Data" : "Data" }),
      P(en
        ? `The models were built from the prospective thrombectomy registry of University Hospital Ostrava (FNO). Of ${num(E.flow[0].n)} registry entries, ${num(nAll)} adults who received endovascular treatment and have a recorded 90-day mRS were analysed (${E.period[0]}–${E.period[1]}). Diagnostic angiography without treatment was excluded. In the registry, an NIHSS of 43 marks death and 99 marks "not assessable"; these codes are never used as scores.`
        : `Modely vycházejí z prospektivního registru trombektomií Fakultní nemocnice Ostrava (FNO). Z ${num(E.flow[0].n)} záznamů registru bylo analyzováno ${num(nAll)} dospělých pacientů po endovaskulární léčbě se známým mRS za 90 dní (${E.period[0]}–${E.period[1]}). Diagnostická angiografie bez léčby byla vyřazena. NIHSS 43 v registru označuje úmrtí a 99 „nehodnotitelné“; tyto kódy se nikdy nepoužívají jako skóre.`),
      h("h3", { text: en ? "Models" : "Modely" }),
      P(en
        ? `Each stage has its own gradient-boosted tree model (LightGBM) with monotone constraints, so that, for example, a longer time to puncture or a higher NIHSS can never raise the predicted chance of independence. Hyperparameters were tuned by cross-validation on the development cohort only. Logistic regression (linear and with restricted cubic splines), XGBoost, a multilayer perceptron, the 7-item FNO bedside score from the earlier registry analysis, published scores (THRIVE, SPAN-100, HIAT2, PRE, NIHSS alone) and refitted published predictor sets (THRIVE-c, MR PREDICTS@24H) serve as comparators. Explanations are exact TreeSHAP values computed in your browser.`
        : `Každá fáze má vlastní model gradientně boostovaných stromů (LightGBM) s monotónními omezeními, takže například delší čas do punkce nebo vyšší NIHSS nikdy nezvýší predikovanou šanci na nezávislost. Hyperparametry byly laděny křížovou validací pouze na vývojové kohortě. Jako srovnání slouží logistická regrese (lineární i s restrikovanými kubickými splajny), XGBoost, vícevrstvý perceptron, 7položkové bedside skóre FNO z dřívější analýzy registru, publikovaná skóre (THRIVE, SPAN-100, HIAT2, PRE, samotné NIHSS) a přepočtené publikované sady prediktorů (THRIVE-c, MR PREDICTS@24H). Vysvětlení jsou přesné hodnoty TreeSHAP počítané ve vašem prohlížeči.`),
      h("h3", { text: en ? "Validation" : "Validace" }),
      P(en
        ? `Temporal validation on patients treated in 2021–2025 gave an AUROC for mRS 0–2 of ${auc("admission")} before thrombectomy, ${auc("procedure")} at the end of the procedure, ${auc("h24")} at 24 hours and ${auc("d7")} on day 7. Outcomes in recent years were worse than the development data predicted, so the published models are refitted on all years and their intercepts shifted to the 2021–2025 outcome level.`
        : `Časová validace na pacientech léčených v letech 2021–2025 ukázala AUROC pro mRS 0–2 ${auc("admission")} před trombektomií, ${auc("procedure")} na konci výkonu, ${auc("h24")} po 24 hodinách a ${auc("d7")} 7. den. Výsledky v posledních letech byly horší, než vývojová data předpovídala, proto jsou zveřejněné modely přetrénovány na všech letech a jejich intercept je posunut na úroveň výsledků 2021–2025.`),
      h("h3", { text: en ? "Limits" : "Omezení" }),
      h("ul", null, (en ? [
        "Single centre; performance elsewhere is unknown until externally validated.",
        "ASPECTS was recorded from 2016 and anaesthesia type from about 2012; earlier patients contribute through missing-value handling.",
        "Since 2021 about half of patients lack a 24-hour NIHSS and many lack day-7 data, so the 24-h and day-7 validation rests on a selected subset.",
        "The day-7 mortality estimate rests on few late deaths and is imprecise.",
        "Collateral status, core volume and number of passes are not in the registry and cannot be used.",
      ] : [
        "Jedno centrum; výkon jinde není znám, dokud nebude externě validován.",
        "ASPECTS je evidováno od roku 2016 a typ anestezie přibližně od roku 2012; starší pacienti přispívají přes zpracování chybějících hodnot.",
        "Od roku 2021 asi polovině pacientů chybí NIHSS za 24 h a mnohým údaje ze 7. dne, takže validace fází 24 h a 7. den stojí na vybrané podskupině.",
        "Odhad mortality 7. den vychází z malého počtu pozdních úmrtí a je nepřesný.",
        "Kolaterální oběh, objem jádra infarktu a počet průchodů registr neobsahuje.",
      ]).map((x) => h("li", { text: x }))),
      h("h3", { text: en ? "Privacy" : "Ochrana údajů" }),
      P(en
        ? `This page contains no patient records. It carries only the trained trees, in which every leaf summarises at least 20 patients, and aggregate statistics in which groups under ${E.min_cell} patients are hidden. Values typed into the calculator stay in your browser and are not stored or sent.`
        : `Tato stránka neobsahuje žádné záznamy pacientů. Obsahuje pouze natrénované stromy, v nichž každý list shrnuje nejméně 20 pacientů, a souhrnné statistiky, ve kterých jsou skryty skupiny pod ${E.min_cell} pacientů. Hodnoty zadané do kalkulačky zůstávají ve vašem prohlížeči a nikam se neukládají ani neodesílají.`));
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
    add(en ? "Intended use" : "Určení", h("span", { text: en ? "Research and teaching. Not a medical device; not for individual treatment decisions." : "Výzkum a výuka. Není zdravotnickým prostředkem; není určeno k rozhodování o léčbě jednotlivce." }));
    const aside = h("aside", { class: "about-side" }, h("div", { class: "card card-pad" }, meta));
    if (DATA.qr_svg && SITE.site_url) {
      const qr = h("div", { class: "qr", html: DATA.qr_svg, role: "img", "aria-label": `QR: ${SITE.site_url}` });
      aside.appendChild(h("div", { class: "card card-pad qr-card" }, qr,
        h("div", null, h("div", { class: "chart-title", text: en ? "Open on your phone" : "Otevřít v telefonu" }), h("div", { class: "copyable", text: SITE.site_url.replace(/^https?:\/\//, "") }))));
    }
    root.appendChild(h("div", { class: "about-grid" }, prose, aside));
    if (window.MTOPApp.dictionaryBlock) root.appendChild(window.MTOPApp.dictionaryBlock());
  }

  // ------------------------------------------------------------------ shared API for the other views
  window.MTOPApp = {
    DATA, B, V, E, X, SITE, M, XE, C, h, S, STAGES, STAGE, FEAT, TARGETS, MODEL_NAMES, LEARNERS, MRS_COLORS,
    t, L, tx, lang: () => lang, num, pct, pct100, signed, stageLabel, idx, mname, statusChip,
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
