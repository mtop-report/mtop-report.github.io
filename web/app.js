/* MTOP application: staged calculator, registry explorer, validation, about. */
(function () {
  "use strict";
  const DATA = JSON.parse(document.getElementById("mtop-data").textContent);
  const B = DATA.bundle;
  const V = DATA.validation;
  const E = DATA.explorer;
  const SITE = DATA.site || {};
  const M = window.MTOPModel;
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

  function render() {
    document.documentElement.lang = lang;
    app.replaceChildren(header(), mainEl(), footer());
    renderView();
  }

  function header() {
    const tabs = [["calc", t().tabCalc], ["registry", t().tabRegistry], ["validation", t().tabValidation], ["about", t().tabAbout]];
    const nav = h("nav", { class: "tabs", role: "tablist", "aria-label": "MTOP" },
      tabs.map(([id, label]) => h("button", {
        class: "tab", role: "tab", type: "button", id: `tab-${id}`, "aria-selected": String(S.view === id), "aria-controls": `view-${id}`,
        onclick: () => { S.view = id; try { history.replaceState(null, "", `#${id}`); } catch (e) { /* sandboxed frame */ } render(); window.scrollTo(0, 0); },
      }, label)));
    const langSw = h("div", { class: "lang", role: "group", "aria-label": "Language" },
      [["en", "EN"], ["cs", "CZ"]].map(([code, lbl]) => h("button", {
        type: "button", "aria-pressed": String(lang === code), onclick: () => { lang = code; store.set("mtop.lang", code); render(); },
      }, lbl)));
    return h("header", { class: "top" }, h("div", { class: "top-inner" },
      h("div", { class: "brand" }, h("span", { class: "brand-mark", html: "MT<span>O</span>P" }), h("span", { class: "brand-sub", text: t().brandSub })),
      nav, langSw));
  }

  function mainEl() {
    return h("main", { class: "wrap", id: "main" },
      h("section", { class: "view", id: "view-calc", role: "tabpanel", "aria-labelledby": "tab-calc", hidden: S.view !== "calc" }),
      h("section", { class: "view", id: "view-registry", role: "tabpanel", "aria-labelledby": "tab-registry", hidden: S.view !== "registry" }),
      h("section", { class: "view", id: "view-validation", role: "tabpanel", "aria-labelledby": "tab-validation", hidden: S.view !== "validation" }),
      h("section", { class: "view", id: "view-about", role: "tabpanel", "aria-labelledby": "tab-about", hidden: S.view !== "about" }));
  }

  function footer() {
    const parts = [h("span", { text: `${t().footer} · v${B.version} · ${B.generated}` })];
    if (SITE.repo_url) parts.push(h("a", { href: SITE.repo_url, target: "_blank", rel: "noopener", text: "GitHub" }));
    return h("footer", { class: "foot" }, h("div", { class: "wrap" }, ...parts));
  }

  function renderView() {
    if (S.view === "calc") renderCalc();
    if (S.view === "registry") renderRegistry();
    if (S.view === "validation") renderValidation();
    if (S.view === "about") renderAbout();
  }

  // ================================================================== calculator
  let readoutEl = null;
  let stickyEl = null;

  function renderCalc() {
    const root = $("#view-calc");
    root.replaceChildren();
    root.appendChild(h("div", { class: "notice", role: "note", html: t().notice }));
    const left = h("div", { class: "inputs" });
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
    setupSticky();
  }

  function setupSticky() {
    if (typeof IntersectionObserver === "undefined") return;
    const head = $(".headline", readoutEl);
    if (!head) return;
    const io = new IntersectionObserver((ents) => {
      for (const e of ents) stickyEl.classList.toggle("show", !e.isIntersecting && S.view === "calc");
    });
    io.observe(head);
  }

  function stageCard(st) {
    const i = idx(st);
    const locked = i > idx(S.current);
    const feats = B.features.filter((f) => f.stage === st && f.type !== "derived");
    const card = h("section", { class: `card stage${locked ? " locked" : ""}`, id: `stage-${st}`, "aria-labelledby": `stage-${st}-h` });
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
    card.appendChild(body);
    card.appendChild(h("div", { class: "stage-locked-note" },
      h("span", { text: t().lockedNote }),
      h("button", { class: "btn btn-primary", type: "button", onclick: () => setCurrent(st) }, `${t().addStage}: ${stageLabel(st)}`)));
    return card;
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
        refresh();
      };
      const clamp = (v) => Math.min(f.max, Math.max(f.min, v));
      range.addEventListener("input", () => { setInput(f.key, Number(range.value)); box.value = range.value; refresh(); });
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
      const seg = h("div", { class: `seg${isCat && opts.length > 3 ? " cols-2" : ""}`, role: "group", "aria-labelledby": `lbl-${f.key}` });
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
    readoutEl.replaceChildren(headlineCard(vs, pv), trajectoryCard(preds), explainCard(vs, pv), whatifCard(vs, x));
    if (stickyEl) $(".inner", stickyEl).replaceChildren(
      h("span", { html: t().sticky(pct(pv.good.p), pct(pv.death.p)) }),
      h("span", { class: "delta", text: stageLabel(vs) }));
  }

  function headlineCard(st, pv) {
    const g = pv.good.p, d = pv.death.p;
    const bin = realityBin(st, g);
    const big = h("div", { class: "big" }, num(g * 100, 0), h("small", { text: lang === "cs" ? " %" : "%" }));
    const card = h("div", { class: "card headline" },
      h("div", { class: "headline-top" },
        h("div", null, h("div", { class: "eyebrow", text: t().estimateAt(stageLabel(st)) }), big, h("div", { class: "big-label", text: t().headline })),
        h("div", { class: "side-stat" }, h("div", { class: "v", text: pct(d) }), h("div", { class: "l", text: t().deathLabel }),
          st === "d7" ? h("div", { class: "l", style: "color:var(--muted)", text: t().lateDeaths }) : null)));
    if (bin) {
      card.appendChild(h("div", { class: "reality", html: t().reality(pct(bin.lo), pct(bin.hi), pct(bin.obs), bin.n) }));
    }
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
    gBlock.appendChild(h("div", { class: "controls" }, h("label", { for: "reg-dim" }, t().groupBy), dimSel));
    const dim = E.dimensions[S.regDim];
    const overallLabel = lang === "cs" ? "Celý soubor" : "Whole cohort";
    const rows = [{ label: overallLabel, n: E.overall.n, segs: E.overall.dist }].concat(dim.groups.map((g) => ({
      label: L(g.label), n: g.n, segs: g.dist, suppressed: g.suppressed || !g.dist,
    })));
    const legend = [0, 1, 2, 3, 4, 5, 6].map((m) => ({ label: `mRS ${m}`, color: MRS_COLORS[m] }));
    const tableRows = rows.map((r) => [r.label, r.suppressed ? t().suppressed : r.n].concat(r.suppressed || !r.segs ? Array(7).fill(null) : r.segs.map((v) => pct100(v, 1))));
    const fg = C.figure(gBlock, {
      title: t().grottaTitle, sub: `${t().grottaSub} ${L(dim.label)}.`, legend,
      table: { columns: [t().variable, "n"].concat([0, 1, 2, 3, 4, 5, 6].map((m) => `mRS ${m}`)), rows: tableRows },
      labels: { table: t().showTable, chart: t().showChart },
    });
    C.grotta(fg.body, { rows, colors: MRS_COLORS, inkOn: MRS_INK, segLabels: legend.map((l) => l.label), fmt: (v) => pct100(v, 1), suppressedText: lang === "cs" ? "méně než 10 pacientů nebo méně než 30 pro rozložení" : "too few patients to show", nLabel: "n", aria: t().grottaTitle });
    root.appendChild(gBlock);

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
      C.lineChart(f.body, { height: 150, x: { type: "cat", cats: yrs }, y: { domain: dom, ticks: ticksArr }, series: [{ label: title, color: "var(--c1)", points: tr.map((r, i) => ({ x: i, y: r[key] })) }], tipFmt: fmt, aria: title });
      sm.appendChild(card);
    };
    single(t().tPatients, "n", [0, 150], (v) => num(v), [0, 50, 100, 150]);
    single(t().tAge, "age", [55, 80], (v) => num(v), [55, 65, 75]);
    single(t().tTici, "tici2b3", [0, 100], (v) => pct100(v), [0, 50, 100]);
    single(t().tOtp, "otp", [0, 300], (v) => `${num(v)} min`, [0, 100, 200, 300]);
    single(t().tGa, "ga", [0, 100], (v) => pct100(v), [0, 50, 100]);
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
  };
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
    const learners = ["LightGBM", "Logistic regression", "XGBoost", "MLP"];
    const aSeries = learners.map((k, i) => ({
      label: mname(k), color: MODEL_COLORS[i], dots: true, offset: (i - 1.5) * 6,
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
    const rSeries = modelKeys.map((k, i) => ({ label: mname(k), color: MODEL_COLORS[i % MODEL_COLORS.length], width: k === "LightGBM" ? 2.5 : 1.5, points: vt.curves.roc[k].map((p) => ({ x: p.fpr, y: p.tpr })) }));
    const fr = C.figure(rCard, { title: t().rocTitle, sub: L(STAGE[S.valStage].title), legend: rSeries.map((s) => ({ label: s.label, color: s.color, type: "line" })),
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
    row2.appendChild(subgroupCard(tg, S.valStage));
    root.appendChild(row2);

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
    const dSeries = [
      { label: "LightGBM", color: "var(--c1)", points: dca.map((r) => ({ x: r.t, y: r.model })) },
      { label: t().treatAll, color: "var(--ink-2)", width: 1.5, points: dca.map((r) => ({ x: r.t, y: r.all })) },
    ];
    const lo = Math.min(-0.05, ...dca.map((r) => Math.max(r.all, -0.1)));
    const hi = Math.max(...dca.map((r) => Math.max(r.model, r.all))) + 0.03;
    const fd = C.figure(dcCard, { title: t().dcaTitle, sub: t().dcaSub, legend: dSeries.map((s) => ({ label: s.label, color: s.color, type: "line" })).concat([{ label: t().treatNone, color: "var(--line-2)", type: "line" }]),
      table: { columns: [t().threshold, "LightGBM", t().treatAll, t().treatNone], rows: dca.map((r) => [pct(r.t), num(r.model, 3), num(r.all, 3), "0"]) },
      labels: { table: t().showTable, chart: t().showChart } });
    C.lineChart(fd.body, { height: 240, x: { type: "lin", domain: [0.05, 0.8], fmt: (v) => pct(v), title: t().threshold, ticks: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] }, y: { domain: [lo, hi], fmt: (v) => num(v, 2), title: t().netBenefit },
      series: dSeries.map((s) => ({ ...s, points: s.points.map((p) => ({ x: p.x, y: Math.max(p.y, lo) })) })), zeroLine: true,
      tipFmt: (v) => num(v, 3), tipHead: (xv) => `${t().threshold} ${pct(xv)}`, aria: t().dcaTitle });
    row3.appendChild(dcCard);
    root.appendChild(row3);
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
        ? `Each stage has its own gradient-boosted tree model (LightGBM) with monotone constraints, so that, for example, a longer time to puncture or a higher NIHSS can never raise the predicted chance of independence. Hyperparameters were tuned by cross-validation on the development cohort only. Logistic regression, XGBoost, a multilayer perceptron and published scores (THRIVE, SPAN-100, NIHSS alone) serve as comparators. Explanations are exact TreeSHAP values computed in your browser.`
        : `Každá fáze má vlastní model gradientně boostovaných stromů (LightGBM) s monotónními omezeními, takže například delší čas do punkce nebo vyšší NIHSS nikdy nezvýší predikovanou šanci na nezávislost. Hyperparametry byly laděny křížovou validací pouze na vývojové kohortě. Jako srovnání slouží logistická regrese, XGBoost, vícevrstvý perceptron a publikovaná skóre (THRIVE, SPAN-100, samotné NIHSS). Vysvětlení jsou přesné hodnoty TreeSHAP počítané ve vašem prohlížeči.`),
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
    if (SITE.repo_url) add(en ? "Code" : "Kód", h("a", { href: SITE.repo_url, target: "_blank", rel: "noopener", text: SITE.repo_url.replace(/^https?:\/\//, "") }));
    add(en ? "Intended use" : "Určení", h("span", { text: en ? "Research and teaching. Not a medical device; not for individual treatment decisions." : "Výzkum a výuka. Není zdravotnickým prostředkem; není určeno k rozhodování o léčbě jednotlivce." }));
    root.appendChild(h("div", { class: "about-grid" }, prose, h("aside", { class: "card card-pad" }, meta)));
  }

  // ------------------------------------------------------------------ boot
  const hash = (location.hash || "").replace("#", "");
  if (["calc", "registry", "validation", "about"].includes(hash)) S.view = hash;
  render();
})();
