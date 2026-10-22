/* MTOP browser models
 * Tree ensembles (LightGBM and XGBoost) with exact TreeSHAP attributions (Lundberg et al.,
 * Nat Mach Intell 2020), mirroring the libraries' own decision rules so results match
 * LightGBM `pred_contrib=True` and XGBoost `pred_contribs=True`; the penalised logistic
 * regressions (linear, with restricted cubic splines, and with interactions of centred inputs),
 * whose exact Shapley values are the coefficient times the standardised value of each term (an
 * interaction split half and half); the explainable boosting machine (EBM), additive by
 * construction (pair terms split half and half); and the stacking, whose attribution is the
 * meta-model's weighted sum of its base models' attributions.
 * No patient data is needed or sent anywhere: everything runs in the browser.
 */
(function (root) {
  "use strict";

  const MISSING_ZERO = 1;
  const MISSING_NAN = 2;
  const K_ZERO = 1.0000000180025095e-35; // LightGBM's kZeroThreshold is the float 1e-35f
  const fround = Math.fround || ((v) => v);

  function treeDepth(t) {
    let best = 0;
    const stack = [[0, 0]];
    while (stack.length) {
      const [n, d] = stack.pop();
      if (t.f[n] < 0) { if (d > best) best = d; continue; }
      stack.push([t.l[n], d + 1], [t.r[n], d + 1]);
    }
    return best;
  }

  // TreeSHAP's base value: the cover-weighted mean leaf value (counts for LightGBM, hessian cover for XGBoost)
  function expectedValue(t) {
    if (t.f.length === 1) return t.v[0];
    let e = 0, total = 0;
    for (let i = 0; i < t.f.length; i++) if (t.f[i] < 0) { e += t.c[i] * t.v[i]; total += t.c[i]; }
    return total ? e / total : 0;
  }

  /** Prepare a tree-ensemble spec (adds depth, expected values, path buffers). */
  function prepare(spec) {
    if (spec._ready) return spec;
    let base = spec.base_margin || 0;
    let maxDepth = 0;
    const lt = spec.op === "lt";
    for (const t of spec.trees) {
      t.lt = lt;
      t.depth = treeDepth(t);
      t.ev = expectedValue(t);
      base += t.ev;
      if (t.depth > maxDepth) maxDepth = t.depth;
    }
    const maxd = maxDepth + 3;
    const size = ((maxd + 1) * (maxd + 2)) / 2 + maxd + 2;
    spec._path = {
      fi: new Int32Array(size), zf: new Float64Array(size),
      of: new Float64Array(size), pw: new Float64Array(size),
    };
    spec.base = base;
    spec._ready = true;
    return spec;
  }

  function decide(t, node, x) {
    let v = x[t.f[node]];
    if (t.lt) { // XGBoost: missing goes to the default child, otherwise float32(x) < threshold goes left
      if (Number.isNaN(v)) return t.d[node] ? t.l[node] : t.r[node];
      return fround(v) < t.t[node] ? t.l[node] : t.r[node];
    }
    // LightGBM stores only |v| > 1e-35 when it reads a row, so tinier values act as exact zero
    if (!Number.isNaN(v) && v >= -K_ZERO && v <= K_ZERO) v = 0.0;
    const mt = t.m[node];
    if (Number.isNaN(v) && mt !== MISSING_NAN) v = 0.0;
    if ((mt === MISSING_ZERO && v >= -K_ZERO && v <= K_ZERO) || (mt === MISSING_NAN && Number.isNaN(v))) {
      return t.d[node] ? t.l[node] : t.r[node];
    }
    return v <= t.t[node] ? t.l[node] : t.r[node];
  }

  function leafValue(t, x) {
    let n = 0;
    while (t.f[n] >= 0) n = decide(t, n, x);
    return t.v[n];
  }

  /** Raw score (log-odds) of a tree ensemble, without recalibration. */
  function rawTrees(spec, x) {
    prepare(spec);
    let s = spec.base_margin || 0;
    for (const t of spec.trees) s += leafValue(t, x);
    return s;
  }

  const sigmoid = (z, k) => 1 / (1 + Math.exp(-(k || 1) * z));

  // ---------------------------------------------------------------- TreeSHAP
  function extendPath(P, o, d, zero, one, feat) {
    P.fi[o + d] = feat; P.zf[o + d] = zero; P.of[o + d] = one; P.pw[o + d] = d === 0 ? 1 : 0;
    for (let i = d - 1; i >= 0; i--) {
      P.pw[o + i + 1] += (one * P.pw[o + i] * (i + 1)) / (d + 1);
      P.pw[o + i] = (zero * P.pw[o + i] * (d - i)) / (d + 1);
    }
  }

  function unwindPath(P, o, d, k) {
    const one = P.of[o + k];
    const zero = P.zf[o + k];
    let next = P.pw[o + d];
    for (let i = d - 1; i >= 0; --i) {
      if (one !== 0) {
        const tmp = P.pw[o + i];
        P.pw[o + i] = (next * (d + 1)) / ((i + 1) * one);
        next = tmp - (P.pw[o + i] * zero * (d - i)) / (d + 1);
      } else {
        P.pw[o + i] = (P.pw[o + i] * (d + 1)) / (zero * (d - i));
      }
    }
    for (let i = k; i < d; ++i) {
      P.fi[o + i] = P.fi[o + i + 1]; P.zf[o + i] = P.zf[o + i + 1]; P.of[o + i] = P.of[o + i + 1];
    }
  }

  function unwoundPathSum(P, o, d, k) {
    const one = P.of[o + k];
    const zero = P.zf[o + k];
    let next = P.pw[o + d];
    let total = 0;
    if (one !== 0) {
      for (let i = d - 1; i >= 0; --i) {
        const tmp = next / ((i + 1) * one);
        total += tmp;
        next = P.pw[o + i] - tmp * zero * (d - i);
      }
    } else {
      for (let i = d - 1; i >= 0; --i) total += P.pw[o + i] / (zero * (d - i));
    }
    return total * (d + 1);
  }

  function recurse(t, x, phi, P, node, d, parentOff, pZero, pOne, pFeat) {
    const o = parentOff + d + 1;
    for (let i = 0; i <= d; i++) {
      P.fi[o + i] = P.fi[parentOff + i]; P.zf[o + i] = P.zf[parentOff + i];
      P.of[o + i] = P.of[parentOff + i]; P.pw[o + i] = P.pw[parentOff + i];
    }
    extendPath(P, o, d, pZero, pOne, pFeat);
    if (t.f[node] < 0) {
      for (let i = 1; i <= d; ++i) {
        const w = unwoundPathSum(P, o, d, i);
        phi[P.fi[o + i]] += w * (P.of[o + i] - P.zf[o + i]) * t.v[node];
      }
      return;
    }
    const split = t.f[node];
    const hot = decide(t, node, x);
    const cold = hot === t.l[node] ? t.r[node] : t.l[node];
    const w = t.c[node];
    const hotZero = t.c[hot] / w;
    const coldZero = t.c[cold] / w;
    let inZero = 1;
    let inOne = 1;
    let k = 0;
    for (; k <= d; ++k) if (P.fi[o + k] === split) break;
    if (k !== d + 1) {
      inZero = P.zf[o + k];
      inOne = P.of[o + k];
      unwindPath(P, o, d, k);
      d -= 1;
    }
    recurse(t, x, phi, P, hot, d + 1, o, hotZero * inZero, inOne, split);
    recurse(t, x, phi, P, cold, d + 1, o, coldZero * inZero, 0, split);
  }

  function shapTrees(spec, x, nCols) {
    prepare(spec);
    const phi = new Float64Array(nCols);
    const P = spec._path;
    for (const t of spec.trees) {
      if (t.f.length === 1) continue;
      P.fi[0] = -1; P.zf[0] = 0; P.of[0] = 0; P.pw[0] = 0;
      recurse(t, x, phi, P, 0, 0, 0, 1, 1, -1);
    }
    return { base: spec.base, phi: Array.from(phi) };
  }

  // ---------------------------------------------------------------- regressions
  const miss = (v) => v === null || v === undefined || v === "" || (typeof v === "number" && Number.isNaN(v));

  /** Restricted cubic spline terms (Harrell) of x for knots t1 < ... < tk: k - 2 values, NaN if x is missing.
   *  Same arithmetic as src/mtop/splines.py rcs_terms. */
  function rcsTerms(x, t) {
    const k = t.length;
    if (miss(x) || Number.isNaN(Number(x))) return Array.from({ length: k - 2 }, () => NaN);
    const v = Number(x);
    const p3 = (z) => Math.pow(Math.max(z, 0), 3);
    const d = (t[k - 1] - t[0]) * (t[k - 1] - t[0]);
    const out = [];
    for (let j = 0; j < k - 2; j++) {
      out.push((p3(v - t[j]) - (p3(v - t[k - 2]) * (t[k - 1] - t[j])) / (t[k - 1] - t[k - 2])
        + (p3(v - t[k - 1]) * (t[k - 2] - t[j])) / (t[k - 1] - t[k - 2])) / d);
    }
    return out;
  }

  /** Expanded row of the spline regression: base columns, then each knotted variable's terms;
   *  `owner[j]` is the base column each expanded column belongs to. */
  function splineExpand(spec, base) {
    const row = base.slice();
    const owner = base.map((_, i) => i);
    for (const [v, knots] of Object.entries(spec.knots)) {
      const i = spec.base_columns.indexOf(v);
      for (const z of rcsTerms(base[i], knots)) { row.push(z); owner.push(i); }
    }
    // interactions (LR + interactions): products of centred inputs, owned half by each input
    for (const [ia, ib, ca, cb] of spec.products || []) {
      const a = base[ia], b = base[ib];
      row.push(miss(a) || miss(b) ? NaN : (Number(a) - ca) * (Number(b) - cb));
      owner.push([ia, ib]);
    }
    return { row, owner };
  }

  // ---------------------------------------------------------------- EBM (explainable boosting)
  /** Interval of v among the cut points: 0 for a missing value, else 1 + the number of cut points <= v
   *  (a value equal to a cut point belongs to the upper interval, as in interpret). */
  function ebmBin(cuts, v) {
    if (miss(v) || Number.isNaN(Number(v))) return 0;
    const x = Number(v);
    let lo = 0, hi = cuts.length;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (cuts[mid] <= x) lo = mid + 1; else hi = mid; }
    return lo + 1;
  }

  /** Score of every term for one model row (main terms: one input; pair terms: two inputs). */
  function ebmTerms(spec, x) {
    return spec.terms.map((t) => {
      if (t.f.length === 1) { const k = ebmBin(t.cuts[0], x[t.f[0]]); return k === 0 ? t.miss : t.s[k - 1]; }
      return t.s[ebmBin(t.cuts[0], x[t.f[0]])][ebmBin(t.cuts[1], x[t.f[1]])];
    });
  }

  /** Penalised logistic regression pipeline (median imputation + indicators, scaling): per-coefficient
   *  terms of the linear predictor and the intercept. `owner` maps each input column to a base column. */
  function lrTerms(spec, x, owner) {
    const filled = x.map((v, i) => (miss(v) ? spec.medians[i] : v));
    const ind = spec.indicators.map((j) => (miss(x[j]) ? 1 : 0));
    const z = filled.concat(ind);
    const own = owner.concat(spec.indicators.map((j) => owner[j]));
    const terms = z.map((v, i) => ((v - spec.mean[i]) / spec.scale[i]) * spec.coef[i]);
    return { terms, own, intercept: spec.intercept + (spec.shift || 0) };
  }

  function lrRaw(spec, x) {
    const t = lrTerms(spec, x, x.map((_, i) => i));
    return t.terms.reduce((a, b) => a + b, t.intercept);
  }

  // ---------------------------------------------------------------- one interface for every learner
  /** Log-odds (without the recent-era shift) of a published learner for a model row. */
  function raw(L, x) {
    if (L.kind === "lgbm" || L.kind === "xgb") return rawTrees(L, x);
    if (L.kind === "lr") return lrRaw(L.spec, x);
    if (L.kind === "spline") return lrRaw(L.spec, splineExpand(L.spec, x).row);
    if (L.kind === "ebm") return ebmTerms(L.spec, x).reduce((a, b) => a + b, L.spec.intercept);
    // stacking: the meta-model on the base models' log-odds
    if (L.kind === "stack") return L.bases.reduce((s, b, k) => s + L.meta.coef[k] * raw(b, x), L.meta.intercept);
    throw new Error(`unknown learner ${L.kind}`);
  }

  /** Probability; `recent` (default true) applies the recent-era intercept. */
  function predict(L, x, opts) {
    const r = raw(L, x);
    const shift = opts && opts.recent === false ? 0 : L.intercept_recent || 0;
    return { raw: r, logit: r + shift, p: sigmoid(r + shift, L.sigmoid) };
  }

  /** Exact attributions per model column on the log-odds scale: base + sum(phi) = raw score. */
  function shap(L, x) {
    if (L.kind === "lgbm" || L.kind === "xgb") return shapTrees(L, x, x.length);
    const phi = new Array(x.length).fill(0);
    // a term owned by two inputs (interaction, EBM pair) is split half and half
    const add = (o, v) => { if (Array.isArray(o)) { phi[o[0]] += v / 2; phi[o[1]] += v / 2; } else phi[o] += v; };
    if (L.kind === "ebm") {
      ebmTerms(L.spec, x).forEach((v, i) => { const f = L.spec.terms[i].f; add(f.length === 1 ? f[0] : f, v); });
      return { base: L.spec.intercept, phi };
    }
    if (L.kind === "stack") {
      let base = L.meta.intercept;
      L.bases.forEach((b, k) => {
        const s = shap(b, x), w = L.meta.coef[k];
        base += w * s.base;
        s.phi.forEach((v, i) => { phi[i] += w * v; });
      });
      return { base, phi };
    }
    let t;
    if (L.kind === "lr") t = lrTerms(L.spec, x, x.map((_, i) => i));
    else {
      const e = splineExpand(L.spec, x);
      t = lrTerms(L.spec, e.row, e.owner);
    }
    t.terms.forEach((v, i) => add(t.own[i], v));
    return { base: t.intercept, phi };
  }

  /** Attributions summed per predictor (one-hot columns grouped). */
  function explain(L, x, columnFeature) {
    const s = shap(L, x);
    const byFeature = {};
    columnFeature.forEach((f, i) => { byFeature[f] = (byFeature[f] || 0) + s.phi[i]; });
    return { base: s.base, phi: s.phi, byFeature };
  }

  // ---------------------------------------------------------------- inputs -> model row
  /** Derived predictors, as src/mtop/features.py derived(). */
  function derivedValue(key, val) {
    const n0 = val("nihss_adm"), n24 = val("nihss_24h");
    switch (key) {
      case "nihss_change_24h": return n24 - n0;
      case "eni_24h":
        if (Number.isNaN(n24)) return NaN;
        if (n24 <= 1) return 1;
        return Number.isNaN(n0) ? NaN : (n0 - n24 >= 8 ? 1 : 0);
      case "night_onset": { const h = val("onset_hour"); return Number.isNaN(h) ? NaN : (h >= 22 || h < 6 ? 1 : 0); }
      case "ivt_to_puncture": return val("onset_to_puncture") - val("onset_to_ivt");
      case "pulse_pressure": return val("sbp") - val("dbp");
      case "hyperglycemia": { const g = val("glucose"); return Number.isNaN(g) ? NaN : (g > 7.8 ? 1 : 0); }
      default: return NaN;
    }
  }

  /** Build the model row for a stage from calculator inputs keyed by predictor. Inputs that need IV
   *  thrombolysis are "not applicable" without it: onset-to-IVT becomes missing (as in the registry)
   *  and "thrombolysis in another hospital" becomes no. */
  function buildRow(columns, featureMeta, inputs) {
    const meta = {};
    for (const f of featureMeta) meta[f.key] = f;
    const noIvt = inputs.ivt === 0;
    const val = (k) => {
      if (noIvt && k === "onset_to_ivt") return NaN;
      if (noIvt && k === "ivt_external") return 0;
      const v = inputs[k];
      return v === null || v === undefined || v === "" || Number.isNaN(v) ? NaN : Number(v);
    };
    return columns.map((col) => {
      const [key, cat] = col.split("__");
      const f = meta[key];
      if (cat !== undefined) {
        const v = inputs[key];
        if (v === null || v === undefined || v === "") return NaN;
        // an option without a model term yet (too few patients, e.g. ACA) is not used by the model
        const o = f && f.options ? f.options.find((o2) => String(o2.value) === String(v)) : null;
        if (o && o.model === false) return NaN;
        return v === cat ? 1 : 0;
      }
      if (f && f.type === "derived") return derivedValue(key, val);
      return val(key);
    });
  }

  const api = { prepare, predict, raw, rawTrees, shap, explain, buildRow, derivedValue, sigmoid, rcsTerms, splineExpand, lrRaw, ebmBin, ebmTerms };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.MTOPModel = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
