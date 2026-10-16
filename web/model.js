/* MTOP browser model
 * Evaluates exported LightGBM trees and computes exact TreeSHAP attributions
 * (Lundberg et al., Nat Mach Intell 2020), mirroring LightGBM's own
 * Tree::Decision / Tree::TreeSHAP so results match `pred_contrib=True`.
 * No patient data is needed or sent anywhere: everything runs in the browser.
 */
(function (root) {
  "use strict";

  const MISSING_ZERO = 1;
  const MISSING_NAN = 2;
  const K_ZERO = 1.0000000180025095e-35; // LightGBM's kZeroThreshold is the float 1e-35f

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

  function expectedValue(t) {
    if (t.f.length === 1) return t.v[0];
    const total = t.c[0];
    let e = 0;
    for (let i = 0; i < t.f.length; i++) if (t.f[i] < 0) e += (t.c[i] / total) * t.v[i];
    return e;
  }

  /** Prepare a model spec from the bundle (adds depth, expected values, path buffers). */
  function prepare(spec) {
    if (spec._ready) return spec;
    let base = 0;
    let maxDepth = 0;
    for (const t of spec.trees) {
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

  /** Raw score (log-odds) without recalibration. */
  function raw(spec, x) {
    prepare(spec);
    let s = 0;
    for (const t of spec.trees) s += leafValue(t, x);
    return s;
  }

  const sigmoid = (z, k) => 1 / (1 + Math.exp(-(k || 1) * z));

  /** Probability; `recent` (default true) applies the recent-era intercept. */
  function predict(spec, x, opts) {
    const r = raw(spec, x);
    const shift = opts && opts.recent === false ? 0 : spec.intercept_recent || 0;
    return { raw: r, logit: r + shift, p: sigmoid(r + shift, spec.sigmoid) };
  }

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

  /** SHAP values per model column (log-odds scale) plus base value. */
  function shap(spec, x) {
    prepare(spec);
    const phi = new Float64Array(spec.columns.length);
    const P = spec._path;
    for (const t of spec.trees) {
      if (t.f.length === 1) continue;
      P.fi[0] = -1; P.zf[0] = 0; P.of[0] = 0; P.pw[0] = 0;
      recurse(t, x, phi, P, 0, 0, 0, 1, 1, -1);
    }
    return { base: spec.base, phi: Array.from(phi) };
  }

  /** SHAP values summed per predictor (one-hot columns grouped). */
  function explain(spec, x) {
    const s = shap(spec, x);
    const byFeature = {};
    spec.column_feature.forEach((f, i) => { byFeature[f] = (byFeature[f] || 0) + s.phi[i]; });
    return { base: s.base, phi: s.phi, byFeature };
  }

  // ---------------------------------------------------------------- inputs -> model row
  /** Build the model row for a stage from calculator inputs keyed by predictor. */
  function buildRow(spec, featureMeta, inputs) {
    const meta = {};
    for (const f of featureMeta) meta[f.key] = f;
    const val = (k) => {
      const v = inputs[k];
      return v === null || v === undefined || v === "" || Number.isNaN(v) ? NaN : v;
    };
    return spec.columns.map((col) => {
      const [key, cat] = col.split("__");
      const f = meta[key];
      if (cat !== undefined) {
        const v = inputs[key];
        if (v === null || v === undefined || v === "") return NaN;
        return v === cat ? 1 : 0;
      }
      if (f && f.type === "derived") {
        if (key === "nihss_change_24h") return val("nihss_24h") - val("nihss_adm");
        return NaN;
      }
      const v = val(key);
      return typeof v === "number" ? v : Number(v);
    });
  }

  const api = { prepare, predict, raw, shap, explain, buildRow, sigmoid };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.MTOPModel = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
