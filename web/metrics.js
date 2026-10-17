/* Validation metrics computed in the browser (local validation by partner centres).
 * Mirrors src/mtop/metrics.py; checked against scikit-learn in tests/test_extras_parity.py. */
(function (root) {
  "use strict";

  function auroc(y, p) {
    const n = y.length;
    const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => p[a] - p[b]);
    const ranks = new Float64Array(n);
    let i = 0;
    while (i < n) {
      let j = i;
      while (j + 1 < n && p[order[j + 1]] === p[order[i]]) j++;
      const r = (i + j) / 2 + 1;
      for (let k = i; k <= j; k++) ranks[order[k]] = r;
      i = j + 1;
    }
    let n1 = 0, s = 0;
    for (let k = 0; k < n; k++) if (y[k] === 1) { n1++; s += ranks[k]; }
    const n0 = n - n1;
    if (!n1 || !n0) return NaN;
    return (s - (n1 * (n1 + 1)) / 2) / (n1 * n0);
  }

  function brier(y, p) {
    let s = 0;
    for (let i = 0; i < y.length; i++) s += (p[i] - y[i]) ** 2;
    return s / y.length;
  }

  const clampP = (v) => Math.min(1 - 1e-6, Math.max(1e-6, v));
  const lg = (v) => { const q = clampP(v); return Math.log(q / (1 - q)); };

  /** Logistic regression y ~ a + b*x + offset by Newton-Raphson (b fixed at 0 when !withSlope). */
  function logisticFit(x, y, offset, withSlope) {
    let a = 0, b = withSlope ? 1 : 0;
    for (let it = 0; it < 100; it++) {
      let g0 = 0, g1 = 0, h00 = 0, h01 = 0, h11 = 0;
      for (let i = 0; i < y.length; i++) {
        const z = a + b * x[i] + (offset ? offset[i] : 0);
        const p = 1 / (1 + Math.exp(-z));
        const w = Math.max(p * (1 - p), 1e-12);
        const r = y[i] - p;
        g0 += r; g1 += r * x[i];
        h00 += w; h01 += w * x[i]; h11 += w * x[i] * x[i];
      }
      let da, db;
      if (withSlope) {
        const det = h00 * h11 - h01 * h01;
        if (Math.abs(det) < 1e-12) break;
        da = (h11 * g0 - h01 * g1) / det;
        db = (-h01 * g0 + h00 * g1) / det;
      } else { da = g0 / h00; db = 0; }
      a += da; b += db;
      if (Math.abs(da) + Math.abs(db) < 1e-10) break;
    }
    return [a, b];
  }

  function citl(y, p) {
    const off = p.map(lg);
    return logisticFit(new Array(y.length).fill(0), y, off, false)[0];
  }

  function slope(y, p) {
    return logisticFit(p.map(lg), y, null, true)[1];
  }

  function calibrationBins(y, p, nBins = 10, minN = 10) {
    const order = Array.from({ length: y.length }, (_, i) => i).sort((a, b) => p[a] - p[b]);
    const out = [];
    const size = y.length / nBins;
    for (let b = 0; b < nBins; b++) {
      const idx = order.slice(Math.round(b * size), Math.round((b + 1) * size));
      if (idx.length < minN) continue;
      let sp = 0, sy = 0;
      for (const i of idx) { sp += p[i]; sy += y[i]; }
      out.push({ pred: sp / idx.length, obs: sy / idx.length, n: idx.length });
    }
    return out;
  }

  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function percentile(sorted, q) {
    const pos = (sorted.length - 1) * q;
    const lo = Math.floor(pos), hi = Math.ceil(pos);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
  }

  /** Percentile bootstrap CI for AUROC and SE of logit(AUROC) (for random-effects pooling). */
  function bootAuroc(y, p, B = 500, seed = 20261016) {
    const rnd = mulberry32(seed);
    const n = y.length;
    const vals = [];
    const yb = new Array(n), pb = new Array(n);
    for (let b = 0; b < B; b++) {
      let s1 = 0;
      for (let i = 0; i < n; i++) { const k = Math.floor(rnd() * n); yb[i] = y[k]; pb[i] = p[k]; s1 += y[k]; }
      if (s1 === 0 || s1 === n) continue;
      vals.push(auroc(yb, pb));
    }
    vals.sort((a, b) => a - b);
    const lv = vals.map((v) => Math.log(v / (1 - v)));
    const m = lv.reduce((a, b) => a + b, 0) / lv.length;
    const sd = Math.sqrt(lv.reduce((a, b) => a + (b - m) ** 2, 0) / (lv.length - 1));
    return { ci: [percentile(vals, 0.025), percentile(vals, 0.975)], seLogit: sd };
  }

  function summary(y, p, opts) {
    const o = opts || {};
    const n = y.length;
    const events = y.reduce((a, b) => a + b, 0);
    const out = { n, events, mean_pred: p.reduce((a, b) => a + b, 0) / n, observed: events / n };
    if (events < (o.minEvents || 10) || n - events < (o.minEvents || 10)) { out.insufficient = true; return out; }
    out.auroc = auroc(y, p);
    const bt = bootAuroc(y, p, o.B || 500, o.seed);
    out.auroc_ci = bt.ci;
    out.se_logit = bt.seLogit;
    out.brier = brier(y, p);
    out.citl = citl(y, p);
    out.slope = slope(y, p);
    out.calibration = calibrationBins(y, p, n >= 250 ? 10 : 5, o.minCell || 10);
    return out;
  }

  // two-sided 97.5 % t quantiles, df 1..30
  const T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.16, 2.145, 2.131,
    2.12, 2.11, 2.101, 2.093, 2.086, 2.08, 2.074, 2.069, 2.064, 2.06, 2.056, 2.052, 2.048, 2.045, 2.042];
  const tq = (df) => (df <= 30 ? T975[Math.max(1, df) - 1] : 1.96);

  /** DerSimonian-Laird random-effects pooling (estimates and SEs on the same scale). */
  function randomEffects(est, se) {
    const k = est.length;
    const w = se.map((s) => 1 / (s * s));
    const sw = w.reduce((a, b) => a + b, 0);
    const fixed = est.reduce((a, e, i) => a + w[i] * e, 0) / sw;
    const q = est.reduce((a, e, i) => a + w[i] * (e - fixed) ** 2, 0);
    const c = sw - w.reduce((a, b) => a + b * b, 0) / sw;
    const tau2 = k > 1 ? Math.max(0, (q - (k - 1)) / c) : 0;
    const wr = se.map((s) => 1 / (s * s + tau2));
    const swr = wr.reduce((a, b) => a + b, 0);
    const mu = est.reduce((a, e, i) => a + wr[i] * e, 0) / swr;
    const seMu = Math.sqrt(1 / swr);
    const i2 = q > 0 ? Math.max(0, (q - (k - 1)) / q) : 0;
    const half = tq(Math.max(1, k - 2)) * Math.sqrt(tau2 + seMu * seMu);
    return { mu, lo: mu - 1.96 * seMu, hi: mu + 1.96 * seMu, piLo: mu - half, piHi: mu + half, tau2, i2, k, weights: wr.map((v) => v / swr) };
  }

  const api = { auroc, brier, logisticFit, citl, slope, calibrationBins, bootAuroc, summary, randomEffects, mulberry32 };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.MTOPMetrics = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
