/* MTOP comparators and distribution layer, evaluated in the browser.
 * Mirrors src/mtop/lr.py, src/mtop/scores.py and src/mtop/extras.py
 * (checked by tests/test_extras_parity.py). The learners themselves are in model.js. */
(function (root) {
  "use strict";
  const sig = (z) => 1 / (1 + Math.exp(-z));
  const logit = (p) => {
    const q = Math.min(1 - 1e-6, Math.max(1e-6, p));
    return Math.log(q / (1 - q));
  };
  const miss = (v) => v === null || v === undefined || v === "" || (typeof v === "number" && Number.isNaN(v));
  const GLU_150 = 150 / 18.016; // 150 mg/dL in mmol/L

  /** Penalised logistic regression pipeline (median imputation + indicators, scaling). */
  function lrPredict(spec, x) {
    const filled = x.map((v, i) => (miss(v) ? spec.medians[i] : v));
    const ind = spec.indicators.map((j) => (miss(x[j]) ? 1 : 0));
    const z = filled.concat(ind);
    let lin = spec.intercept + (spec.shift || 0);
    for (let i = 0; i < z.length; i++) lin += ((z[i] - spec.mean[i]) / spec.scale[i]) * spec.coef[i];
    return sig(lin);
  }

  /** Published predictor-set terms: num, bin, in (membership), min(x, c), max(x - c, 0). */
  function termValues(terms, inputs) {
    return terms.map((t) => {
      const [kind, key, arg] = t;
      const v = inputs[key];
      if (miss(v)) return NaN;
      if (kind === "num" || kind === "bin") return Number(v);
      if (kind === "in") return arg.includes(String(v)) ? 1 : 0;
      if (kind === "min") return Math.min(Number(v), arg);
      if (kind === "max0") return Math.max(Number(v) - arg, 0);
      return NaN;
    });
  }

  const num = (v) => (miss(v) ? NaN : Number(v));

  /** Items of the FNO bedside score: bands (label, test) and the points of the band that applies. */
  const band = (key, bands) => ({
    key, bands,
    match(x) { const v = num(x[key]); return Number.isNaN(v) ? -1 : bands.findIndex((b) => b.test(v)); },
    points(x) { const i = this.match(x); return i < 0 ? 0 : bands[i].pts; },
  });
  const BEDSIDE = [
    band("age", [{ label: "<60", pts: 0, test: (v) => v < 60 }, { label: "60–79", pts: 2, test: (v) => v < 80 }, { label: "≥80", pts: 5, test: () => true }]),
    band("nihss_adm", [{ label: "0–9", pts: 0, test: (v) => v <= 9 }, { label: "10–14", pts: 1, test: (v) => v <= 14 }, { label: "15–19", pts: 3, test: (v) => v <= 19 }, { label: "≥20", pts: 5, test: () => true }]),
    band("glucose", [{ label: "<7.8", pts: 0, test: (v) => v < 7.8 }, { label: "7.8–11.0", pts: 2, test: (v) => v < 11.1 }, { label: "≥11.1", pts: 3, test: () => true }]),
    band("mrs_pre", [{ label: "0", pts: 0, test: (v) => v < 1 }, { label: "≥1", pts: 2, test: () => true }]),
    band("aspects", [{ label: "6–10", pts: 0, test: (v) => v > 5 }, { label: "0–5", pts: 2, test: () => true }]),
    band("onset_to_puncture", [{ label: "<180", pts: 0, test: (v) => v < 180 }, { label: "180–299", pts: 1, test: (v) => v < 300 }, { label: "≥300", pts: 2, test: () => true }]),
    band("stroke_prior", [{ label: "no", pts: 0, test: (v) => v !== 1 }, { label: "yes", pts: 1, test: () => true }]),
  ];

  const POINTS = {
    "THRIVE": (x) => {
      const n = num(x.nihss_adm), a = num(x.age);
      if (Number.isNaN(n) || Number.isNaN(a)) return NaN;
      const z = (k) => (miss(x[k]) ? 0 : Number(x[k]));
      return (n <= 10 ? 0 : n <= 20 ? 1 : 2) + (a <= 59 ? 0 : a <= 79 ? 1 : 2) + z("htn") + z("dm") + z("arrhythmia");
    },
    "SPAN-100": (x) => {
      const n = num(x.nihss_adm), a = num(x.age);
      if (Number.isNaN(n) || Number.isNaN(a)) return NaN;
      return a + n >= 100 ? 1 : 0;
    },
    "SPAN index": (x) => num(x.age) + num(x.nihss_adm),
    "HIAT2": (x) => {
      const a = num(x.age), n = num(x.nihss_adm), g = num(x.glucose), s = num(x.aspects);
      if ([a, n, g, s].some(Number.isNaN)) return NaN;
      return (a <= 59 ? 0 : a <= 79 ? 2 : 4) + (n <= 10 ? 0 : n <= 20 ? 1 : 2) + (g >= GLU_150 ? 1 : 0) + (s <= 7 ? 3 : 0);
    },
    "PRE": (x) => num(x.age) + 2 * num(x.nihss_adm) - 10 * num(x.aspects),
    // FNO 7-item bedside score (0-20, higher = worse); an item that is not recorded scores 0
    "Bedside score": (x) => {
      const a = num(x.age), n = num(x.nihss_adm);
      if (Number.isNaN(a) || Number.isNaN(n)) return NaN;
      return BEDSIDE.reduce((s, item) => s + item.points(x), 0);
    },
    "NIHSS 24 h": (x) => num(x.nihss_24h),
    "NIHSS day 7": (x) => num(x.nihss_7d),
  };

  function pointsProb(map, s) {
    const v = Number.isNaN(s) ? map.fill : s;
    return sig(map.a + map.b * v + (map.shift || 0));
  }

  /** Probability of mRS 0-2 from a comparator for one patient (inputs keyed by predictor). */
  function comparatorProb(comp, inputs) {
    if (comp.kind === "refit") return lrPredict(comp.spec, termValues(comp.terms, inputs));
    return null;
  }

  /** Cumulative-logit probabilities of mRS 0..6 from the two chosen models' scores. */
  function ordinalProbs(zg, zd, ord) {
    const eta = ord.bg * zg + ord.bd * zd;
    const cum = [0].concat(ord.theta.map((th) => sig(th - eta))).concat([1]);
    const out = [];
    for (let k = 0; k < 7; k++) out.push(Math.max(0, cum[k + 1] - cum[k]));
    return out;
  }

  /** Split the independent and dependent bands into grades; headline probabilities stay fixed. */
  function fullDistribution(pGood, pDeath, q) {
    let g = pGood, d = pDeath;
    if (g + d > 0.98) { const k = 0.98 / (g + d); g *= k; d *= k; }
    const dep = 1 - g - d;
    const sg = q[0] + q[1] + q[2], sd = q[3] + q[4] + q[5];
    return [
      (g * q[0]) / sg, (g * q[1]) / sg, (g * q[2]) / sg,
      (dep * q[3]) / sd, (dep * q[4]) / sd, (dep * q[5]) / sd,
      d,
    ];
  }

  /** Bootstrap 95 % range for a deployed estimate, from the risk bin it falls into. */
  function uncertainty(bins, p) {
    if (!bins || !bins.length) return null;
    let best = bins[0], bestD = Infinity;
    for (const b of bins) {
      if (p >= b.lo && p < b.hi) { best = b; bestD = 0; break; }
      const d = Math.min(Math.abs(p - b.lo), Math.abs(p - b.hi));
      if (d < bestD) { bestD = d; best = b; }
    }
    const z = logit(p);
    return [sig(z + best.off_lo), sig(z + best.off_hi)];
  }

  const api = { sig, logit, lrPredict, termValues, POINTS, BEDSIDE, pointsProb, comparatorProb, ordinalProbs, fullDistribution, uncertainty, GLU_150 };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.MTOPExtra = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
