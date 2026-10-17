// Helper for test_extras_parity.py: evaluate web/models_extra.js and web/metrics.js on a job file.
// usage: node node_extras.js input.json output.json
const fs = require("fs");
const path = require("path");
const XE = require(path.join(__dirname, "..", "web", "models_extra.js"));
const MX = require(path.join(__dirname, "..", "web", "metrics.js"));

const [inp, out] = process.argv.slice(2);
const job = JSON.parse(fs.readFileSync(inp, "utf8"));
const nanify = (a) => a.map((v) => (v === null ? NaN : v));
const res = {};

if (job.lr) res.lr = job.lr.rows.map((r) => XE.lrPredict(job.lr.spec, nanify(r)));
if (job.points) {
  res.points = {};
  for (const name of Object.keys(XE.POINTS)) res.points[name] = job.points.map((x) => {
    const v = XE.POINTS[name](x);
    return Number.isNaN(v) ? null : v;
  });
}
if (job.terms) res.terms = job.terms.rows.map((x) => XE.termValues(job.terms.spec, x).map((v) => (Number.isNaN(v) ? null : v)));
if (job.ordinal) {
  res.ordinal = job.ordinal.rows.map(([zg, zd, pg, pd]) => {
    const q = XE.ordinalProbs(zg, zd, job.ordinal.spec);
    return { q, dist: XE.fullDistribution(pg, pd, q) };
  });
}
if (job.metrics) {
  const { y, p } = job.metrics;
  res.metrics = { auroc: MX.auroc(y, p), brier: MX.brier(y, p), citl: MX.citl(y, p), slope: MX.slope(y, p) };
}
if (job.meta) res.meta = MX.randomEffects(job.meta.est, job.meta.se);
fs.writeFileSync(out, JSON.stringify(res));
