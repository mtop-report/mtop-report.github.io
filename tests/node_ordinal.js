// Helper for test_ordinal.py: evaluate an MTOP ordinal model with web/model.js on rows from a JSON file.
// usage: node node_ordinal.js input.json output.json
const fs = require("fs");
const path = require("path");
const M = require(path.join(__dirname, "..", "web", "model.js"));

const [inp, out] = process.argv.slice(2);
const job = JSON.parse(fs.readFileSync(inp, "utf8"));
const L = job.model;
const res = job.rows.map((r) => {
  const x = r.map((v) => (v === null ? NaN : v));
  const o = M.ordinal(L, x, { recent: job.recent !== false });
  const s = M.shap(L, x);
  return { lp: o.lp, p: o.p, base: s.base, phi: s.phi };
});
fs.writeFileSync(out, JSON.stringify(res));
