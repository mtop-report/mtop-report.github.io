// Helper for test_parity.py: evaluate a published learner with web/model.js on rows from a JSON file.
// usage: node node_eval.js input.json output.json
const fs = require("fs");
const path = require("path");
const M = require(path.join(__dirname, "..", "web", "model.js"));

const [inp, out] = process.argv.slice(2);
const job = JSON.parse(fs.readFileSync(inp, "utf8"));
const spec = job.spec;
const res = job.rows.map((r) => {
  const x = r.map((v) => (v === null ? NaN : v));
  const s = M.shap(spec, x);
  return { raw: M.raw(spec, x), base: s.base, phi: s.phi };
});
fs.writeFileSync(out, JSON.stringify(res));
