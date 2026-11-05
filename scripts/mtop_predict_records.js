// MTOP predictions from the app's own code (window.MTOPApp.predictStage, with its averaging of unknown
// inputs) on validator-style records (python -m mtop.mrsclean records), with the validator's
// eligibility rule; identical to what "Validate on your data" computes. Serve the built site locally
// (python -m http.server in site/), then:
//   node scripts/mtop_predict_records.js http://localhost:8765/index.html records.json preds.json [chosen|all|indicators]
// (chosen: the calculator's model; all: every published learner; indicators: the calculator's model
// with the model's own missing-value indicators instead of averaging unknown inputs)
// Row-level output: write it only to a git-ignored folder (data/raw/...).
const { chromium } = require('playwright');
const fs = require('fs');
(async () => {
  const [url, recFile, outFile, learnerMode] = process.argv.slice(2);
  const recs = JSON.parse(fs.readFileSync(recFile, 'utf8'));
  const browser = await chromium.launch();
  const page = await browser.newPage();
  page.on('pageerror', (e) => console.log('PAGEERROR', e.message));
  await page.goto(url);
  await page.waitForFunction(() => window.MTOPApp && window.MTOPApp.predictStage, null, { timeout: 120000 });
  const meta = await page.evaluate(() => {
    const A = window.MTOPApp;
    const st = ['admission', 'procedure', 'h24'];
    return Object.fromEntries(st.map((s) => [s, Object.fromEntries(A.TARGETS.map((t) => {
      const e = A.entryOf(s, t); return [t, { chosen: e.chosen, learners: Object.keys(e.learners) }]; }))]));
  });
  const out = { meta, preds: {} };
  const CH = 500;
  const t0 = Date.now();
  for (let i = 0; i < recs.length; i += CH) {
    const chunk = recs.slice(i, i + CH);
    const res = await page.evaluate(({ chunk, mode }) => {
      const A = window.MTOPApp;
      const eligible = (x, st) => x.mrs90 !== null && (A.idx(st) < 1 || x.tici !== null) && (st !== 'h24' || x.nihss_24h !== null);
      return chunk.map((x) => {
        const row = {};
        for (const st of ['admission', 'procedure', 'h24']) {
          if (!eligible(x, st)) { row[st] = null; continue; }
          row[st] = {};
          for (const tg of A.TARGETS) {
            const e = A.entryOf(st, tg);
            const names = mode === 'all' ? [e.chosen].concat(Object.keys(e.learners).filter((k) => k !== e.chosen)) : [e.chosen];
            row[st][tg] = {};
            for (const nm of names) {
              // "indicators": the model's own missing-value handling (development median plus a
              // missing indicator) instead of the calculator's averaging over registry values
              row[st][tg][nm] = mode === 'indicators'
                ? A.M.predict(e.learners[nm], A.M.buildRow(e.columns, A.B.features, x), { recent: true }).p
                : A.predictStage(st, x, tg, nm).p;
            }
          }
        }
        return row;
      });
    }, { chunk, mode: learnerMode || 'chosen' });
    for (let j = 0; j < res.length; j++) out.preds[i + j] = res[j];
    if (i === 0 || (i / CH) % 10 === 0) console.log(`rows ${i + res.length} / ${recs.length}, ${((Date.now() - t0) / 1000).toFixed(0)} s`);
  }
  fs.writeFileSync(outFile, JSON.stringify(out));
  console.log('done', ((Date.now() - t0) / 1000).toFixed(0), 's');
  await browser.close();
})();
