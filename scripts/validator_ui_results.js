// Run the in-browser validator UI ("Validate on your data") on a template file and save the results
// file it exports, to check other computations against the app itself:
//   node scripts/validator_ui_results.js http://localhost:8765/index.html template.csv results.json
const { chromium } = require('playwright');
(async () => {
  const [url, file, outJson] = process.argv.slice(2);
  const browser = await chromium.launch();
  const ctx = await browser.newContext({ acceptDownloads: true });
  const page = await ctx.newPage();
  page.on('pageerror', (e) => console.log('PAGEERROR', e.message));
  await page.goto(url + '#validate');
  await page.waitForSelector('#lv-file', { state: 'attached', timeout: 60000 });
  await page.setInputFiles('#lv-file', file);
  const run = page.getByRole('button', { name: /Run MTOP on/ });
  await run.waitFor({ timeout: 60000 });
  await run.click();
  await page.getByRole('button', { name: 'Prepare download' }).waitFor({ timeout: 900000 });
  await page.getByRole('button', { name: 'Prepare download' }).click();
  const [dl] = await Promise.all([page.waitForEvent('download'), page.getByRole('link', { name: 'Download' }).last().click()]);
  await dl.saveAs(outJson);
  console.log('saved', outJson);
  await browser.close();
})();
