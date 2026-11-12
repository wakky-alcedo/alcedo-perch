const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch();
  const pg = await b.newPage({ viewport: { width: 1280, height: 1000 }, deviceScaleFactor: 1 });
  const errs = []; pg.on('pageerror', e => errs.push(e.message));
  await pg.goto('file://' + __dirname + '/' + (process.env.F || 'test.html')); await pg.waitForTimeout(600);
  const shot = async (name, id, fold, mode, view) => {
    await pg.evaluate(([id, fold, mode]) => { window.__wing.loadVersion(id); const m = document.getElementById(mode); if (m && m.offsetParent) m.click();
      const f = document.getElementById('fold'); f.value = fold; f.dispatchEvent(new Event('input')); }, [id, fold, mode]);
    await pg.click(`[data-view="${view}"]`); await pg.waitForTimeout(600);
    await (await pg.$('#stage')).screenshot({ path: name });
  };
  const list = JSON.parse(process.env.SHOTS);
  for (const s of list) await shot(...s);
  if (process.env.FULL) { await pg.screenshot({ path: process.env.FULL, fullPage: true }); }
  console.log('errors', JSON.stringify(errs));
  await b.close();
})();
