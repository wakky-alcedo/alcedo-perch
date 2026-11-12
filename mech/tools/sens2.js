const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch(); const pg = await b.newPage();
  await pg.goto('file://' + __dirname + '/' + (process.env.F || 'test_v4.html')); await pg.waitForTimeout(500);
  const r = await pg.evaluate((id) => {
    const w = window.__wing; w.loadVersion(id); const base = w.params(), m0 = w.metrics(base), out = [];
    for (const k of Object.keys(w.ver.dims)) { const row = [k];
      for (const d of [-0.1, 0.1]){ const p = Object.assign({}, base); p[k] += d; const m = w.metrics(p);
        row.push((m.ok?'':'X')+[m.E0-m0.E0, m.E1-m0.E1, m.M0-m0.M0, m.M1-m0.M1].map(v=>v.toFixed(1)).join('/')); }
      out.push(row.join('  ')); }
    return out;
  }, process.env.V || 'v4');
  console.log(r.join('\n')); await b.close();
})();
