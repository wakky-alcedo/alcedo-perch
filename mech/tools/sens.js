const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch(); const pg = await b.newPage();
  const errs = []; pg.on('pageerror', e => errs.push(e.message));
  await pg.goto('file://' + __dirname + '/' + (process.env.F || 'test.html')); await pg.waitForTimeout(500);
  const r = await pg.evaluate((id) => {
    const w = window.__wing; w.loadVersion(id); const base = w.params(), m0 = w.metrics(base);
    const out = {base:{ok:m0.ok, tmin:m0.tmin, E:[m0.E0,m0.E1], M:[m0.M0,m0.M1], prot:m0.prot, fold:m0.fold, rods:w.rodTypes(m0.pth)}, worst:0, worstKey:'', bad:[]};
    for (const k of Object.keys(w.ver.dims)) for (const d of [-0.1, 0.1]){
      const p = Object.assign({}, base); p[k] += d; const m = w.metrics(p);
      const dv = Math.max(Math.abs(m.E1 - m0.E1), Math.abs(m.M1 - m0.M1), Math.abs(m.E0 - m0.E0), Math.abs(m.M0 - m0.M0));
      if (!m.ok) out.bad.push(k + d); if (dv > out.worst){ out.worst = dv; out.worstKey = k + (d > 0 ? '+' : '-'); } }
    // hidden-ness by phase
    const prof = []; m0.pth.forEach((q,i) => { if (i % 20 === 0) prof.push(+w.protrusion(q).toFixed(2)); });
    out.prof = prof; return out;
  }, process.env.V || 'v4');
  console.log(JSON.stringify(r)); console.log('errors', JSON.stringify(errs)); await b.close();
})();
