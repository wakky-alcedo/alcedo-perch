const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch();
  const pg = await b.newPage({ viewport: { width: 1280, height: 1000 }, deviceScaleFactor: 1 });
  const errs = []; pg.on('pageerror', e => errs.push(e.message)); pg.on('console', m => { if (m.type()==='error') errs.push(m.text()); });
  await pg.goto('file:///tmp/claude-0/-home-claude/952cd200-ff57-5ba5-a0b1-39649ee2dff7/scratchpad/' + (process.env.F||'test.html') + ''); await pg.waitForTimeout(800);
  const r = await pg.evaluate(() => {
    const w = window.__wing, out = [];
    for (const v of w.VERSIONS){ if (!v.topo) continue; w.loadVersion(v.id);
      for (const t of [{tS:0,tE:0,tW:0}, v.tilt3d]){ w.setTilts(t); const m = w.metrics(w.params());
        out.push({id:v.id, tilt:t.tS+'/'+t.tE+'/'+t.tW, ok:m.ok, tmin:+m.tmin.toFixed(1), E:[+m.E0.toFixed(1), +m.E1.toFixed(1)], M:[+m.M0.toFixed(1), +m.M1.toFixed(1)], prot:+m.prot.toFixed(1), fold:+m.fold.toFixed(2)}); } }
    w.loadVersion(w.VERSIONS[0].id); return out;
  });
  r.forEach(x => console.log(JSON.stringify(x)));
  console.log('errors', JSON.stringify(errs));
  await b.close();
})();
