// Screenshots of the simulator with the CG mesh body. usage: node stage1/shots.js outdir
const { chromium } = require('playwright'); const path = require('path');
(async () => {
  const out = process.argv[2] || path.join(__dirname, '..', '..', 'results', 'stage1');
  const b = await chromium.launch(); const pg = await b.newPage({ viewport: { width: 1400, height: 1000 } });
  const errs = []; pg.on('pageerror', e => errs.push(e.message)); pg.on('console', m => { if (m.type() === 'error') errs.push(m.text()); });
  await pg.goto('file:///' + path.resolve(__dirname, '..', '..', 'kingfisher_wing_fold.html').split(path.sep).join('/')); await pg.waitForTimeout(600);
  const set = async (shape, wing, fold, view, alpha) => {
    await pg.evaluate(([shape, wing, fold, alpha]) => { const s = document.getElementById('bodyShape'); s.value = shape; s.dispatchEvent(new Event('change'));
      const w = document.getElementById('ckWing'); w.checked = wing; w.dispatchEvent(new Event('change'));
      const a = document.getElementById('bodyA'); a.value = alpha; a.dispatchEvent(new Event('input'));
      const f = document.getElementById('fold'); f.value = fold; f.dispatchEvent(new Event('input')); }, [shape, wing, fold, alpha]);
    await pg.click(`[data-view="${view}"]`); await pg.waitForTimeout(700);
  };
  const list = [['asis', true, 1, 'side', 0.45, 'sim_asis_folded_side.png'], ['asis', true, 1, 'top', 0.45, 'sim_asis_folded_top.png'],
                ['asis', false, 0, 'iso', 0.35, 'sim_asis_open_iso.png'], ['slim40', true, 1, 'back', 0.45, 'sim_slim40_folded_back.png'],
                ['ell', false, 1, 'side', 0.3, 'sim_ell_folded_side.png']];
  for (const [s, w, f, v, a, name] of list) { await set(s, w, f, v, a); await (await pg.$('#stage')).screenshot({ path: path.join(out, name) }); }
  console.log('errors', JSON.stringify(errs)); await b.close();
})();
