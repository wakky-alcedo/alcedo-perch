// Dump the full motion of a simulator version (bones, feather quads, mechanism samples) to JSON.
// usage: node stage1/dump_path.js [versionId] [out.json]   (run from mech/tools)
const { chromium } = require('playwright');
const path = require('path'), fs = require('fs');
(async () => {
  const id = process.argv[2] || 'v4', out = process.argv[3] || path.join(__dirname, '..', '..', 'results', 'stage1', `${id}_path.json`);
  const b = await chromium.launch(); const pg = await b.newPage();
  const errs = []; pg.on('pageerror', e => errs.push(e.message));
  await pg.goto('file:///' + path.resolve(__dirname, '..', '..', 'kingfisher_wing_fold.html').split(path.sep).join('/')); await pg.waitForTimeout(500);
  const r = await pg.evaluate((id) => {
    const w = window.__wing; w.loadVersion(id); const p = w.params(); const pth = w.path(p); const rt = w.rodTypes(pth);
    return { id, params: p, rodTypes: rt, body: w.ver.body, target: w.ver.target, steps: pth.map(q => ({ u: q.u, ok: q.ok, E: q.E, W: q.W, Ru: q.Ru, Rm: q.Rm, CM: q.CM, T: q.T, Edeg: q.Edeg, Mdeg: q.Mdeg,
      trans: q.trans, Rh: q.Rh, rods: q.mech.rods, pivots: q.mech.pivots, plates: q.mech.plates.map(pl => pl.slice(0, 3)), plateKind: q.mech.plates.map(pl => pl[3]),
      samples: w.mechSamples(q), prot: w.mechSamples(q).map(X => w.outsideDist(X, q)) })) };
  }, id);
  // feathers: evaluate featherQuads inside the page scope is not exported; recompute in Python from E,W,Ru,Rm instead.
  fs.mkdirSync(path.dirname(out), { recursive: true }); fs.writeFileSync(out, JSON.stringify(r));
  console.log('steps', r.steps.length, 'rodTypes', r.rodTypes.join(''), 'errors', JSON.stringify(errs));
  await b.close();
})();
