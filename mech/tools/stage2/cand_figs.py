"""Fold figures (folded wing + linkage vs CG body) for each stage-2 candidate -> results/local/cand_<id>.png (CG-derived: local only)."""
import json, pathlib, numpy as np
from problem import Evaluator
from body import Body
from viz import fold_figure

ROOT = pathlib.Path(__file__).resolve().parents[2]
cands = json.load(open(ROOT / 'results' / 'stage2' / 'candidates.json', encoding='utf-8'))
b = Body(45)
for c in cands:
    ev = Evaluator(c['ec'], c['wc'], 45)
    F, G, e = ev(np.array([c['x']]), extra=True)
    pts = np.array([np.asarray(p)[0, -1] for p in e['f']['pts']])
    t = f"{c['id']} {c['ec']}/{c['wc']} ({c['pick']}): err {F[0,0]:.1f} deg, tmin {-F[0,1]:.0f} deg, prot {F[0,2]:.1f} mm, sens {F[0,3]:.1f} deg, fit {F[0,4]:.1f}, part {F[0,5]:.0f} mm"
    fold_figure(e['d'], b, ROOT / 'results' / 'local' / f"cand_{c['id']}.png", t, 0, pts)
    print(c['id'], pts.shape)
