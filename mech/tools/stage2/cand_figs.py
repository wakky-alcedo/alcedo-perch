"""Fold figures (folded wing + linkage vs CG body) for each stage-2 candidate -> results/local/cand_<id>.png (CG-derived: local only)."""
import sys, json, pathlib, numpy as np
from problem import Evaluator
from body import Body
from viz import fold_figure

ROOT = pathlib.Path(__file__).resolve().parents[2]
TILT = 'tilt' in sys.argv; SUB_ = '2e' if '2e' in sys.argv else '2d' if '2d' in sys.argv else 'tilt' if TILT else ''
def make_ev(c):
    if c.get('dsg'): return Evaluator(c['ec'], c['wc'], 45, tilt='A' if c['dsg'] == 'A' else True, stack=True)
    return Evaluator(c['ec'], c['wc'], 45, tilt=TILT)

cands = json.load(open(ROOT / 'results' / 'stage2' / SUB_ / 'candidates.json', encoding='utf-8'))
b = Body(45)
for c in cands:
    ev = make_ev(c)
    F, G, e = ev(np.array([c['x']]), extra=True)
    pts = np.array([np.asarray(p)[0, -1] for p in e['f']['pts']])
    t = f"{c['id']} {c.get('dsg', '')} {c['ec']}/{c['wc']} ({c['pick']}): err {F[0,0]:.1f} deg, tmin {-F[0,1]:.0f} deg, prot {F[0,2]:.1f} mm, sens {F[0,3]:.1f} deg, fit {F[0,4]:.1f}, part {F[0,5]:.0f} mm"
    fold_figure(e['d'], b, ROOT / 'results' / 'local' / f"cand_{c['id']}.png", t, 0, pts)
    print(c['id'], pts.shape)
