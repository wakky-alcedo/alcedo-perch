import json, time, pathlib, numpy as np
from problem import Evaluator
from mech import *
ROOT = pathlib.Path(__file__).resolve().parents[2]
P = json.load(open(ROOT / 'results' / 'v4_pose.json')); sim = json.load(open(ROOT / 'results' / 'stage1' / 'v4_path.json')); p = sim['params']
axis, th, Rf = shoulder(ypr_to_rv(np.array([P['x'][:3]]))); Ab = hinge_frame(axis)[0]; G1 = np.array([p['G1x'], p['G1y'], p['G1z']]); g = Ab[:, :2].T @ G1
X = np.array([[*ypr_to_rv(np.array([P['x'][:3]]))[0], P['E1'], P['M1'], g[0], g[1], p['h1x'], p['h1y'], p['h1z'], p['rb1'], p['Lc1'], p['wx'], p['wy'], 1,
               p['G2x'], p['G2y'], p['h3x'], p['h3y'], p['rb3'], p['Lc3'], p['qx'], p['qy'], 0]])
for w in (40, 45, 50):
    ev = Evaluator('body', 'ulna', w)
    F, G, x = ev(X, extra=True)
    print('width', w, 'F', F.round(2), '\n  G', G.round(2))
    print('  swivel %.1f  margins %s  trans %s  lev %.2f  part %.1f  rodforce %.2f N/Nmm  worst(0.05) %.2f deg  sweep %.1f  tipd %.1f  bury %.1f  cover %.1f' % (
        x['swivel'][0], x['margins'][0].round(2), x['trans'][0].round(1), x['lev'][0], x['part'][0], x['rodforce'][0], x['worst005'][0], x['sweep'][0],
        x['fitx']['tipd'][0], x['fitx']['bury_max'][0], x['fitx']['cover'][0]))
ev = Evaluator('body', 'ulna', 45)
rng = np.random.default_rng(0)
for ec, wc in [('body', 'ulna'), ('hum', 'ulna'), ('ulna', 'hum')]:
    ev = Evaluator(ec, wc, 45); lo = np.array([v[1] for v in ev.vars]); hi = np.array([v[2] for v in ev.vars])
    Xr = lo + (hi - lo) * rng.random((200, len(lo)))
    t = time.time(); F, G = ev(Xr); dt = time.time() - t
    print(ec, wc, 'eval 200 random: %.2f s, feasible %d, valid %d' % (dt, np.sum(np.all(G <= 0, 1)), np.sum(G[:, 0] <= 0)))
