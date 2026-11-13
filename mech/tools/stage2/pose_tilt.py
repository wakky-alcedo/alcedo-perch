"""Feasibility check: does a tilted wrist hinge (hand folding out of the arm plane, as in the CT) fit the folded pose better than the
planar wing? Pose-only, fold-fit (incl. CT term) without the sweep check. usage: pose_tilt.py seed [planar]"""
import sys, json, pathlib, numpy as np
from scipy.optimize import differential_evolution
import problem
from problem import Evaluator
from mech import shoulder
from kin import *

ROOT = pathlib.Path(__file__).resolve().parents[2]
seed = int(sys.argv[1]); PLANAR = len(sys.argv) > 2
base = problem.folded_wing


def folded_tilt(Rf, E1, M1, fz=None, tw=None):
    out = base(Rf, E1, M1, fz)
    if tw is None: return out
    a = unit(np.c_[tw, np.ones(len(tw))])
    Ru = out['Ru']; Rm = Ru @ rodrigues(a, -(np.pi - M1 * D))
    W = out['W']; T = W + mv(Rm, np.array([LMAN, 0, 0]))
    off = np.zeros((len(E1), 1, 3)) if fz is None else np.stack([0 * fz, 0 * fz, fz], -1)[:, None]
    pri = W[:, None] + np.einsum('pij,pnj->pni', Rm, PRI_F[None] + off)
    t = np.linspace(0, 1, 6)[None, :, None]; E = out['E']
    bones = np.concatenate([E[:, None] * t[:, 2:], E[:, None] + (W - E)[:, None] * t, W[:, None] + (T - W)[:, None] * t], 1)
    out.update(T=T, Rm=Rm, pri=pri, bones=bones); return out


ev = Evaluator('body', 'ulna', 45)
TW = None
problem.folded_wing = lambda Rf, E1, M1, fz=None: folded_tilt(Rf, E1, M1, fz, TW)


def cost(x):
    global TW
    X = np.atleast_2d(x); axis, th, Rf = shoulder(X[:, :3])
    TW = None if PLANAR else X[:, 6:8]
    f, fx = ev.fold_fit(dict(axis=axis, th=th, Rf=Rf, E1=X[:, 3], M1=X[:, 4], fz=X[:, 5]))
    return f[0] + 0.5 * max(0, np.degrees(th[0]) - 175), fx


B = [(-3.3, 3.3)] * 3 + [(10, 25), (10, 50), (0, 3)] + ([] if PLANAR else [(-3, 3), (-3, 3)])
r = differential_evolution(lambda x: cost(x)[0], B, seed=seed, maxiter=300, popsize=25, tol=1e-10, polish=True)
c, fx = cost(r.x)
tilt = None if PLANAR else float(np.degrees(np.arctan(np.hypot(*r.x[6:8]))))
row = dict(seed=seed, planar=PLANAR, cost=float(c), x=r.x.round(3).tolist(), wrist_axis_tilt_deg=tilt,
           **{k: np.round(v[0], 2).tolist() for k, v in fx.items() if k in ('parts', 'ct_pos', 'ct_hand', 'tipd', 'bury_max', 'dorsal')})
print(json.dumps(row), flush=True)
json.dump(row, open(ROOT / 'results' / 'stage2' / 'pose_only' / f'tilt_w45_s{seed}{"_planar" if PLANAR else ""}.json', 'w'))
