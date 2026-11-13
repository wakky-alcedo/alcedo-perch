"""Feasibility check: how well can a single oblique shoulder hinge + planar folded wing fit the CG body's folded-wing position?
Pose variables only (yaw, pitch, roll, E1, M1). Objective = fold-fit + penalty for sweeping through the body.
usage: pose_only.py width seed"""
import sys, json, pathlib, numpy as np
from scipy.optimize import differential_evolution
from problem import Evaluator
from mech import shoulder

ROOT = pathlib.Path(__file__).resolve().parents[2]
width, seed = float(sys.argv[1]), int(sys.argv[2]); NOSWEEP = len(sys.argv) > 3
ev = Evaluator('body', 'ulna', width)


def pose_d(X):
    axis, th, Rf = shoulder(X[:, :3])
    return dict(axis=axis, th=th, Rf=Rf, E1=X[:, 3], M1=X[:, 4], fz=X[:, 5])


def cost(X):
    X = np.atleast_2d(X); d = pose_d(X)
    fit, fx = ev.fold_fit(d); sw = ev.sweep(d)
    c = fit + (0 if NOSWEEP else 3) * np.maximum(0, -3 - sw) + 0.5 * np.maximum(0, np.degrees(d['th']) - 175)
    return c


B = [(-3.3, 3.3)] * 3 + [(10, 25), (10, 50), (0, 3)]
r = differential_evolution(lambda x: cost(x)[0], B, seed=seed, maxiter=300, popsize=25, tol=1e-10, polish=True, vectorized=False)
X = r.x[None]; d = pose_d(X); fit, fx = ev.fold_fit(d); sw = ev.sweep(d)
row = dict(width=width, seed=seed, cost=float(r.fun), x=r.x.round(2).tolist(), fit=float(fit[0]), sweep=float(sw[0]), theta=float(np.degrees(d['th'][0])),
           axis=d['axis'][0].round(4).tolist(), **{k: np.round(v[0], 2).tolist() for k, v in fx.items()})
print(json.dumps(row), flush=True)
out = ROOT / 'results' / 'stage2' / 'pose_only'; out.mkdir(parents=True, exist_ok=True)
json.dump(row, open(out / f'w{int(width)}_s{seed}{"_nosweep" if NOSWEEP else ""}.json', 'w'))
