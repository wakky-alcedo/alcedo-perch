"""Design B (tilted wrist), pose only: how well can the folded wing fit the body and the CT when the wrist hinge is tilted about the
ulna (tw), the primaries have a root twist (phi) and a shifted folded angle (dl)? Fold fit incl. the CT hand-direction term, plus the
sweep check (feathers must not pass > 4 mm through the body). usage: pose_tilt.py seed [planar]
Output: results/stage2/pose_only/tiltB_w45_s<seed>[_planar].json"""
import sys, json, pathlib, numpy as np
from scipy.optimize import differential_evolution
from problem import Evaluator, SWEEP_MIN
from mech import shoulder, POSE_VARS, TILT_VARS

ROOT = pathlib.Path(__file__).resolve().parents[2]
seed = int(sys.argv[1]); PLANAR = len(sys.argv) > 2
ev = Evaluator('ulna', 'hum', 45, tilt=not PLANAR)
B = [(lo, hi) for _, lo, hi in POSE_VARS] + ([] if PLANAR else [(lo, hi) for n, lo, hi in TILT_VARS if n in ('tw', 'phi', 'dl')])


def pose(X):
    axis, th, Rf = shoulder(X[:, :3])
    d = dict(axis=axis, th=th, Rf=Rf, E1=X[:, 3], M1=X[:, 4], fz=X[:, 5], tilt=not PLANAR)
    if not PLANAR: d.update(tw=np.radians(X[:, 6]), phi=np.radians(X[:, 7]), dl=X[:, 8])
    return d


def cost(x, full=False):
    X = np.atleast_2d(x); d = pose(X)
    f, fx = ev.fold_fit(d); sw = ev.sweep(d)
    c = f + 2 * np.maximum(0, SWEEP_MIN - sw) + 0.5 * np.maximum(0, np.degrees(d['th']) - 175)
    return (c, fx, sw) if full else c


r = differential_evolution(lambda x: cost(x)[0], B, seed=seed, maxiter=400, popsize=30, tol=1e-10, polish=True, vectorized=False)
c, fx, sw = cost(r.x, True)
row = dict(seed=seed, planar=PLANAR, cost=float(c[0]), x=r.x.round(4).tolist(), sweep=float(sw[0]),
           **{k: np.round(v[0], 2).tolist() for k, v in fx.items() if k in ('parts', 'ct_pos', 'ct_hand', 'tipd', 'bury_max', 'dorsal')})
print(json.dumps(row), flush=True)
json.dump(row, open(ROOT / 'results' / 'stage2' / 'pose_only' / f'tiltB_w45_s{seed}{"_planar" if PLANAR else ""}.json', 'w'))
