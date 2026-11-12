"""v3: deeper fold + every linkage part hidden inside the wing/body.

Allowed region for any mechanism point (top view, at every pose):
  body ellipse | propatagium triangle (shoulder, elbow, wrist) |
  band behind the humerus | feather side of the ulna | feather side of the hand.
Two chains (elbow: body bellcrank; wrist: humerus bellcrank) are optimised separately against the prescribed motion.
"""
import json, sys, os
import numpy as np
from scipy.optimize import differential_evolution

D = np.pi / 180
LH, LU, LCMC, LDIG = 25.7, 32.0, 13.3, 7.0
LMAN = LCMC + LDIG
cfg = dict(AH0=-15.0, AH1=-85.0, E0=150.0, E1=18.0, M0=160.0, M1=15.0)
which = sys.argv[1]
if len(sys.argv) > 2:
    cfg.update(json.loads(sys.argv[2]))
N = 61
u = np.linspace(0, 1, N)
a = (cfg['AH0'] + (cfg['AH1'] - cfg['AH0']) * u) * D
Et = cfg['E0'] + (cfg['E1'] - cfg['E0']) * u
Mt = cfg['M0'] + (cfg['M1'] - cfg['M0']) * u
au = a + (180 - Et) * D
am = au - (180 - Mt) * D
Ew = LH * np.c_[np.cos(a), np.sin(a)]
Ww = Ew + LU * np.c_[np.cos(au), np.sin(au)]
SUB = np.arange(0, N, 5)          # poses used for the hidden-part check


def Rm(t):
    c, s = np.cos(t), np.sin(t)
    return np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)


def rot2(t, v):
    c, s = np.cos(t), np.sin(t)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def circfit(P):
    x, y = P[:, 0], P[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones_like(x)]
    (cx, cy, c), *_ = np.linalg.lstsq(A, x * x + y * y, rcond=None)
    r = np.sqrt(max(c + cx * cx + cy * cy, 1e-12))
    return np.array([cx, cy]), r, np.hypot(x - cx, y - cy) - r


def ang_at(Pa, Pb, Pc):
    v1, v2 = Pa - Pb, Pc - Pb
    cs = np.sum(v1 * v2, 1) / np.maximum(np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1), 1e-12)
    t = np.degrees(np.arccos(np.clip(cs, -1, 1)))
    return min(t.min(), 180 - t.max())


def fourbar_rocker(crank_pts, G, rb, Lc, br):
    d = G[None] - crank_pts
    L = np.linalg.norm(d, axis=1)
    margin = np.minimum(Lc + rb - L, L - abs(Lc - rb)).min()
    if margin < 0.3:
        return None
    a_ = (Lc * Lc - rb * rb + L * L) / (2 * L)
    h = np.sqrt(np.maximum(Lc * Lc - a_ * a_, 0))
    ex = d / L[:, None]
    m = crank_pts + a_[:, None] * ex
    perp = np.c_[-ex[:, 1], ex[:, 0]]
    B = m + (h[:, None] * perp) * (1 if br else -1)
    beta = np.unwrap(np.arctan2(B[:, 1] - G[1], B[:, 0] - G[0]))
    return beta, B, margin


def outside(X, k):
    """distance (mm) by which world point X lies outside every allowed region at pose k (0 if hidden)"""
    best = 1e9
    # body ellipse (top view)
    ex, ey = (X[0] + 13) / 13, (X[1] + 12) / 30
    best = min(best, max(0.0, (np.hypot(ex, ey) - 1) * 13))
    # propatagium triangle
    S = np.zeros(2); E = Ew[k]; W = Ww[k]
    tri = [S, E, W]
    sgn = np.sign((E[0] - S[0]) * (W[1] - S[1]) - (E[1] - S[1]) * (W[0] - S[0]))
    dmax = 0.0
    for i in range(3):
        p, q = tri[i], tri[(i + 1) % 3]
        e = q - p; n = np.array([-e[1], e[0]]) * sgn / (np.hypot(*e) + 1e-12)   # inward normal
        dmax = max(dmax, -np.dot(X - p, n))
    best = min(best, max(0.0, dmax))
    # bands in bone frames: (origin, angle, length, y_min)
    for org, ang, L, ymin in ((S, a[k], LH, -12.0), (E, au[k], LU, -38.0), (W, am[k], LMAN, -45.0)):
        loc = rot2(-ang, X - org)
        d = max(0.0, loc[1] - 1.0, ymin - loc[1], -2 - loc[0], loc[0] - L - 2)
        best = min(best, d)
    return best


def seg_pts(P, Q, n=4):
    t = np.linspace(0, 1, n + 1)
    return [P + (Q - P) * ti for ti in t]


def evaluate(x, detail=False):
    Gx, Gy, hx, hy, rb1, Lc1, wx, wy, br = x
    br = br > 0.5
    G = np.array([Gx, Gy])
    if which == 'elbow':
        crank = np.einsum('kij,j->ki', Rm(a), np.array([hx, hy]))
        follower = Ew + np.einsum('kij,j->ki', Rm(au), np.array([wx, wy]))
        joint = Ew
    elif which == 'wrist':
        rel_u = au - a; rel_m = am - a
        E_loc = np.array([LH, 0.0])
        crank = E_loc + np.einsum('kij,j->ki', Rm(rel_u), np.array([hx, hy]))
        W_loc = E_loc + LU * np.c_[np.cos(rel_u), np.sin(rel_u)]
        follower = W_loc + np.einsum('kij,j->ki', Rm(rel_m), np.array([wx, wy]))
        joint = W_loc
    else:  # wristU: everything expressed in the ulna frame (origin at the elbow, x along the ulna)
        rel_u = au - a
        crank = np.einsum('kij,j->ki', Rm(-rel_u), np.array([hx - LH, hy]))   # point on the humerus (humerus coords from shoulder)
        W_loc = np.repeat(np.array([[LU, 0.0]]), N, 0)
        follower = W_loc + np.einsum('kij,j->ki', Rm(am - au), np.array([wx, wy]))
        joint = W_loc
    r = fourbar_rocker(crank, G, rb1, Lc1, br)
    if r is None:
        return 1e3 if not detail else None
    beta, B1, margin1 = r
    if np.max(np.abs(np.diff(beta))) > 6 * D:
        return 1e3 if not detail else None
    swing = abs(beta[-1] - beta[0]) / D
    t1 = ang_at(crank, B1, np.repeat(G[None], N, 0))
    F_b = np.einsum('kij,kj->ki', Rm(-beta), follower - G[None])
    C, L2, res = circfit(F_b)
    B2 = G[None] + np.einsum('kij,j->ki', Rm(beta), C)
    t2 = ang_at(B2, follower, joint)
    resmax = np.max(np.abs(res)); arm2 = np.linalg.norm(C)
    pen = 6 * max(0, resmax - 0.05)
    pen += 0.4 * max(0, 40 - t1) + 0.4 * max(0, 40 - t2)
    pen += 0.2 * max(0, 3 - arm2) + 0.2 * max(0, 3 - rb1)
    pen += 0.6 * max(0, arm2 - 14) + 0.6 * max(0, L2 - 38) + 0.6 * max(0, 6 - L2)
    pen += 0.6 * max(0, rb1 - 14) + 0.6 * max(0, Lc1 - 38)
    pen += 1.5 * max(0, 4.5 - np.hypot(wx, wy)) + 1.5 * max(0, 4.5 - np.hypot(hx, hy)) + 0.05 * max(0, swing - 100)
    # hidden-part check in world coordinates
    hid = 0.0
    for k in SUB:
        if which == 'elbow':
            P = dict(G=G, H=crank[k], B1=B1[k], B2=B2[k], U=follower[k])
            to_w = lambda v: v
        elif which == 'wrist':
            P = dict(G=G, H=crank[k], B1=B1[k], B2=B2[k], U=follower[k])
            to_w = lambda v, k=k: rot2(a[k], v)
        else:
            P = dict(G=G, H=crank[k], B1=B1[k], B2=B2[k], U=follower[k])
            to_w = lambda v, k=k: Ew[k] + rot2(au[k], v)
        Pw = {n: to_w(v) for n, v in P.items()}
        pts = [Pw['G'], Pw['B1'], Pw['B2'], Pw['H'], Pw['U']]
        pts += seg_pts(Pw['H'], Pw['B1']) + seg_pts(Pw['B2'], Pw['U']) + seg_pts(Pw['G'], Pw['B2'], 2) + seg_pts(Pw['B1'], Pw['B2'], 2)
        for X in pts:
            hid += outside(X, k)
    pen += 0.08 * hid
    score = pen - 0.02 * min(t1, t2)
    if detail:
        return dict(G=G.tolist(), h=[hx, hy], rb1=rb1, Lc1=Lc1, w=[wx, wy], br=int(br), b2=C.tolist(), L2=float(L2),
                    t1=float(t1), t2=float(t2), resmax=float(resmax), swing=float(swing), margin1=float(margin1), hidden=float(hid))
    return score


if which == 'elbow':
    B = [(-26, 4), (-40, 10), (-6, 30), (-14, 6), (3, 16), (5, 40), (-12, 32), (-14, 3), (0, 1)]
elif which == 'wrist':
    B = [(0, 45), (-20, 6), (-6, 34), (-14, 3), (3, 16), (5, 40), (-12, 20), (-14, 3), (0, 1)]
else:
    # G on the ulna (x along, y), crank h on the humerus (shoulder coords), w on the hand
    B = [(-4, 22), (-14, 2), (8, 30), (-14, 3), (3, 16), (5, 40), (-12, 20), (-14, 3), (0, 1)]

if __name__ == '__main__':
    best = None
    seeds = [int(s) for s in os.environ.get('SEEDS', '0,1,2').split(',')]
    for seed in seeds:
        r = differential_evolution(evaluate, B, seed=seed, maxiter=int(os.environ.get('MAXIT', '450')), popsize=28,
                                   tol=1e-12, polish=False, mutation=(0.5, 1.0), recombination=0.8)
        d = evaluate(r.x, detail=True)
        print(which, 'seed', seed, 'score', round(r.fun, 3), d and json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in d.items()}), flush=True)
        if d and (best is None or r.fun < best[0]):
            best = (r.fun, r.x.tolist(), d)
    json.dump(dict(cfg=cfg, which=which, x=best[1], d=best[2]), open('v3_%s.json' % which, 'w'))
    print('BEST', which, json.dumps(best[2]), flush=True)
