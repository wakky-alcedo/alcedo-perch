"""Two-stage linkage for each coupled joint (still 1 motor).

ELBOW chain (body frame):  humerus(input) --rod1--> body bellcrank (pivot G) --rod2--> ulna
WRIST chain (humerus frame): ulna(input rel. humerus) --rod3--> humerus bellcrank (pivot G2) --rod4--> hand

Stage 1 of each chain is a 4-bar solved by forward kinematics; stage 2 is synthesised exactly by fitting a
circle to the follower point expressed in the bellcrank frame.
"""
import json, sys
import numpy as np
from scipy.optimize import differential_evolution

D = np.pi / 180
LH, LU, LCMC, LDIG = 25.7, 32.0, 13.3, 7.0
cfg = dict(AH0=-15.0, AH1=-75.0, E0=150.0, E1=25.0, M0=160.0, M1=25.0)
which = sys.argv[1] if len(sys.argv) > 1 else 'elbow'
if len(sys.argv) > 2:
    cfg.update(json.loads(sys.argv[2]))
N = 61
u = np.linspace(0, 1, N)
a = (cfg['AH0'] + (cfg['AH1'] - cfg['AH0']) * u) * D
Et = cfg['E0'] + (cfg['E1'] - cfg['E0']) * u
Mt = cfg['M0'] + (cfg['M1'] - cfg['M0']) * u
au = a + (180 - Et) * D
am = au - (180 - Mt) * D


def Rm(t):
    c, s = np.cos(t), np.sin(t)
    return np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)


def circfit(P):
    x, y = P[:, 0], P[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones_like(x)]
    (cx, cy, c), *_ = np.linalg.lstsq(A, x * x + y * y, rcond=None)
    r = np.sqrt(max(c + cx * cx + cy * cy, 1e-12))
    return np.array([cx, cy]), r, np.hypot(x - cx, y - cy) - r


def ang_at(Pa, Pb, Pc):
    v1, v2 = Pa - Pb, Pc - Pb
    n1, n2 = np.linalg.norm(v1, axis=1), np.linalg.norm(v2, axis=1)
    cs = np.sum(v1 * v2, 1) / np.maximum(n1 * n2, 1e-12)
    t = np.degrees(np.arccos(np.clip(cs, -1, 1)))
    return min(t.min(), 180 - t.max())


def fourbar_rocker(crank_pts, G, rb, Lc, br):
    """crank point trajectory (N,2); rocker pivot G, rocker length rb, coupler length Lc.
    returns rocker angle beta (N,), rocker tip pts, ok, margin"""
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


def evaluate(x, detail=False):
    # x: Gx, Gy, hx, hy, rb1, Lc1, wx, wy, br(0/1 continuous)
    Gx, Gy, hx, hy, rb1, Lc1, wx, wy, br = x
    br = br > 0.5
    G = np.array([Gx, Gy])
    if which == 'elbow':
        frame_rot = np.zeros(N)                     # body frame
        crank = np.einsum('kij,j->ki', Rm(a), np.array([hx, hy]))           # point on humerus
        E = LH * np.c_[np.cos(a), np.sin(a)]
        follower = E + np.einsum('kij,j->ki', Rm(au), np.array([wx, wy]))  # point on ulna
        joint, base_link = E, None
    else:
        rel_u = au - a
        rel_m = am - a
        E_loc = np.array([LH, 0.0])
        crank = E_loc + np.einsum('kij,j->ki', Rm(rel_u), np.array([hx, hy]))  # point on ulna (humerus frame)
        W_loc = E_loc + LU * np.c_[np.cos(rel_u), np.sin(rel_u)]
        follower = W_loc + np.einsum('kij,j->ki', Rm(rel_m), np.array([wx, wy]))  # point on hand
    r = fourbar_rocker(crank, G, rb1, Lc1, br)
    if r is None:
        return 1e3 if not detail else None
    beta, B1, margin1 = r
    swing = abs(beta[-1] - beta[0]) / D
    if np.max(np.abs(np.diff(beta))) > 6 * D:
        return 1e3 if not detail else None
    # transmission at stage 1: angle at B1 between coupler and rocker
    t1 = ang_at(crank, B1, np.repeat(G[None], N, 0))
    # stage 2 synthesis in bellcrank frame
    F_b = np.einsum('kij,kj->ki', Rm(-beta), follower - G[None])
    C, L2, res = circfit(F_b)
    B2 = G[None] + np.einsum('kij,j->ki', Rm(beta), C)
    t2 = ang_at(B2, follower, follower - np.c_[np.ones(N), np.zeros(N)])  # placeholder overwritten below
    # transmission at stage 2: angle at follower between rod2 and the driven link
    if which == 'elbow':
        t2 = ang_at(B2, follower, E)
    else:
        t2 = ang_at(B2, follower, W_loc)
    resmax = np.max(np.abs(res))
    arm2 = np.linalg.norm(C)
    pen = 0.0
    pen += 6 * max(0, resmax - 0.05)
    pen += 0.4 * max(0, 40 - t1) + 0.4 * max(0, 40 - t2)
    pen += 0.2 * max(0, 3 - arm2) + 0.2 * max(0, 3 - rb1)
    # size limits (kingfisher 1:1 mm): keep parts compact so they fit along the bones
    pen += 0.6 * max(0, arm2 - 14) + 0.6 * max(0, L2 - 38) + 0.6 * max(0, 6 - L2)
    pen += 0.6 * max(0, rb1 - 14) + 0.6 * max(0, Lc1 - 38)
    pen += 0.6 * max(0, np.hypot(wx, wy) - (14 if which == 'wrist' else 30))
    pen += 0.6 * max(0, abs(wy) - 8)
    # the follower must have a real lever on the driven bone, and the bellcrank swing must stay moderate
    pen += 1.5 * max(0, 4.5 - np.hypot(wx, wy)) + 0.05 * max(0, swing - 100)
    # the stage-1 crank point also needs a lever about its own joint
    if which == 'wrist':
        pen += 1.5 * max(0, 4.5 - np.hypot(hx, hy))
    else:
        pen += 1.5 * max(0, 4.5 - np.hypot(hx, hy))
    # keep the bellcrank pivot in a buildable zone
    if which == 'elbow':
        pen += 0.6 * max(0, Gx - 3) + 0.6 * max(0, -13 - Gx) + 0.6 * max(0, abs(Gy) - 24)
    else:
        pen += 0.6 * max(0, np.hypot(Gx - LH, Gy) - 14)
    score = pen - 0.02 * min(t1, t2)
    if detail:
        return dict(G=G.tolist(), h=[hx, hy], rb1=rb1, Lc1=Lc1, w=[wx, wy], br=int(br), b2=C.tolist(), L2=float(L2),
                    t1=float(t1), t2=float(t2), resmax=float(resmax), swing=float(swing), margin1=float(margin1))
    return score


if which == 'elbow':
    B = [(-32, 6), (-30, 30), (-10, 30), (-12, 12), (3, 25), (5, 60), (-12, 32), (-14, 14), (0, 1)]
else:
    B = [(0, 45), (-20, 20), (-6, 32), (-12, 12), (3, 22), (5, 50), (-12, 20), (-14, 14), (0, 1)]

best = None
for seed in range(int(__import__('os').environ.get('SEEDS','3'))):
    r = differential_evolution(evaluate, B, seed=seed, maxiter=500, popsize=30, tol=1e-12, polish=False,
                               mutation=(0.5, 1.0), recombination=0.8)
    d = evaluate(r.x, detail=True)
    print(which, 'seed', seed, 'score', round(r.fun, 3), d and {k: (round(v, 2) if isinstance(v, float) else v) for k, v in d.items()}, flush=True)
    if d and (best is None or r.fun < best[0]):
        best = (r.fun, r.x.tolist(), d)
json.dump(dict(cfg=cfg, which=which, x=best[1], d=best[2]), open('two_%s.json' % which, 'w'))
print('BEST', json.dumps(best[2]))
