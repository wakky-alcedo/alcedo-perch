"""v4 linkage synthesis on the outline-fitted motion.

Motion: humerus frame Rh(u) = Rot(axis, theta*u) @ Ro (single shoulder hinge), elbow E(u), wrist M(u) linear.
ELBOW chain (spatial): humerus point H1 -> rod1 -> body bellcrank 1 (axis parallel to the shoulder hinge, in the
  plane of H1's circle, so this stage is PLANAR: revolute joints) -> rod2 with BALL ends -> ulna lever U.
  Stage 2 is synthesised exactly by a sphere fit in the bellcrank frame.
WRIST chain (planar, in the wing plane): humerus crank C3 -> rod3 -> bellcrank 2 on the ulna -> rod4 -> hand lever Q.
  All revolute joints. Stage 2 synthesised by a circle fit in the bellcrank frame.
"""
import json, os, sys
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial.transform import Rotation as Rot

D = np.pi / 180
LH, LU, LCMC, LDIG = 25.7, 32.0, 13.3, 7.0
LMAN = LCMC + LDIG
BC, BR = np.array([-9.0, -18.0, -6.0]), np.array([12.0, 32.0, 12.0])
P = json.load(open('v4_pose.json'))
Ro = np.array(P['Ro']); ax = np.array(P['axis']); TH = P['theta'] * D
E0, E1, M0, M1 = P['E0'], P['E1'], P['M0'], P['M1']
which = sys.argv[1]
N = 61
u = np.linspace(0, 1, N)
Rh = np.array([Rot.from_rotvec(ax * TH * t).as_matrix() @ Ro for t in u])
Et = E0 + (E1 - E0) * u
Mt = M0 + (M1 - M0) * u
phiE = (180 - Et) * D                 # ulna rotation about the elbow axis, relative to the humerus
psi = -(180 - Mt) * D                 # hand rotation relative to the ulna


def Rz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


Ru = np.array([Rh[k] @ Rz(phiE[k]) for k in range(N)])
Ew = np.array([Rh[k] @ np.array([LH, 0, 0]) for k in range(N)])
Ww = np.array([Ew[k] + Ru[k] @ np.array([LU, 0, 0]) for k in range(N)])
# frame of the shoulder hinge: e1, e2 perpendicular to the axis
e1 = np.cross(ax, [0, 0, 1.0]); e1 /= np.linalg.norm(e1); e2 = np.cross(ax, e1)
Ab = np.c_[e1, e2, ax]
SUB = np.arange(0, N, 5)


def body_sd(X):
    return (np.linalg.norm((X - BC) / BR) - 1) * 12.0


def wing_outside(Xh, E_ang, M_ang):
    """X in the humerus frame (origin shoulder). Distance outside the wing region (triangle, bands)."""
    if abs(Xh[2]) > 4:
        return abs(Xh[2]) - 4 + 0.0
    S = np.zeros(2); E = np.array([LH, 0.0])
    au = (180 - E_ang) * D; W = E + LU * np.array([np.cos(au), np.sin(au)])
    am = au - (180 - M_ang) * D; T = W + LMAN * np.array([np.cos(am), np.sin(am)])
    X = Xh[:2]
    best = 1e9
    tri = [S, E, W]
    sg = np.sign((E[0] - S[0]) * (W[1] - S[1]) - (E[1] - S[1]) * (W[0] - S[0])) or 1
    dmax = 0.0
    for i in range(3):
        p, q = tri[i], tri[(i + 1) % 3]; e = q - p; n = np.array([-e[1], e[0]]) * sg / (np.hypot(*e) + 1e-12)
        dmax = max(dmax, -np.dot(X - p, n))
    best = min(best, max(0.0, dmax))
    for org, ang, L, ymin in ((S, 0.0, LH, -12.0), (E, au, LU, -38.0), (W, am, LMAN, -45.0)):
        c, s = np.cos(-ang), np.sin(-ang); d = X - org; loc = np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])
        best = min(best, max(0.0, loc[1] - 1.0, ymin - loc[1], -2 - loc[0], loc[0] - L - 2))
    return best


def outside(Xw, k):
    if body_sd(Xw) <= -0.5:
        return 0.0
    Xh = Rh[k].T @ Xw
    return min(max(0.0, body_sd(Xw) + 0.5), wing_outside(Xh, Et[k], Mt[k]))


def sphere_fit(Pts):
    A = np.c_[2 * Pts, np.ones(len(Pts))]
    b = np.sum(Pts ** 2, 1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:3]; r = np.sqrt(max(sol[3] + c @ c, 1e-12))
    return c, r, np.linalg.norm(Pts - c, axis=1) - r


def circ_fit(P2):
    x, y = P2[:, 0], P2[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones_like(x)]
    (cx, cy, c), *_ = np.linalg.lstsq(A, x * x + y * y, rcond=None)
    r = np.sqrt(max(c + cx * cx + cy * cy, 1e-12))
    return np.array([cx, cy]), r, np.hypot(x - cx, y - cy) - r


def cc2(c1, r1, c2, r2, br):
    d = c2 - c1; L = np.linalg.norm(d, axis=1)
    margin = np.minimum(r1 + r2 - L, L - np.abs(r1 - r2)).min()
    a_ = (r1 * r1 - r2 * r2 + L * L) / (2 * L)
    h = np.sqrt(np.maximum(r1 * r1 - a_ * a_, 0)); ex = d / L[:, None]
    m = c1 + a_[:, None] * ex; perp = np.c_[-ex[:, 1], ex[:, 0]]
    return m + (h[:, None] * perp) * (1 if br else -1), margin


def trans_planar(Pa, Pb, Pc):
    v1, v2 = Pa - Pb, Pc - Pb
    t = np.degrees(np.arccos(np.clip(np.sum(v1 * v2, 1) / np.maximum(np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1), 1e-12), -1, 1)))
    return min(t.min(), 180 - t.max())


def seg(Pa, Pb, n=4):
    return [Pa + (Pb - Pa) * t for t in np.linspace(0, 1, n + 1)]


def eval_elbow(x, detail=False):
    hx, hy, hz, g1, g2, rb1, Lc1, wx, wy, br = x
    br = br > 0.5
    H1 = np.array([Rh[k] @ np.array([hx, hy, hz]) for k in range(N)])
    axial = H1[0] @ ax
    H2 = H1 @ Ab[:, :2]                        # coordinates in the hinge plane
    G2d = np.array([g1, g2])
    B1_2d, margin = cc2(np.repeat(G2d[None], N, 0), rb1, H2, Lc1, br)
    if margin < 0.3:
        return 1e3 if not detail else None
    beta = np.unwrap(np.arctan2(B1_2d[:, 1] - g2, B1_2d[:, 0] - g1))
    if np.max(np.abs(np.diff(beta))) > 6 * D:
        return 1e3 if not detail else None
    t1 = trans_planar(H2, B1_2d, np.repeat(G2d[None], N, 0))
    Gw = Ab @ np.array([g1, g2, axial])
    U = np.array([Ew[k] + Ru[k] @ np.array([wx, wy, 0]) for k in range(N)])
    Fb = np.array([Ab @ Rz(b) for b in beta])
    Ub = np.einsum('kji,kj->ki', Fb, U - Gw)  # U in the bellcrank frame
    c, L2, res = sphere_fit(Ub)
    B2 = Gw + np.einsum('kij,j->ki', Fb, c)
    # spatial transmission at U: angle between rod and U's velocity (tangent of its circle about the elbow axis)
    nE = Rh[:, :, 2]
    tang = np.cross(nE, U - Ew); tang /= np.linalg.norm(tang, axis=1)[:, None]
    rod = U - B2; rod /= np.linalg.norm(rod, axis=1)[:, None]
    t2 = np.degrees(np.arcsin(np.clip(np.abs(np.sum(rod * tang, 1)), 0, 1))).min()
    resmax = np.abs(res).max(); arm2 = np.linalg.norm(c)
    pen = 6 * max(0, resmax - 0.05) + 0.4 * max(0, 40 - t1) + 0.4 * max(0, 40 - t2)
    pen += 0.6 * max(0, arm2 - 16) + 0.6 * max(0, L2 - 40) + 0.6 * max(0, 6 - L2) + 0.6 * max(0, rb1 - 16) + 0.6 * max(0, Lc1 - 40)
    pen += 0.2 * max(0, 3 - arm2) + 0.2 * max(0, 3 - rb1)
    pen += 1.5 * max(0, 4.5 - np.hypot(wx, wy)) + 1.5 * max(0, 4.5 - np.hypot(hx - LH * 0, hy))
    swing = abs(beta[-1] - beta[0]) / D
    pen += 0.05 * max(0, swing - 140)
    hid = 0.0
    for k in SUB:
        B1w = Gw + Fb[k] @ np.array([rb1, 0, 0])
        pts = [Gw, B1w, B2[k], H1[k], U[k]] + seg(H1[k], B1w) + seg(B2[k], U[k], 6) + seg(Gw, B2[k], 2) + seg(B1w, B2[k], 2)
        hid += sum(outside(X, k) for X in pts)
    pen += 0.08 * hid
    score = pen - 0.02 * min(t1, t2)
    if detail:
        return dict(h=[hx, hy, hz], g=[g1, g2], axial=float(axial), G=Gw.tolist(), rb1=rb1, Lc1=Lc1, w=[wx, wy], br=int(br), b2=c.tolist(), L2=float(L2),
                    t1=float(t1), t2=float(t2), resmax=float(resmax), swing=float(swing), margin=float(margin), hidden=float(hid))
    return score


def eval_wrist(x, detail=False):
    """planar, in the humerus frame; bellcrank 2 on the ulna, crank C3 on the humerus"""
    Gx, Gy, hx, hy, rb1, Lc1, wx, wy, br = x
    br = br > 0.5
    rel_u = phiE; am_rel = psi                         # ulna wrt humerus, hand wrt ulna
    crank = np.array([np.array([[np.cos(-r), -np.sin(-r)], [np.sin(-r), np.cos(-r)]]) @ np.array([hx - LH, hy]) for r in rel_u])  # ulna frame
    W_loc = np.repeat(np.array([[LU, 0.0]]), N, 0)
    follower = W_loc + np.array([[np.cos(a) * wx - np.sin(a) * wy, np.sin(a) * wx + np.cos(a) * wy] for a in am_rel])
    G = np.array([Gx, Gy])
    B1, margin = cc2(np.repeat(G[None], N, 0), rb1, crank, Lc1, br)
    if margin < 0.3:
        return 1e3 if not detail else None
    beta = np.unwrap(np.arctan2(B1[:, 1] - Gy, B1[:, 0] - Gx))
    if np.max(np.abs(np.diff(beta))) > 6 * D:
        return 1e3 if not detail else None
    t1 = trans_planar(crank, B1, np.repeat(G[None], N, 0))
    Fb = np.array([[[np.cos(b), -np.sin(b)], [np.sin(b), np.cos(b)]] for b in beta])
    Fl = np.einsum('kji,kj->ki', Fb, follower - G)
    C, L2, res = circ_fit(Fl)
    B2 = G + np.einsum('kij,j->ki', Fb, C)
    t2 = trans_planar(B2, follower, W_loc)
    resmax = np.abs(res).max(); arm2 = np.linalg.norm(C); swing = abs(beta[-1] - beta[0]) / D
    pen = 6 * max(0, resmax - 0.05) + 0.4 * max(0, 40 - t1) + 0.4 * max(0, 40 - t2)
    pen += 0.6 * max(0, arm2 - 14) + 0.6 * max(0, L2 - 38) + 0.6 * max(0, 6 - L2) + 0.6 * max(0, rb1 - 14) + 0.6 * max(0, Lc1 - 38)
    pen += 0.2 * max(0, 3 - arm2) + 0.2 * max(0, 3 - rb1)
    pen += 1.5 * max(0, 4.5 - np.hypot(wx, wy)) + 1.5 * max(0, 4.5 - np.hypot(hx - LH, hy)) + 0.05 * max(0, swing - 140)
    hid = 0.0
    for k in SUB:
        cu, su = np.cos(rel_u[k]), np.sin(rel_u[k])
        toh = lambda v: np.array([LH + cu * v[0] - su * v[1], su * v[0] + cu * v[1], 0.0])   # ulna frame -> humerus frame
        B1k = B1[k]
        pts = [G, B1k, B2[k], crank[k], follower[k]] + seg(crank[k], B1k) + seg(B2[k], follower[k], 6) + seg(G, B2[k], 2) + seg(B1k, B2[k], 2)
        hid += sum(wing_outside(toh(X), Et[k], Mt[k]) for X in pts)
    pen += 0.08 * hid
    score = pen - 0.02 * min(t1, t2)
    if detail:
        return dict(G=[Gx, Gy], h=[hx, hy], rb1=rb1, Lc1=Lc1, w=[wx, wy], br=int(br), b2=C.tolist(), L2=float(L2), t1=float(t1), t2=float(t2),
                    resmax=float(resmax), swing=float(swing), margin=float(margin), hidden=float(hid))
    return score


if __name__ == '__main__':
    if which == 'elbow':
        f = eval_elbow
        B = [(-6, 26), (-14, 4), (-4, 4), (-40, 40), (-40, 40), (3, 16), (5, 40), (-12, 10), (-12, 6), (0, 1)]
    else:
        f = eval_wrist
        B = [(-4, 22), (-14, 4), (8, 30), (-14, 3), (3, 16), (5, 40), (-12, 20), (-14, 4), (0, 1)]
    seeds = [int(s) for s in os.environ.get('SEEDS', '0,1,2').split(',')]
    out = []
    for seed in seeds:
        r = differential_evolution(f, B, seed=seed, maxiter=int(os.environ.get('MAXIT', '300')), popsize=26, tol=1e-12, polish=False,
                                   mutation=(0.5, 1.0), recombination=0.8)
        d = f(r.x, detail=True)
        out.append(dict(score=float(r.fun), x=r.x.tolist(), d=d))
        print(which, 'seed', seed, 'score %.3f' % r.fun, json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in (d or {}).items()}), flush=True)
    json.dump(out, open('v4_%s_all.json' % which, 'w'), default=float)
    best = min(out, key=lambda o: o['score'])
    print('BEST', which, 'score %.3f' % best['score'], flush=True)
