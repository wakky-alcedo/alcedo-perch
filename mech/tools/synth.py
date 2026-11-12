"""Exact-ish synthesis of the two coupling loops by circle fitting.

Loop 1 (elbow): target ulna attitude au(u). A point U fixed on the ulna traces a curve in the body frame.
  If that curve is (nearly) a circle, a rod from its centre A reproduces the motion. Grid-search U, fit circle.
Loop 2 (wrist): in the HUMERUS frame, a point Q fixed on the hand traces a curve. Its circle centre is P (on
  the humerus) and the radius is the radius-bar length.
"""
import json, sys
import numpy as np

D = np.pi / 180
LH, LU, LCMC, LDIG = 25.7, 32.0, 13.3, 7.0
cfg = dict(AH0=-15.0, AH1=-75.0, E0=150.0, E1=25.0, M0=160.0, M1=25.0)
if len(sys.argv) > 1:
    cfg.update(json.loads(sys.argv[1]))
N = 81
u = np.linspace(0, 1, N)
a = (cfg['AH0'] + (cfg['AH1'] - cfg['AH0']) * u) * D
E = cfg['E0'] + (cfg['E1'] - cfg['E0']) * u
M = cfg['M0'] + (cfg['M1'] - cfg['M0']) * u
au = a + (180 - E) * D          # ulna direction (body frame)
am = au - (180 - M) * D         # hand direction


def R(t):
    c, s = np.cos(t), np.sin(t)
    return np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)


def circfit(P):
    x, y = P[:, 0], P[:, 1]
    Am = np.c_[2 * x, 2 * y, np.ones_like(x)]
    b = x * x + y * y
    (cx, cy, c), *_ = np.linalg.lstsq(Am, b, rcond=None)
    r = np.sqrt(max(c + cx * cx + cy * cy, 1e-12))
    res = np.hypot(x - cx, y - cy) - r
    return np.array([cx, cy]), r, res


def trans(Pa, Pb, Pc):
    """angle at Pb between Pb->Pa and Pb->Pc, degrees"""
    v1, v2 = Pa - Pb, Pc - Pb
    cs = np.sum(v1 * v2, 1) / np.linalg.norm(v1, axis=1) / np.linalg.norm(v2, axis=1)
    return np.degrees(np.arccos(np.clip(cs, -1, 1)))


Epts = LH * np.c_[np.cos(a), np.sin(a)]
Wpts = Epts + LU * np.c_[np.cos(au), np.sin(au)]

# ---------- loop 1 ----------
best1 = []
for s in np.linspace(0.5, 30, 119):
    for n in np.linspace(-12, 12, 97):
        w = np.array([s, n])
        Up = Epts + np.einsum('kij,j->ki', R(au), w)
        C, r, res = circfit(Up)
        err = np.sqrt(np.mean(res ** 2))
        # body-side ground pivot only (x <= 6, inside the torso's lateral edge region)
        if C[0] > 6 or C[0] < -30 or abs(C[1]) > 35 or r > 75:
            continue
        tr = trans(Epts, Up, np.repeat(C[None], N, 0))
        tmin = min(tr.min(), 180 - tr.max())
        best1.append((err, tmin, s, n, C[0], C[1], r))
best1 = np.array(best1)
# rank: residual small and transmission angle large
score1 = best1[:, 0] + 0.08 * np.maximum(0, 35 - best1[:, 1])
o1 = best1[np.argsort(score1)[:5]]
print('LOOP1 top: err(mm) tmin s n Ax Ay Lrod')
for row in o1:
    print(np.round(row, 3))

# ---------- loop 2 (humerus frame) ----------
rel_u = au - a          # ulna wrt humerus
rel_m = am - a          # hand wrt humerus
E_loc = np.array([LH, 0.0])
W_loc = E_loc + LU * np.c_[np.cos(rel_u), np.sin(rel_u)]
best2 = []
for qx in np.linspace(-10, 6, 81):
    for qy in np.linspace(-6, 10, 81):
        q = np.array([qx, qy])
        Qp = W_loc + np.einsum('kij,j->ki', R(rel_m), q)
        C, r, res = circfit(Qp)
        err = np.sqrt(np.mean(res ** 2))
        p = C - E_loc   # offset from elbow, humerus frame
        if np.hypot(*p) > 14 or r < 12 or r > 50:
            continue
        tr = trans(W_loc, Qp, np.repeat(C[None], N, 0))
        tmin = min(tr.min(), 180 - tr.max())
        # radius should run on the cranial (anterior) side of the ulna: signed distance of its midpoint
        mid = 0.5 * (Qp + C)
        dvec = W_loc - E_loc
        nrm = np.c_[-dvec[:, 1], dvec[:, 0]] / np.linalg.norm(dvec, axis=1)[:, None]
        side = np.sum((mid - E_loc) * nrm, 1).min()
        best2.append((err, tmin, qx, qy, p[0], p[1], r, side))
best2 = np.array(best2)
score2 = best2[:, 0] + 0.08 * np.maximum(0, 35 - best2[:, 1]) + 0.3 * np.maximum(0, 1.5 - best2[:, 7])
o2 = best2[np.argsort(score2)[:5]]
print('LOOP2 top: err(mm) tmin qx qy px py Lr side')
for row in o2:
    print(np.round(row, 3))
json.dump(dict(cfg=cfg, loop1=o1[0].tolist(), loop2=o2[0].tolist()), open('synth_out.json', 'w'))
