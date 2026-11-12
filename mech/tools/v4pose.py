"""v4 step A: choose the FOLDED wing pose so the folded wing follows a perched kingfisher's outline,
and the shoulder is a single hinge about the Euler axis that carries the open humerus frame to the folded one.

Body model (1:1 mm, origin at the right shoulder, x lateral, y cranial, z dorsal):
  torso ellipsoid centre (-9, -18, -6), radii (12, 32, 12); tail plate y -46..-84.
  Total length bill tip -> tail tip ~150 mm (Alcedo atthis 160-170 mm incl. bill ~38 mm). Wing length
  (wrist to primary tip) 78.8 mm (BTO, live adults).
Folded-pose criteria (perched kingfisher):
  - folded wing lies ON the body: bones/feathers neither sink >3 mm into the torso nor stand >6 mm off it;
  - in top view everything stays inside the torso outline (+2 mm), wing tips over the rump (y -62..-50, x -10..-1);
  - wrist (carpal bend) sits at the front of the folded wing (y 0..12);
  - during the whole motion (rotation about the fixed hinge axis) the wing never sweeps through the body.
"""
import json, os, sys
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial.transform import Rotation as Rot

D = np.pi / 180
LH, LU, LCMC, LDIG = 25.7, 32.0, 13.3, 7.0
LMAN = LCMC + LDIG
BC, BR = np.array([-9.0, -18.0, -6.0]), np.array([12.0, 32.0, 12.0])
PRIM = [(0.05, 40, -82, -8), (0.15, 42, -75, -7), (0.25, 44, -68, -6), (0.35, 46, -61, -5), (0.45, 48, -54, -4),
        (0.55, 50, -46, -3.5), (0.68, 53, -38, -3), (0.80, 56, -29, -2.5), (0.92, 57, -20, -2), (1.00, 53, -11, -1.5)]
SEC = [(0.96 - i * 0.095, 40 - i * 0.9, -100 - i * 3.0, -170 + i * 0.8) for i in range(10)]
E0, M0, AH0 = 150.0, 160.0, -15.0
Ro = Rot.from_euler('z', AH0, degrees=True).as_matrix()      # open humerus frame: x along bone, z = wing-plane normal (up)


def Rz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def wing_points(Rh, E, M, fe, fw):
    """bones + feathers for a humerus frame Rh and joint angles E, M (deg). fe/fw: feather spread 0..1"""
    Ep = Rh @ np.array([LH, 0, 0])
    Ru = Rh @ Rz((180 - E) * D)
    W = Ep + Ru @ np.array([LU, 0, 0])
    Rm = Ru @ Rz(-(180 - M) * D)
    T = W + Rm @ np.array([LMAN, 0, 0])
    bones = [Ep * t for t in np.linspace(0.2, 1, 5)] + [Ep + (W - Ep) * t for t in np.linspace(0, 1, 7)] + [W + (T - W) * t for t in np.linspace(0, 1, 4)]
    feath, tips = [], []
    for f, L, a0, a1 in SEC:
        a = (a1 + (a0 - a1) * fe) * D
        base = Ep + Ru @ np.array([LU * f, 0, 0]); d = Ru @ np.array([np.cos(a), np.sin(a), 0])
        feath += [base + d * L * 0.5, base + d * L]
    for f, L, a0, a1 in PRIM:
        a = (a1 + (a0 - a1) * fw) * D
        base = W + Rm @ np.array([LMAN * f, 0, 0]); d = Rm @ np.array([np.cos(a), np.sin(a), 0])
        feath += [base + d * L * 0.5, base + d * L]; tips.append(base + d * L)
    return dict(E=Ep, W=W, T=T, bones=np.array(bones), feath=np.array(feath), tips=np.array(tips))


def body_sdist(P):
    """approximate signed distance to the torso ellipsoid surface (mm, + outside)"""
    q = (P - BC) / BR
    return (np.linalg.norm(q, axis=-1) - 1) * 12.0


def halfwidth(y):
    t = 1 - ((y - BC[1]) / BR[1]) ** 2
    return np.where(t > 0, BR[0] * np.sqrt(np.clip(t, 0, 1)), np.where(y < -46, 7.0, 0.0))


def folded_frame(x):
    yaw, pitch, roll = x[:3]
    # humerus direction from yaw (about z, from +x) and pitch (down negative); roll about the bone axis
    return Rot.from_euler('zyx', [yaw, -pitch, roll], degrees=True).as_matrix()


def cost(x, detail=False):
    Rf = folded_frame(x); E1, M1 = x[3], x[4]
    g = wing_points(Rf, E1, M1, 0, 0)
    c = 0.0
    allp = np.vstack([g['bones'], g['feath']])
    s = body_sdist(allp)
    on_torso = allp[:, 1] > -48
    c += np.sum(np.maximum(0, -2 - s[on_torso]) ** 2) + np.sum(np.maximum(0, s[on_torso] - 5) ** 2) * 0.5
    # behind the torso the folded primaries rest on the rump / tail base (dorsal, z about -4..+3)
    rz = allp[~on_torso, 2]
    c += np.sum(np.maximum(0, -4 - rz) ** 2) + np.sum(np.maximum(0, rz - 3) ** 2)
    # top-view outline
    over = np.abs(allp[:, 0] - BC[0]) - (halfwidth(allp[:, 1]) + 2)
    c += np.sum(np.maximum(0, over) ** 2)
    tip = g['tips'][7:9].mean(0)
    c += 2 * (max(0, tip[1] + 58) ** 2 + max(0, -72 - tip[1]) ** 2 + max(0, tip[0] + 1) ** 2 + max(0, -10 - tip[0]) ** 2)
    c += 2 * (max(0, -g['W'][1]) ** 2 + max(0, g['W'][1] - 12) ** 2)
    # path through the fixed hinge: no sweeping through the body
    R_rel = Rot.from_matrix(Rf @ Ro.T)
    rv = R_rel.as_rotvec(); th = np.linalg.norm(rv)
    if th > 175 * D:
        c += 50
    pen_path = 0.0
    for u in np.linspace(0.1, 0.9, 9):
        Rh = Rot.from_rotvec(rv * u).as_matrix() @ Ro
        gu = wing_points(Rh, E0 + (E1 - E0) * u, M0 + (M1 - M0) * u, 1 - u, 1 - u)
        su = body_sdist(np.vstack([gu['bones'][1:], gu['feath']]))
        pen_path += np.sum(np.maximum(0, -3 - su) ** 2)
    c += 0.3 * pen_path
    c += 0.002 * (th / D)   # prefer smaller motor rotation
    if detail:
        return dict(cost=c, Rf=Rf.tolist(), E1=E1, M1=M1, axis=(rv / th).tolist(), theta=th / D, tip=tip.tolist(), W=g['W'].tolist(),
                    sd_min=float(s.min()), sd_max=float(s.max()), over_max=float(over.max()), path_pen=pen_path)
    return c


if __name__ == '__main__':
    B = [(-180, -60), (-40, 60), (-150, 150), (12, 40), (12, 40)]
    res = []
    for seed in range(2):
        r = differential_evolution(cost, B, seed=seed, maxiter=400, popsize=30, tol=1e-12, polish=True)
        d = cost(r.x, detail=True)
        res.append((r.fun, r.x.tolist(), d))
        print('seed', seed, 'cost %.3f' % r.fun, 'x', np.round(r.x, 1), 'theta %.0f' % d['theta'], 'axis', np.round(d['axis'], 3),
              'tip', np.round(d['tip'], 1), 'W', np.round(d['W'], 1), 'sd %.1f..%.1f over %.1f path %.1f' % (d['sd_min'], d['sd_max'], d['over_max'], d['path_pen']), flush=True)
    res.sort(key=lambda t: t[0])
    json.dump(dict(x=res[0][1], d=res[0][2], all=[(r[0], r[1]) for r in res]), open('v4_pose.json', 'w'))
