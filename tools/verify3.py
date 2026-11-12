"""Independent forward kinematics of the full 10-bar (2D): motor angle -> every link, step by step."""
import json
import numpy as np

D = np.pi / 180
LH, LU, LCMC, LDIG = 25.7, 32.0, 13.3, 7.0
import sys; el = json.load(open(sys.argv[1]))['d']
wr = json.load(open(sys.argv[2]))['d']
cfg = json.load(open(sys.argv[1]))['cfg']


def R(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s], [s, c]])


def cc(c1, r1, c2, r2):
    d = c2 - c1; L = np.hypot(*d)
    margin = min(r1 + r2 - L, L - abs(r1 - r2))
    a = (r1 * r1 - r2 * r2 + L * L) / (2 * L)
    h = np.sqrt(max(r1 * r1 - a * a, 0)); ex = d / L; m = c1 + a * ex; pp = np.array([-ex[1], ex[0]])
    return [m + h * pp, m - h * pp], margin


def near(cands, prev):
    return min(cands, key=lambda p: np.sum((p - prev) ** 2))


def ang(a, b, c):
    v1, v2 = a - b, c - b
    return np.degrees(np.arccos(np.clip(np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2), -1, 1)))


N = 241
us = np.linspace(0, 1, N)
G1 = np.array(el['G']); h1 = np.array(el['h']); rb1 = el['rb1']; Lc1 = el['Lc1']; b2 = np.array(el['b2']); L2 = el['L2']; w = np.array(el['w'])
G2l = np.array(wr['G']); h3 = np.array(wr['h']); rb3 = wr['rb1']; Lc3 = wr['Lc1']; b4 = np.array(wr['b2']); L4 = wr['L2']; q = np.array(wr['w'])
prev = {}
rows = []
for i, u in enumerate(us):
    a = (cfg['AH0'] + (cfg['AH1'] - cfg['AH0']) * u) * D
    Rh = R(a); E = Rh @ np.array([LH, 0])
    H1 = Rh @ h1
    # stage 1 elbow: bellcrank tip B1 on circle(G1, rb1) and circle(H1, Lc1)
    c, m1 = cc(G1, rb1, H1, Lc1)
    if i == 0:
        # pick the branch consistent with the optimiser (br flag): compare with its convention
        B1 = c[0] if el['br'] else c[1]
        # optimiser: B = m + h*perp*(+1 if br) with perp built from d = G - crank, i.e. circles swapped; resolve by trying both
    else:
        B1 = near(c, prev['B1'])
    if i == 0:
        # resolve the branch properly by checking which gives target elbow angle
        best = None
        for cand1 in c:
            beta = np.arctan2(*(cand1 - G1)[::-1])
            B2 = G1 + R(beta) @ b2
            c2, _ = cc(E, np.hypot(*w), B2, L2)
            for U in c2:
                au = np.arctan2(*(U - E)[::-1]) - np.arctan2(w[1], w[0])
                Edeg = 180 - (au - a) / D
                err = abs(((Edeg - cfg['E0']) + 180) % 360 - 180)
                if best is None or err < best[0]:
                    best = (err, cand1, U)
        B1 = best[1]; Ustart = best[2]
    beta = np.arctan2(*(B1 - G1)[::-1])
    B2 = G1 + R(beta) @ b2
    c2, m2 = cc(E, np.hypot(*w), B2, L2)
    U = Ustart if i == 0 else near(c2, prev['U'])
    au = np.arctan2(*(U - E)[::-1]) - np.arctan2(w[1], w[0])
    Ru = R(au); W = E + Ru @ np.array([LU, 0])
    # wrist chain (world frame): bellcrank 2 pivot on humerus
    G2 = Rh @ G2l
    C3 = E + Ru @ h3
    c3, m3 = cc(G2, rb3, C3, Lc3)
    if i == 0:
        best = None
        for cand3 in c3:
            beta2 = np.arctan2(*(cand3 - G2)[::-1])
            B4 = G2 + R(beta2) @ b4
            c4, _ = cc(W, np.hypot(*q), B4, L4)
            for Q in c4:
                am = np.arctan2(*(Q - W)[::-1]) - np.arctan2(q[1], q[0])
                Mdeg = 180 - (au - am) / D
                err = abs(((Mdeg - cfg['M0']) + 180) % 360 - 180)
                if best is None or err < best[0]:
                    best = (err, cand3, Q)
        B3 = best[1]; Qstart = best[2]
    else:
        B3 = near(c3, prev['B3'])
    beta2 = np.arctan2(*(B3 - G2)[::-1])
    B4 = G2 + R(beta2) @ b4
    c4, m4 = cc(W, np.hypot(*q), B4, L4)
    Q = Qstart if i == 0 else near(c4, prev['Q'])
    am = np.arctan2(*(Q - W)[::-1]) - np.arctan2(q[1], q[0])
    T = W + R(am) @ np.array([LCMC + LDIG, 0])
    Edeg = ang(np.zeros(2), E, W); Mdeg = ang(E, W, T)
    t = [ang(H1, B1, G1), ang(B2, U, E), ang(C3, B3, G2), ang(B4, Q, W)]
    tm = [min(x, 180 - x) for x in t]
    rows.append(dict(u=u, a=a / D, E=Edeg, M=Mdeg, tmin=min(tm), t=tm, margin=min(m1, m2, m3, m4), span=T[0],
                     beta=beta, phiE=au - a, beta2=beta2 - a, psi=am - au, tipy=T[1]))
    prev = dict(B1=B1, U=U, B3=B3, Q=Q)

E = np.array([r['E'] for r in rows]); M = np.array([r['M'] for r in rows])
Et = cfg['E0'] + (cfg['E1'] - cfg['E0']) * us; Mt = cfg['M0'] + (cfg['M1'] - cfg['M0']) * us
print('elbow  start/end %.1f -> %.1f  max err %.2f deg' % (E[0], E[-1], np.max(np.abs(E - Et))))
print('wrist  start/end %.1f -> %.1f  max err %.2f deg' % (M[0], M[-1], np.max(np.abs(M - Mt))))
print('min transmission per joint', np.round(np.min([r['t'] for r in rows], 0), 1))
print('min lock margin %.2f mm' % min(r['margin'] for r in rows))
print('max step (E,M) %.2f %.2f' % (np.max(np.abs(np.diff(E))), np.max(np.abs(np.diff(M)))))
print('span open %.1f folded %.1f mm (ratio %.0f%%)' % (rows[0]['span'], rows[-1]['span'], 100 * rows[-1]['span'] / rows[0]['span']))
print('folded tip y %.1f (negative = toward tail)' % rows[-1]['tipy'])
init = dict(beta1=rows[0]['beta'], phiE=rows[0]['phiE'], beta2=rows[0]['beta2'], psi=rows[0]['psi'])
print('INIT', json.dumps(init))
json.dump(dict(el=el, wr=wr, cfg=cfg, init=init), open(sys.argv[3], 'w'))
