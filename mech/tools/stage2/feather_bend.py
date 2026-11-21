"""Stage 2f: feathers bowed along the shaft (printed flat, then heat-formed on a jig; user decision 2026-11-15).
A flat folded wing cuts the round body like a chord: the middle of every feather lies 3-6 mm inside the body surface while root and
tip are near it. Here each feather is bent in the plane that contains its shaft and the wing normal (+z = dorsal):
    z(r) = r tan(alpha) - r^2 / (2 R)          r = distance from the feather root along the shaft
alpha (root angle, set by the printed root mount) and R (jig radius) are common to all secondaries and to all primaries.
For each candidate the pair (alpha, R) per group that minimises the fold fit is found on a grid. Reported:
- fold fit and its parts, flat vs bent;
- camber of the open wing = arc height over the root-tip chord / feather length (real bird wings: a few %);
- feather-feather: for neighbouring feathers that overlap in plan, how much their height difference changes during the fold
  (flat: 0). That change must be taken by feather spacing or by flexing; reported as the max over the fold.
Option DROOP=<deg> (user 2026-11-21 'tip slightly lowered'): the primaries' root-tip chord (longest primary, 56 mm) must point
at least DROOP deg below the wing plane in the open wing; finer R grid. chord_deg is reported for both groups.
Option CAMBER_MAX=<x> / CAMBER_MAX_SEC=<x>: cap on the primaries' / secondaries' camber; FB_OUT=<name>: output file name (for sweeps).
usage: feather_bend.py [ID ...]   (MIDLINE=1)  -> results/stage2/<STAGE>/feather_bend.json"""
import sys, os, json, numpy as np
IDS, sys.argv = sys.argv[1:], sys.argv[:1] + ['45', '2e']
from analyze import make_ev, OUT
import kin
from kin import SEC, PRIM, LU, LMAN

PER = 5
A_GRID = np.radians([0, 5, 10, 15, 20, 25, 30, 35])
R_GRID = np.array([25, 35, 50, 70, 100, 150, np.inf])
FLAT0 = (kin.SEC_F.copy(), kin.pri_local, kin.feather_local)


def r_of(table, per=PER):
    """distance from the root along the shaft of every sample point (same order as feather_local)"""
    out = []
    for f, L, a0, a1 in table: out += [L * t for t in np.linspace(0.15, 1, per)] + [L * 0.82] * 2
    return np.array(out)


RS, RP = r_of(SEC), r_of(PRIM)
bow = lambda r, a, R: r * np.tan(a) - (0 if np.isinf(R) else r ** 2 / (2 * R))
chord = lambda a, R, L: float(np.degrees(np.arctan(bow(L, a, R) / L)))
DROOP = float(os.environ.get('DROOP', 'nan'))
LP = max(L for f, L, a0, a1 in PRIM)
if DROOP == DROOP: R_GRID = np.array([25, 30, 35, 40, 45, 50, 60, 70, 85, 100, 125, 150, 200, np.inf])
CAMBER_MAX = float(os.environ.get('CAMBER_MAX', 'inf'))   # optional cap on the primaries' camber (same measure as reported)
CAMBER_MAX_SEC = float(os.environ.get('CAMBER_MAX_SEC', 'inf'))   # same for the secondaries (measured over 40 mm)
ok_s = lambda a, R: camber(a, R, 40) <= CAMBER_MAX_SEC
ok_p = lambda a, R: (not (DROOP == DROOP) or chord(a, R, LP) <= -DROOP) and camber(a, R, 50) <= CAMBER_MAX


def patch(aS, RS_, aP, RP_):
    sec0, pl0, fl0 = FLAT0
    zs, zp = bow(RS, aS, RS_), bow(RP, aP, RP_)
    kin.SEC_F = sec0 + np.c_[0 * zs, 0 * zs, zs]
    kin.PRI_F = fl0(0.0)[1] + np.c_[0 * zp, 0 * zp, zp]
    def pl(spread, dl, per=PER):
        X = pl0(spread, dl, per); X = X.copy(); X[..., 2] += zp; return X
    kin.pri_local = pl


def unpatch():
    kin.SEC_F, kin.pri_local = FLAT0[0], FLAT0[1]; kin.PRI_F = FLAT0[2](0.0)[1]


def fit_of(ev, d, prm):
    patch(*prm)
    try:
        F, fx = ev.fold_fit(d)
    finally:
        unpatch()
    return float(F[0]), fx


def camber(a, R, L):
    """arc height over the root-tip chord, / L (open wing: the feather keeps its shape)"""
    r = np.linspace(0, L, 200); z = bow(r, a, R); ch = z[-1] * r / L
    return float(np.max(np.abs(z - ch)) / L)


def rub(table, a, R, half=3.6, n=21):
    """max change during the fold (spread 1 -> 0) of the height difference of neighbouring feathers where they overlap in plan
    (in the bone frame: rotation about each root is in the wing plane, so only the bow matters)"""
    worst = 0.0
    for i in range(len(table) - 1):
        dz = []
        for sp in np.linspace(0, 1, n):
            P = []
            for f, L, a0, a1 in (table[i], table[i + 1]):
                ang = np.radians(a1 + (a0 - a1) * sp); d = np.array([np.cos(ang), np.sin(ang)])
                base = np.array([(LU if table is SEC else LMAN) * f, 0.0]); r = np.linspace(0.1, 1, 30) * L
                P.append((base, d, L))
            (b1, d1, L1), (b2, d2, L2) = P
            # points of feather 1 shaft; their distance across feather 2's shaft
            r1 = np.linspace(0.1, 1, 30) * L1; X = b1 + r1[:, None] * d1
            r2 = (X - b2) @ d2; off = np.abs((X - b2) @ np.array([-d2[1], d2[0]]))
            m = (off < 2 * half) & (r2 > 0) & (r2 < L2)
            if m.any(): dz.append((bow(r1[m], a, R) - bow(r2[m], a, R)))
            else: dz.append(np.array([]))
        allv = np.concatenate([v for v in dz if v.size]) if any(v.size for v in dz) else np.zeros(1)
        worst = max(worst, float(allv.max() - allv.min()))
    return worst


def main(ids):
    cands = json.load(open(OUT / 'candidates.json', encoding='utf-8'))
    if ids: cands = [c for c in cands if c['id'] in ids]
    out = {}
    for c in cands:
        ev = make_ev(c['ec'], c['wc'], 45, c['dsg'])
        F, G, e = ev(np.asarray(c['x'], float)[None], extra=True); d = e['d']
        f0, fx0 = fit_of(ev, d, (0, np.inf, 0, np.inf))
        best = (f0, (0, np.inf, 0, np.inf))
        # secondaries and primaries are almost independent: search each with the other flat, then refine jointly around the best
        bs = min(((fit_of(ev, d, (a, R, 0, np.inf))[0], a, R) for a in A_GRID for R in R_GRID if ok_s(a, R)))
        bp = min(((fit_of(ev, d, (bs[1], bs[2], a, R))[0], a, R) for a in A_GRID for R in R_GRID if ok_p(a, R)))
        prm = (bs[1], bs[2], bp[1], bp[2]); f1, fx1 = fit_of(ev, d, prm)
        sd = lambda fx: dict(bury_max=round(float(fx['bury_max'][0]), 2), float_max=round(float(fx['float_max'][0]), 2))
        r = dict(fit_flat=round(f0, 2), fit_bent=round(f1, 2), parts_flat=np.round(fx0['parts'][0], 2).tolist(),
                 parts_bent=np.round(fx1['parts'][0], 2).tolist(), flat=sd(fx0), bent=sd(fx1),
                 sec=dict(alpha_deg=round(float(np.degrees(prm[0])), 1), R_mm=None if np.isinf(prm[1]) else float(prm[1]),
                          camber=round(camber(prm[0], prm[1], 40), 3), chord_deg=round(chord(prm[0], prm[1], 40), 1), rub_mm=round(rub(SEC, prm[0], prm[1]), 2)),
                 pri=dict(alpha_deg=round(float(np.degrees(prm[2])), 1), R_mm=None if np.isinf(prm[3]) else float(prm[3]),
                          camber=round(camber(prm[2], prm[3], 50), 3), chord_deg=round(chord(prm[2], prm[3], LP), 1), rub_mm=round(rub(PRIM, prm[2], prm[3]), 2)))
        if DROOP == DROOP: r['droop_min_deg'] = DROOP
        out[c['id']] = r
        print(c['id'], c['dsg'], 'fit flat', r['fit_flat'], '-> bent', r['fit_bent'], 'flat', r['flat'], 'bent', r['bent'],
              'sec', r['sec'], 'pri', r['pri'], flush=True)
    json.dump(out, open(OUT / os.environ.get('FB_OUT', 'feather_bend.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main(IDS)
