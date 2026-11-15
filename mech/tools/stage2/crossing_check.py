"""Independent check of parts that cross in plan view (one wing, all 41 poses, layer plan of check_2e.json).
Every part is a chain of short slices with a footprint radius r (in the arm plane) and a half height h:
plates (bones, bellcranks, rods 2 / 4): h = TP / 2 at their level; rod 1 (3 mm bamboo skewer): a round rod, h = r = R_SKEW at its
kinematic height (it is not moved). Two slices touch when they overlap in plan (r1 + r2 + GAP) AND in height (h1 + h2 + GAP).
The clearance of a slice pair is the larger of the two gaps (either one apart is enough). Pairs that share a joint are skipped
within JR mm (plan) of that joint. Unlike stack.py this does not use a fixed layer pitch for the skewer: its true diameter counts.
usage: crossing_check.py   (MIDLINE=1 for stage 2f)  -> results/stage2/<STAGE>/crossing_check.json"""
import sys, json, numpy as np
sys.argv = sys.argv[:1] + ['45', '2e']
from analyze import make_ev, OUT
from problem import U, GAP
from stack import Geometry, T, TP, R_SKEW

NS = 24            # slices per segment
JR = 3.0           # mm around a shared joint where the two parts are meant to meet


def slices(p, q, n=NS):
    t = np.linspace(0, 1, n)[:, None, None]
    return p[None] + t * (q - p)[None]                          # (n, S, 3)


def check(c, plan):
    ev = make_ev(c['ec'], c['wc'], 45, c['dsg'])
    F, G, e = ev(np.asarray(c['x'], float)[None], extra=True)
    g = Geometry(ev, e['f'], e['d'], steps=np.arange(len(U)), node_limit=1)
    parts = {}
    for nm, caps in g.caps.items():
        lst = []
        for j, (p, q, r) in enumerate(caps):
            X = slices(p[0], q[0])                                  # humerus frame; plates flattened (z = their own plane)
            if nm == 'rod1':
                z = X[..., 2]; h = np.full_like(z, R_SKEW)
            else:
                z = X[..., 2] + plan.get(nm, 0) * T; h = np.full_like(z, TP / 2)
            lst.append((X[..., :2], z, np.full_like(z, r), h))
        parts[nm] = lst
    joints = {}
    for a, b, X, ball in g.J:
        joints.setdefault(frozenset((a, b)), []).append(X[0][..., :2])   # (S, 2)
    names = list(parts); res = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if {a, b} == {'hand', 'uln'}: continue                 # hand / ulna: their separation is the wrist hinge itself
            js = joints.get(frozenset((a, b)), [])
            best = (np.inf, None)
            for xa, za, ra, ha in parts[a]:
                for xb, zb, rb, hb in parts[b]:
                    dxy = np.linalg.norm(xa[:, None] - xb[None], axis=-1)              # (na, nb, S)
                    gp = dxy - ra[:, None] - rb[None] - GAP
                    gz = np.abs(za[:, None] - zb[None]) - ha[:, None] - hb[None] - GAP
                    cl = np.maximum(gp, gz)
                    for J in js:                                   # near a shared joint: by design
                        near = (np.linalg.norm(xa - J[None], axis=-1)[:, None] < JR) | (np.linalg.norm(xb - J[None], axis=-1)[None] < JR)
                        cl = np.where(near, np.inf, cl)
                    k = np.unravel_index(np.argmin(cl), cl.shape)
                    if cl[k] < best[0]:
                        best = (float(cl[k]), dict(u=round(float(U[k[2]]), 3), plan_gap=round(float(gp[k]), 2), z_gap=round(float(gz[k]), 2)))
            if best[0] < 1.0:
                res[f'{a}-{b}'] = dict(clear=round(best[0], 2), **best[1])
    return res


def main():
    cands = json.load(open(OUT / 'candidates.json', encoding='utf-8'))
    chk = json.load(open(OUT / 'check_2e.json', encoding='utf-8'))
    out = {}
    for c in cands:
        r = check(c, chk[c['id']]['plan']); out[c['id']] = r
        bad = {k: v for k, v in r.items() if v['clear'] < 0}
        print(c['id'], c['dsg'], 'plan', chk[c['id']]['plan'], 'CLASH' if bad else 'ok', bad if bad else
              {k: v['clear'] for k, v in sorted(r.items(), key=lambda t: t[1]['clear'])[:3]}, flush=True)
    json.dump(out, open(OUT / 'crossing_check.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
