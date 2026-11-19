"""Left / right wings: clearance between the parts of one wing and the mirror image of the other (both wings driven together,
same u). Body midline: x = MID in the shoulder frame (centre of the CG torso sections, = -SX of the CT shoulder rule).
Plates (hum, uln, bellcranks, their rods) are moved into the humerus plane and then by their level of the stage-2e plan along the
humerus z axis, as stack.py places them; rods 1 / 4 and the hand keep their kinematic position plus the level offset (hand: along
its plane normal, approximated by the humerus z axis).
usage: mirror_check.py  -> results/stage2/2e/mirror_check.json"""
import json, pathlib, numpy as np
from analyze import make_ev
from problem import U
from stack import seg2, T, R_SKEW
from problem import GAP
ROOT = pathlib.Path(__file__).resolve().parents[2]
from problem import STAGE, TAG
OUT = ROOT / 'results' / 'stage2' / (STAGE + TAG)
MID = -12.7


def build(c, plan):
    """capsules (name, p, q, r) of one wing over all 41 poses, placed as stack.py places them; p, q: (S, 3)"""
    ev = make_ev(c['ec'], c['wc'], 45, c['dsg'])
    F, G, e = ev(np.asarray(c['x'], float)[None], extra=True); f = e['f']
    P_, adj, _ = ev.parts(f, k=np.arange(len(U)))
    zax = f['Rh'][0][:, :, 2]                       # humerus z axis in the body frame, per step (S,3)
    flat = ('hum', 'uln', 'bc1', 'rod2', 'bc2', 'rod3')     # plates: into the humerus plane (as stack.py), then to their level
    caps = []
    for nm, lst in P_.items():
        pos = lambda X: X + (plan.get(nm, 0) * T - (np.sum(X * zax, -1) if nm in flat else 0 * X[:, 0]))[:, None] * zax
        for j, (p, q, r) in enumerate(lst):
            if nm == 'rod1' and j == 0: r = R_SKEW
            caps.append((nm, pos(p[0]), pos(q[0]), r))
    return caps, f


mir = lambda X: X * np.array([-1, 1, 1]) + np.array([2 * MID, 0, 0])


def main():
    cands = json.load(open(OUT / 'candidates.json', encoding='utf-8'))
    chk = json.load(open(OUT / 'check_2e.json', encoding='utf-8'))
    res = {}
    for c in cands:
        plan = chk[c['id']]['plan']
        caps, f = build(c, plan)
        worst = {}
        for na, pa, qa, ra in caps:
            for nb, pb, qb, rb in caps:
                d, _, _ = seg2(pa, qa, mir(pb), mir(qb))
                cl = d - ra - rb                          # surface clearance per step
                key = tuple(sorted((na, nb)))
                i = int(np.argmin(cl))
                if key not in worst or cl[i] < worst[key][0]: worst[key] = (float(cl[i]), i)
        A = np.asarray(f['elbow_pts']['A'])[0, 0]
        mn = min(worst.items(), key=lambda t: t[1][0])
        res[c['id']] = dict(anchor_A=[round(float(v), 1) for v in A], anchor_from_mid=round(float(A[0] - MID), 1),
                            min_clear=round(mn[1][0], 2), pair=mn[0], u=round(float(U[mn[1][1]]), 3),
                            clashes={f'{a}-{b}': round(v[0], 2) for (a, b), v in sorted(worst.items(), key=lambda t: t[1][0]) if v[0] < GAP})
        print(c['id'], c['dsg'], res[c['id']], flush=True)
    json.dump(res, open(OUT / 'mirror_check.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
