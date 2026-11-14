"""Stage-2d check: the layer plan in real heights (humerus frame, z = elbow axis).

The layer count used in the optimisation only says which parts must not share a layer. Here every part gets an actual height:
- planar parts (bones, bellcranks, the planar rods 2 and 3) are flat plates; each may be shifted along its own hinge axis by a
  whole number of layers (pitch LAYER_T); this does not change the kinematics. The hand shifts along the tilted wrist axis.
- spatial rods (rod 1 from the body, rod 4 with rod ends) stay where the kinematics put them (their ball centres are fixed).
- a pin between two plates runs straight from one plate to the other; a ball on a plate sits on a stud from the plate to the
  kinematic ball centre. Any third part that a pin / stud passes through is a clash (pins end flush with the outer faces).
- two parts clash if their footprints (capsule radius + GAP) overlap and their heights differ by less than one layer pitch.
All level choices (-3..3 layers per plate) are enumerated. Reported: clash-free or not, and the smallest stack thickness
(local: plates whose footprints overlap) in the body part and the wing part. Rod 1 (from the body anchor, which lies 10+ mm
below the arm plane) is checked for clashes but not counted in the thickness.
usage: stack_check.py [ids...]   -> results/stage2/2d/stack_check.json"""
import sys, json, itertools, pathlib, numpy as np
from problem import Evaluator, LAYER_T, GAP, R_PIN, U

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / 'results' / 'stage2' / '2d'
LV = np.arange(-3, 4)                    # level choices for a plate
TP = LAYER_T - GAP                       # plate thickness
STEP_MM = 0.5
FIXED = ('rod1', 'rod4')                 # spatial rods (ball ends)


def joints(ec, wc, ep, wp, E, W):
    J = [('hum', 'uln', E), ('uln', 'hand', W)]
    if ec == 'direct': J += [('rod1', 'uln', ep['U'])]
    elif ec == 'hum': J += [('bc1', 'hum', ep['G1']), ('bc1', 'rod1', ep['B1']), ('bc1', 'rod2', ep['B2']), ('rod2', 'uln', ep['U'])]
    else: J += [('bc1', 'uln', ep['G1']), ('bc1', 'rod1', ep['B1']), ('bc1', 'rod2', ep['B2']), ('rod2', 'hum', ep['K'])]
    if wc == 'direct': J += [('rod4', 'hum', wp['B4']), ('rod4', 'hand', wp['Q'])]
    else:
        on, other = ('hum', 'uln') if wc == 'hum' else ('uln', 'hum')
        J += [('bc2', on, wp['G2']), ('rod3', other, wp['C3']), ('bc2', 'rod3', wp['B3']), ('bc2', 'rod4', wp['B4']), ('rod4', 'hand', wp['Q'])]
    return J


def sample(p, q, r):
    """points along a capsule axis (steps, n, 3) and radius"""
    L = np.linalg.norm(q - p, axis=-1).max()
    n = max(1, int(np.ceil(L / STEP_MM)) + 1)
    t = np.linspace(0, 1, n)
    return p[:, None] + (q - p)[:, None] * t[None, :, None], np.full(n, r)


class Stack:
    def __init__(self, c):
        self.c = c
        ev = Evaluator(c['ec'], c['wc'], 45, tilt=True)
        F, G, e = ev(np.array([c['x']]), extra=True); f = e['f']
        k = np.arange(len(U))
        P_, adj, (S, E, W, T) = ev.parts(f, k=k)
        Rh = f['Rh'][0]
        h = lambda X: np.einsum('sji,sj->si', Rh, X[0] if X.ndim == 3 else X)
        tw = float(e['d']['tw'][0])
        a_u = np.array([0, -np.sin(tw), np.cos(tw)])
        w = np.einsum('sji,sjk,k->si', Rh, f['Ru'][0], a_u)                 # wrist hinge axis in the humerus frame
        w *= np.sign(w[:, 2:3])
        self.Eh, self.Wh, self.Th = h(E), h(W), h(T)
        self.names = list(P_)
        self.plane = {nm: ('fixed' if nm in FIXED else 'hand' if nm == 'hand' else 'z') for nm in self.names}
        Wh = self.Wh
        def flat(nm, X):                                  # project onto the part's own plate (level 0)
            if self.plane[nm] == 'z': return X * np.array([1, 1, 0])
            if self.plane[nm] == 'hand': return X - np.sum((X - Wh) * w, -1)[..., None] * w
            return X
        self.shift = {nm: (np.tile([0, 0, 1.], (len(k), 1)) if self.plane[nm] == 'z' else w if self.plane[nm] == 'hand' else np.zeros((len(k), 3)))
                      for nm in self.names}
        self.lv = {nm: (np.array([0]) if self.plane[nm] == 'fixed' else LV) for nm in self.names}
        # point clouds at level 0
        self.pts = {}
        for nm in self.names:
            ps, rs = [], []
            for p, q, r in P_[nm]:
                a, b = flat(nm, h(p)), flat(nm, h(q))
                x, rr = sample(a, b, r); ps.append(x); rs.append(rr)
            self.pts[nm] = (np.concatenate(ps, 1), np.concatenate(rs))
        ep = {kk: h(v) for kk, v in f['elbow_pts'].items()}; wp = {kk: h(v) for kk, v in f['wrist_pts'].items()}
        self.J = [(a, b, flat(a, X), flat(b, X)) for a, b, X in joints(c['ec'], c['wc'], ep, wp, self.Eh, self.Wh)]
        pairs = set(tuple(sorted(p)) for p in adj)
        assert pairs == set(tuple(sorted((a, b))) for a, b, *_ in self.J), (pairs, [(a, b) for a, b, *_ in self.J])

    def wing(self, X):
        d = lambda a, b: np.linalg.norm(X - (a + (b - a) * np.clip(np.sum((X - a) * (b - a), -1) / np.sum((b - a) ** 2, -1), 0, 1)[..., None]), axis=-1)
        sE, sW, sT = self.Eh[:, None], self.Wh[:, None], self.Th[:, None]
        return np.minimum(d(sE, sW), d(sW, sT)) < d(0 * sE, sE)

    def pair_tables(self, a, b):
        """clash (bool) and local thickness in body / wing (mm) for each level pair"""
        (pa, ra), (pb, rb) = self.pts[a], self.pts[b]
        la, lb = self.lv[a], self.lv[b]
        d0 = pa[:, :, None, :2] - pb[:, None, :, :2]
        dxy0 = np.linalg.norm(d0, axis=-1)
        cand = dxy0 < (ra[:, None] + rb[None] + GAP + 1.5)[None]
        s, i, j = np.nonzero(cand)
        clash = np.zeros((len(la), len(lb)), bool); thb = np.zeros_like(clash, float); thw = np.zeros_like(clash, float)
        if len(s) == 0: return clash, thb, thw
        A, B = pa[s, i], pb[s, j]; rr = ra[i] + rb[j]
        for x, l1 in enumerate(la):
            for y, l2 in enumerate(lb):
                Aa = A + l1 * LAYER_T * self.shift[a][s]; Bb = B + l2 * LAYER_T * self.shift[b][s]
                dxy = np.linalg.norm(Aa[:, :2] - Bb[:, :2], axis=-1); dz = np.abs(Aa[:, 2] - Bb[:, 2])
                clash[x, y] = np.any((dxy < rr + GAP) & (dz < LAYER_T - 1e-9))
                ov = dxy < rr
                if ov.any():
                    wg = self._wing_pts(s, (Aa + Bb) / 2)
                    t = dz + TP
                    thb[x, y] = t[ov & ~wg].max(initial=0); thw[x, y] = t[ov & wg].max(initial=0)
        return clash, thb, thw

    def _wing_pts(self, s, X):
        E, W, T = self.Eh[s], self.Wh[s], self.Th[s]
        d = lambda a, b: np.linalg.norm(X - (a + (b - a) * np.clip(np.sum((X - a) * (b - a), -1) / np.sum((b - a) ** 2, -1), 0, 1)[:, None]), axis=-1)
        return np.minimum(d(E, W), d(W, T)) < d(0 * E, E)

    def pin_table(self, jn, c):
        """clash of the pin / stud of joint jn with part c, for each (level a, level b, level c)"""
        a, b, Xa, Xb = self.J[jn]
        pc, rc = self.pts[c]
        la, lb, lc = self.lv[a], self.lv[b], self.lv[c]
        tab = np.zeros((len(la), len(lb), len(lc)), bool)
        near = np.linalg.norm(pc[:, :, :2] - Xa[:, None, :2], axis=-1) < R_PIN + rc[None] + GAP + 3.0
        s, i = np.nonzero(near)
        if len(s) == 0: return tab
        Cp = pc[s, i]; r = R_PIN + rc[i] + GAP
        t = np.linspace(0, 1, 25)
        for x, l1 in enumerate(la):
            for y, l2 in enumerate(lb):
                A = Xa[s] + l1 * LAYER_T * self.shift[a][s]; B = Xb[s] + l2 * LAYER_T * self.shift[b][s]
                ext = (B - A); L = np.linalg.norm(ext, axis=-1, keepdims=True); u = np.where(L > 1e-9, ext / np.maximum(L, 1e-9), 0)
                A2 = A - u * TP / 2; B2 = B + u * TP / 2                     # flush with the outer faces of both parts
                pin = A2[:, None] + (B2 - A2)[:, None] * t[None, :, None]   # (m, 25, 3)
                for z, l3 in enumerate(lc):
                    C = (Cp + l3 * LAYER_T * self.shift[c][s])[:, None]
                    dxy = np.linalg.norm(pin[..., :2] - C[..., :2], axis=-1); dz = np.abs(pin[..., 2] - C[..., 2])
                    tab[x, y, z] = np.any((dxy < r[:, None]) & (dz < (LAYER_T + GAP) / 2 - 1e-9) & (L > 0.3))
        return tab

    def solve(self):
        nm = self.names; n = len(nm)
        free = [k for k in nm if self.plane[k] != 'fixed']
        idx = {k: i for i, k in enumerate(nm)}
        combos = np.array(list(itertools.product(*[range(len(self.lv[k])) for k in nm])), np.int8)   # level index per part
        nclash = np.zeros(len(combos), np.int16); TB = np.zeros(len(combos)); TW = np.zeros(len(combos))
        why = {}; fixed_clash = []
        for a, b in itertools.combinations(nm, 2):
            cl, tb, tw_ = self.pair_tables(a, b)
            ia, ib = combos[:, idx[a]], combos[:, idx[b]]
            if self.plane[a] == 'fixed' and self.plane[b] == 'fixed':      # neither can move: a real clash whatever the levels
                if cl.any(): fixed_clash.append(f'{a} / {b}')
                cl = np.zeros_like(cl)
            hit = cl[ia, ib]; nclash += hit; why[(a, b)] = hit
            if 'rod1' not in (a, b): TB = np.maximum(TB, tb[ia, ib]); TW = np.maximum(TW, tw_[ia, ib])
        for jn, (a, b, *_ ) in enumerate(self.J):
            for c in nm:
                if c in (a, b): continue
                tab = self.pin_table(jn, c)
                hit = tab[combos[:, idx[a]], combos[:, idx[b]], combos[:, idx[c]]]; nclash += hit; why[(f'pin {a}-{b}', c)] = hit
        # pinned plates must not sit on the same level (covered by the footprint test at the joint) ; pick the best plan
        ok = nclash == 0
        res = dict(id=self.c['id'], ec=self.c['ec'], wc=self.c['wc'], parts=nm, fixed_clash=fixed_clash, n_plans_ok=int(ok.sum()), n_plans=len(combos))
        if ok.any():
            score = np.maximum(TB / 5.0, TW / 3.0) + 1e-3 * (TB + TW)
            score[~ok] = np.inf; i = int(np.argmin(score))
            res.update(feasible=True, body_mm=float(TB[i]), wing_mm=float(TW[i]),
                       within_budget=bool(TB[i] <= 5.0 + 1e-6 and TW[i] <= 3.0 + 1e-6),
                       n_plans_within_budget=int(np.sum(ok & (TB <= 5.0 + 1e-6) & (TW <= 3.0 + 1e-6))))
        else:
            i = int(np.argmin(nclash + 1e-3 * np.maximum(TB, TW)))
            res.update(feasible=False, min_clashes=int(nclash[i]), body_mm=float(TB[i]), wing_mm=float(TW[i]),
                       clashes=[f'{k[0]} / {k[1]}' for k, v in why.items() if v[i]])
        res['levels'] = {k: int(self.lv[k][combos[i, idx[k]]]) for k in nm}
        res['studs_mm'] = {f'{a}-{b}': round(float(np.max(np.linalg.norm((Xb + self.lv[b][combos[i, idx[b]]] * LAYER_T * self.shift[b])
                                                                           - (Xa + self.lv[a][combos[i, idx[a]]] * LAYER_T * self.shift[a]), axis=-1))), 2)
                           for a, b, Xa, Xb in self.J}
        return res


def main():
    cands = json.load(open(OUT / 'candidates.json', encoding='utf-8'))
    ids = sys.argv[1:] or [c['id'] for c in cands]
    out = {}
    for c in cands:
        if c['id'] not in ids: continue
        r = Stack(c).solve(); out[c['id']] = r
        print(json.dumps(r, ensure_ascii=False), flush=True)
    path = OUT / 'stack_check.json'
    old = json.load(open(path, encoding='utf-8')) if path.exists() else {}
    old.update(out); json.dump(old, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
