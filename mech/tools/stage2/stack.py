"""Stage 2e: layer plan in real heights, fast enough for the optimisation loop (population-vectorised geometry + a small
backtracking search per design). The exact post-check is stack_check.py.

Frame: humerus frame, z = elbow axis. Parts:
- plates: humerus, ulna, bellcranks, planar rods 2/3 (flat, in the arm plane), the hand (flat, in its own plane through the
  wrist; it moves along the tilted wrist axis, approximated here by z), and rod 4 (wrist rod with rod ends; low force, so it
  may be cranked: its body is shifted by whole layers and short cranks / studs lead back to the ball centres).
- fixed: rod 1 (elbow drive from the body; carries the elbow load -> straight bamboo skewer, radius R_SKEW). It stays where
  the kinematics put it.
Each plate gets a level l in -3..3 (height l * LAYER_T on top of its own plane). Rules:
- two parts clash where their footprints (radius + GAP) overlap and their heights differ by less than one pitch LAYER_T;
- a pin (revolute) runs between the two plates; a ball joint has studs from each plate to the kinematic ball centre; a third
  part whose height lies strictly inside that span (pins flush with the outer faces) and whose footprint reaches the pin clashes;
- thickness (user decision 2026-11-14, option b): where the footprints of two link parts overlap, the local stack
  (height difference + plate thickness) must stay within BUD_BODY (body part) / BUD_WING (wing part). The hand / ulna pair is
  exempt (their separation is the designed tilt of the hand, judged by the fit to the body), and so is rod 1 (it comes up
  from the body anchor 10+ mm below the arm plane).
Returns per design the number of parts that could not be placed (0 = a clash-free plan within budget exists)."""
import numpy as np
from problem import LAYER_T, GAP, R_PIN, R_ROD, U

T = LAYER_T; TP = LAYER_T - GAP
LV = np.arange(-3, 4); NL = len(LV)
BUD_BODY, BUD_WING = 5.0, 3.0
R_SKEW = 1.5                       # 3 mm bamboo skewer
STEPS = np.arange(0, len(U), 2)
FIXED = ('rod1',)
ORDER = ['hum', 'uln', 'hand', 'bc1', 'rod2', 'bc2', 'rod3', 'rod4']
NODE_LIMIT = 20000
import os
GRADED = os.environ.get('STACK_GRADED') == '1'   # add a graded term (ball-stud thickness excess) to the unplaced count


def seg2(p1, q1, p2, q2):
    """closest points of two segments in the xy plane: distance, parameters s (on 1) and t (on 2)"""
    d1, d2, r = (q1 - p1)[..., :2], (q2 - p2)[..., :2], (p1 - p2)[..., :2]
    a, e, f = np.sum(d1 * d1, -1), np.sum(d2 * d2, -1), np.sum(d2 * r, -1)
    c, b = np.sum(d1 * r, -1), np.sum(d1 * d2, -1)
    den = a * e - b * b
    sc = np.where(den > 1e-9, np.clip((b * f - c * e) / np.where(den > 1e-9, den, 1), 0, 1), 0.0)
    tc = (b * sc + f) / np.maximum(e, 1e-12)
    sc = np.where(tc < 0, np.clip(-c / np.maximum(a, 1e-12), 0, 1), np.where(tc > 1, np.clip((b - c) / np.maximum(a, 1e-12), 0, 1), sc))
    tc = np.where(e > 1e-12, np.clip(tc, 0, 1), 0.0)
    sc = np.where(a > 1e-12, sc, 0.0)
    x1, x2 = p1[..., :2] + d1 * sc[..., None], p2[..., :2] + d2 * tc[..., None]
    return np.linalg.norm(x1 - x2, axis=-1), sc, tc


def joints(ec, wc, ep, wp, E, W):
    """(part a, part b, point, ball joint?)"""
    J = [('hum', 'uln', E, False), ('uln', 'hand', W, False)]
    if ec == 'direct': J += [('rod1', 'uln', ep['U'], True)]
    elif ec == 'hum': J += [('bc1', 'hum', ep['G1'], False), ('bc1', 'rod1', ep['B1'], True), ('bc1', 'rod2', ep['B2'], False), ('rod2', 'uln', ep['U'], False)]
    else: J += [('bc1', 'uln', ep['G1'], False), ('bc1', 'rod1', ep['B1'], True), ('bc1', 'rod2', ep['B2'], False), ('rod2', 'hum', ep['K'], False)]
    if wc == 'direct': J += [('rod4', 'hum', wp['B4'], True), ('rod4', 'hand', wp['Q'], True)]
    else:
        on, other = ('hum', 'uln') if wc == 'hum' else ('uln', 'hum')
        J += [('bc2', on, wp['G2'], False), ('rod3', other, wp['C3'], False), ('bc2', 'rod3', wp['B3'], False), ('bc2', 'rod4', wp['B4'], True), ('rod4', 'hand', wp['Q'], True)]
    return J


class Geometry:
    """all level-independent geometry of a population, and the constraint tables"""

    def __init__(self, ev, f, d, steps=None, node_limit=NODE_LIMIT):
        k = STEPS if steps is None else steps; self.node_limit = node_limit
        P_, adj, (S, E, W, Tt) = ev.parts(f, k=k)
        Rh = f['Rh'][:, k]
        h = lambda X: np.einsum('psji,psj->psi', Rh, X)
        self.P = Rh.shape[0]
        tw = d['tw'] if 'tw' in d else np.zeros(self.P)
        a_u = np.stack([0 * tw, -np.sin(tw), np.cos(tw)], -1)
        w = np.einsum('psji,psjk,pk->psi', Rh, f['Ru'][:, k], a_u); w *= np.sign(w[..., 2:3] + 1e-12)
        self.E, self.W, self.T = h(E), h(W), h(Tt)
        Wh = self.W
        plane_hand = lambda X: X - np.sum((X - Wh) * w, -1)[..., None] * w
        flat = {}
        for nm in P_:
            if nm in ('hum', 'uln', 'bc1', 'rod2', 'bc2', 'rod3'): flat[nm] = lambda X: X * np.array([1, 1, 0.])
            elif nm == 'hand': flat[nm] = plane_hand
            else: flat[nm] = lambda X: X                  # rod 4 (cranked, base = its kinematic line), rod 1 (fixed)
        self.caps = {}
        for nm, lst in P_.items():
            cs = []
            for j, (p, q, r) in enumerate(lst):
                if nm == 'rod1' and j == 0: r = R_SKEW
                cs.append((flat[nm](h(p)), flat[nm](h(q)), r))
            self.caps[nm] = cs
        self.names = list(P_)
        self.var = [n for n in ORDER if n in self.names]
        ep = {kk: h(v[:, k]) for kk, v in f['elbow_pts'].items()}; wp = {kk: h(v[:, k]) for kk, v in f['wrist_pts'].items()}
        self.J = joints(ev.ec, ev.wc, ep, wp, self.E, self.W)
        self.base = lambda nm, X: (plane_hand(X)[..., 2] if nm == 'hand' else X[..., 2] if nm in ('rod1', 'rod4') else 0 * X[..., 2])
        self.tables()

    def wing(self, X):
        def d(a, b):
            ab = b - a; t = np.clip(np.sum((X - a) * ab, -1) / np.maximum(np.sum(ab * ab, -1), 1e-12), 0, 1)
            return np.linalg.norm(X - a - ab * t[..., None], axis=-1)
        return np.minimum(d(self.E, self.W), d(self.W, self.T)) < d(0 * self.E, self.E)

    def tables(self):
        P = self.P; Lz = LV * T
        unary = {v: np.zeros((P, NL), bool) for v in self.var}
        binary = {}; ternary = {}
        def bin_(a, b):
            key = (a, b) if self.var.index(a) < self.var.index(b) else (b, a)
            if key not in binary: binary[key] = np.zeros((P, NL, NL), bool)
            return key
        # ---- part pairs
        for i, a in enumerate(self.names):
            for b in self.names[i + 1:]:
                if a in FIXED and b in FIXED: continue
                thick = {a, b} != {'hand', 'uln'} and 'rod1' not in (a, b)
                for pa, qa, ra in self.caps[a]:
                    for pb, qb, rb in self.caps[b]:
                        dd, s, t = seg2(pa, qa, pb, qb)
                        za = pa[..., 2] + s * (qa[..., 2] - pa[..., 2]); zb = pb[..., 2] + t * (qb[..., 2] - pb[..., 2])
                        xa = pa + s[..., None] * (qa - pa); xb = pb + t[..., None] * (qb - pb)
                        bud = np.where(self.wing((xa + xb) / 2), BUD_WING, BUD_BODY)
                        near = dd < ra + rb + GAP; tight = dd < ra + rb
                        if not (near.any()): continue
                        if a in FIXED or b in FIXED:          # one fixed: unary on the other
                            v, zv, zf = (b, zb, za) if a in FIXED else (a, za, zb)
                            dz = (zv[..., None] + Lz) - zf[..., None]                       # (P,S,NL)
                            bad = (near[..., None] & (np.abs(dz) < T - 1e-6))
                            unary[v] |= bad.any(1)
                        else:
                            key = bin_(a, b); sw = key != (a, b)
                            for x, la in enumerate(Lz):
                                dz = (za + la)[..., None] - (zb[..., None] + Lz)           # (P,S,NL) over lb
                                bad = near[..., None] & (np.abs(dz) < T - 1e-6)
                                if thick: bad |= tight[..., None] & (np.abs(dz) + TP > bud[..., None] + 1e-6)
                                bb = bad.any(1)
                                if sw: binary[key][:, :, x] |= bb
                                else: binary[key][:, x, :] |= bb
        # ---- pins and studs
        for a, b, X, ball in self.J:
            zX = X[..., 2]
            ba, bb_ = self.base(a, X), self.base(b, X)
            bud = np.where(self.wing(X), BUD_WING, BUD_BODY)
            la_set = Lz if a not in FIXED else np.array([0.])
            lb_set = Lz if b not in FIXED else np.array([0.])
            # span[x, y] -> (P,S) lo / hi
            lo = np.empty((len(la_set), len(lb_set)) + zX.shape); hi = np.empty_like(lo)
            for x, la in enumerate(la_set):
                for y, lb in enumerate(lb_set):
                    za, zb = ba + la, bb_ + lb
                    lo[x, y] = np.minimum(za, zb); hi[x, y] = np.maximum(za, zb)
                    if ball: lo[x, y] = np.minimum(lo[x, y], zX); hi[x, y] = np.maximum(hi[x, y], zX)
            # thickness of a ball joint's stack
            if ball:
                bad = ((hi - lo + TP) > bud + 1e-6).any(-1)                       # (na, nb, P)
                self._put(unary, binary, ternary, a, b, None, np.moveaxis(bad, -1, 0))
            for c in self.names:
                if c in (a, b): continue
                for pc, qc, rc in self.caps[c]:
                    dd, s, t = seg2(X, X, pc, qc)
                    near = dd < R_PIN + rc + GAP
                    if not near.any(): continue
                    zc = pc[..., 2] + t * (qc[..., 2] - pc[..., 2])
                    lc_set = Lz if c not in FIXED else np.array([0.])
                    zcl = zc[None] + lc_set[:, None, None]                    # (nc,P,S)
                    bad = near[None, None, None] & (lo[:, :, None] - T + 1e-6 < zcl[None, None]) & (zcl[None, None] < hi[:, :, None] + T - 1e-6)
                    bad = np.moveaxis(bad.any(-1), -1, 0)                                # (P, na, nb, nc)
                    self._put(unary, binary, ternary, a, b, c, bad)
        self.unary, self.binary, self.ternary = unary, binary, ternary

    def _put(self, unary, binary, ternary, a, b, c, bad):
        """bad: (P, na, nb[, nc]) with size-1 axes for fixed parts -> store on the free parts"""
        vs = [a, b] + ([c] if c is not None else [])
        free = [i for i, v in enumerate(vs) if v not in FIXED]
        bad = bad.reshape((self.P,) + tuple(bad.shape[1:]))
        for i in reversed(range(len(vs))):
            if vs[i] in FIXED: bad = bad.take(0, axis=i + 1)
        fv = [vs[i] for i in free]
        if len(fv) == 0: return
        if len(fv) == 1: unary[fv[0]] |= bad; return
        idx = sorted(range(len(fv)), key=lambda i: self.var.index(fv[i]))
        assert bad.ndim == len(fv) + 1, (a, b, c, bad.shape, vs)
        bad = np.transpose(bad, [0] + [i + 1 for i in idx]); key = tuple(fv[i] for i in idx)
        if len(key) == 2:
            if key not in binary: binary[key] = np.zeros((self.P, NL, NL), bool)
            binary[key] |= bad
        else:
            if key not in ternary: ternary[key] = np.zeros((self.P, NL, NL, NL), bool)
            ternary[key] |= bad

    def solve(self, p):
        """backtracking for design p -> (number of parts left unplaced at the deepest point, plan)"""
        var = self.var; n = len(var); pos = {v: i for i, v in enumerate(var)}
        full = (1 << NL) - 1
        un = [full & ~int(sum(1 << j for j in np.nonzero(self.unary[v][p])[0])) for v in var]
        bins = [[] for _ in range(n)]
        for (a, b), tab in self.binary.items():
            t = tab[p]; ia, ib = pos[a], pos[b]
            bins[ib].append((ia, [full & ~int(sum(1 << j for j in np.nonzero(t[x])[0])) for x in range(NL)]))
        ters = [[] for _ in range(n)]
        for (a, b, c), tab in self.ternary.items():
            t = tab[p]; ia, ib, ic = pos[a], pos[b], pos[c]
            m = [[full & ~int(sum(1 << j for j in np.nonzero(t[x, y])[0])) for y in range(NL)] for x in range(NL)]
            ters[ic].append((ia, ib, m))
        lev = [0] * n; best = [0, None]; nodes = [0]
        order = sorted(range(NL), key=lambda j: abs(LV[j]))
        def bt(i):
            if i > best[0]: best[0] = i; best[1] = list(lev[:i])
            if i == n: return True
            nodes[0] += 1
            if nodes[0] > self.node_limit: return False
            dom = un[i]
            for ia, m in bins[i]: dom &= m[lev[ia]]
            for ia, ib, m in ters[i]: dom &= m[lev[ia]][lev[ib]]
            for j in order:
                if dom >> j & 1:
                    lev[i] = j
                    if bt(i + 1): return True
            return False
        bt(0)
        plan = {var[i]: int(LV[l]) for i, l in enumerate(best[1] or [])}
        return n - best[0], plan


def full_check(ev, x):
    """post-check of one design: all 41 poses, no search limit -> (unplaced parts, plan, stud / pin lengths in mm)"""
    F, G, e = ev(np.asarray(x)[None], extra=True)
    g = Geometry(ev, e['f'], e['d'], steps=np.arange(len(U)), node_limit=10 ** 7)
    v, plan = g.solve(0)
    studs = {}
    if v == 0:
        for a, b, X, ball in g.J:
            za = g.base(a, X)[0] + plan.get(a, 0) * T; zb = g.base(b, X)[0] + plan.get(b, 0) * T
            zz = [za, zb] + ([X[0, :, 2]] if ball else [])
            studs[f'{a}-{b}'] = round(float(np.max(np.max(zz, 0) - np.min(zz, 0))), 2)
    return int(v), plan, studs


def ball_excess(g):
    """(P,) mm: for every ball joint, how far the best level of its plate partner(s) leaves the stud stack over budget"""
    ex = np.zeros(g.P)
    for a, b, X, ball in g.J:
        if not ball: continue
        bud = np.where(g.wing(X), BUD_WING, BUD_BODY).min(1)                  # (P,)
        for c in (a, b):
            if c in ('rod1', 'rod4'): continue
            dz = np.abs(X[..., 2][..., None] - g.base(c, X)[..., None] - LV * T).max(1)   # (P, NL)
            ex += np.maximum(0, dz.min(1) + TP - bud)
    return ex


def stack_constraint(ev, f, d):
    G = Geometry(ev, f, d)
    out = np.zeros(G.P); plans = []
    for p in range(G.P):
        v, plan = G.solve(p); out[p] = v; plans.append(plan)
    if GRADED:
        out = out + (out > 0) * np.minimum(0.9, 0.1 * ball_excess(G))
    return out, plans
