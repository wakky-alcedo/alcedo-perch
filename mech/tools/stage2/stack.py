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
import os, numpy as np
from problem import LAYER_T, GAP, R_PIN, R_ROD, U

T = LAYER_T; TP = LAYER_T - GAP
LV = np.arange(-3, 4); NL = len(LV)
BUD_BODY, BUD_WING = 5.0, 3.0
BUD_JOINT = float(os.environ.get("STACK_BUD_JOINT", 4.0))   # user 2026-11-15: at a ball joint the local stack may be 4 mm (or more) on the wing side too
R_SKEW = 1.5                       # 3 mm bamboo skewer
STEPS = np.arange(0, len(U), 2)
FIXED = ('rod1',)
ORDER = ['hum', 'uln', 'hand', 'bc1', 'rod2', 'bc2', 'rod3', 'rod4']
NODE_LIMIT = 20000
# half height of each part: plates TP / 2; rod 1 is a round skewer (stage 2f fix: before, every pair only needed one pitch T apart,
# and plan overlap was taken from 3D distance with the plates at their own plane, which missed the skewer crossing a shifted plate)
HALF = {'rod1': R_SKEW}
SLOPED, NSUB = ('rod1', 'rod4', 'hand'), int(os.environ.get('STACK_NSUB', 8))
hh = lambda n: HALF.get(n, TP / 2)
FLAT = np.array([1., 1., 0.])
GRADED = os.environ.get('STACK_GRADED') == '1'   # add a graded term (ball-stud thickness excess) to the unplaced count
JR_CLIP = 3.0                                        # = crossing_check.JR
JB_ZONE = 5.0                                        # mm (plan) around a ball joint where its stack budget applies to the two parts it joins
CLIP = os.environ.get('STACK_CLIP', '1') == '1'      # exact sloped-rod / flat-plate test (clip_tables); 0 = before 2026-11-17


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
        self.caps = {}; self.raw = {}
        for nm, lst in P_.items():
            cs = []; self.raw[nm] = []
            for j, (p, q, r) in enumerate(lst):
                if nm == 'rod1' and j == 0: r = R_SKEW
                P0, Q0 = flat[nm](h(p)), flat[nm](h(q)); self.raw[nm].append((P0, Q0, r))
                if nm in SLOPED and not np.array_equal(P0, Q0):     # (end caps of zero length: nothing to split)                          # height varies along it: one closest point per pair is not enough
                    for a0, a1 in zip(np.linspace(0, 1, NSUB + 1)[:-1], np.linspace(0, 1, NSUB + 1)[1:]):
                        cs.append((P0 + a0 * (Q0 - P0), P0 + a1 * (Q0 - P0), r))
                else:
                    cs.append((P0, Q0, r))
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

    def wing_at(self, X, p, s):
        """wing() for scattered points X (N,3) of design p, pose s (N,)"""
        E, W, T_ = self.E[p, s], self.W[p, s], self.T[p, s]
        def d(a, b):
            ab = b - a; t = np.clip(np.sum((X - a) * ab, -1) / np.maximum(np.sum(ab * ab, -1), 1e-12), 0, 1)
            return np.linalg.norm(X - a - ab * t[..., None], axis=-1)
        return np.minimum(d(E, W), d(W, T_)) < d(0 * E, E)

    def tables(self):
        if os.environ.get('STACK_REF') == '1': return self.tables_ref()
        return self.tables_fast()

    def tables_fast(self):
        """same tables as tables_ref (2026-11-16 speed-up, checked equal): each cap of a part against all caps of the other part
        in one call, and the level tests only on the (design, pose, cap) entries that overlap in plan"""
        P = self.P; Lz = LV * T
        unary = {v: np.zeros((P, NL), bool) for v in self.var}
        binary = {}; ternary = {}
        stk = {nm: (np.stack([c[0] for c in cs]), np.stack([c[1] for c in cs]), np.array([c[2] for c in cs], float))
               for nm, cs in self.caps.items()}

        def orby(pidx, B):
            """OR of the rows of B grouped by design (pidx sorted) -> (P, ...)"""
            out = np.zeros((P,) + B.shape[1:], bool)
            if len(pidx): u, st = np.unique(pidx, return_index=True); out[u] = np.logical_or.reduceat(B, st, axis=0)
            return out

        mh = lambda p, q: ((p[..., :2] + q[..., :2]) / 2, np.linalg.norm((q - p)[..., :2], axis=-1) / 2)   # plan midpoint, half length
        MH = {nm: mh(v[0], v[1]) for nm, v in stk.items()}

        def cand(m, h, nm, lim):
            """(design, cap, pose) entries of part nm whose plan distance to the segment (m, h) may be < lim -- a lower bound of the
            distance from the midpoints, so seg2 runs only on these (exact same values as running it everywhere); sorted by design"""
            mB, hB = MH[nm]
            lb = np.linalg.norm(m[None] - mB, axis=-1) - h[None] - hB
            p, k, si = np.nonzero(np.moveaxis(lb < lim + 1e-6, 0, 1)); return p, k, si

        for i, a in enumerate(self.names):
            for b in self.names[i + 1:]:
                if a in FIXED and b in FIXED: continue
                thick = {a, b} != {'hand', 'uln'} and 'rod1' not in (a, b)
                PB, QB, RB = stk[b]; PBf, QBf = PB * FLAT, QB * FLAT
                thr = hh(a) + hh(b) + GAP - 1e-6
                for pa, qa, ra in self.caps[a]:
                    mA, hA = mh(pa, qa)
                    p, k, si = cand(mA, hA, b, (ra + RB).max() + GAP)
                    dd, s_, t_ = seg2(pa[p, si] * FLAT, qa[p, si] * FLAT, PBf[k, p, si], QBf[k, p, si])
                    near = dd < ra + RB[k] + GAP
                    if not near.any(): continue
                    p, k, si, dd, s_, t_ = p[near], k[near], si[near], dd[near], s_[near], t_[near]
                    pa_, qa_, pb_, qb_ = pa[p, si], qa[p, si], PB[k, p, si], QB[k, p, si]
                    za = pa_[:, 2] + s_ * (qa_[:, 2] - pa_[:, 2]); zb = pb_[:, 2] + t_ * (qb_[:, 2] - pb_[:, 2])
                    if a in FIXED or b in FIXED:
                        v, zv, zf = (b, zb, za) if a in FIXED else (a, za, zb)
                        dz = (zv[:, None] + Lz) - zf[:, None]
                        unary[v] |= orby(p, np.abs(dz) < thr)
                    else:
                        key = bin_key = (a, b) if self.var.index(a) < self.var.index(b) else (b, a)
                        if key not in binary: binary[key] = np.zeros((P, NL, NL), bool)
                        dz = (za[:, None, None] + Lz[None, :, None]) - (zb[:, None, None] + Lz[None, None, :])   # (N, la, lb)
                        bad = np.abs(dz) < thr
                        if thick:
                            tight = dd < (ra + RB)[k]
                            if tight.any():
                                xa = pa_ + s_[:, None] * (qa_ - pa_); xb = pb_ + t_[:, None] * (qb_ - pb_)
                                mid = (xa + xb) / 2; wg = self.wing_at(mid, p, si)
                                bud = np.where(wg, BUD_WING, BUD_BODY)
                                if CLIP:                  # 2026-11-17: next to a ball joint the two share, the joint budget
                                    for X in [X for a_, b_, X, ball in self.J if ball and {a_, b_} == {a, b}]:
                                        nj = np.linalg.norm((mid - X[p, si])[:, :2], axis=-1) < JB_ZONE
                                        bud = np.where(wg & nj, max(BUD_WING, BUD_JOINT), bud)
                                bad |= tight[:, None, None] & (np.abs(dz) + TP > bud[:, None, None] + 1e-6)
                        bb = orby(p, bad)
                        binary[key] |= bb if key == (a, b) else np.swapaxes(bb, 1, 2)
        # ---- pins and studs
        for a, b, X, ball in self.J:
            zX = X[..., 2]
            ba, bb_ = self.base(a, X), self.base(b, X)
            bud = np.where(self.wing(X), max(BUD_WING, BUD_JOINT), BUD_BODY)
            la_set = Lz if a not in FIXED else np.array([0.])
            lb_set = Lz if b not in FIXED else np.array([0.])
            za = ba[None, None] + la_set[:, None, None, None]; zb = bb_[None, None] + lb_set[None, :, None, None]
            lo, hi = np.minimum(za, zb), np.maximum(za, zb)
            if ball: lo, hi = np.minimum(lo, zX), np.maximum(hi, zX)
            if ball:
                bad = ((hi - lo + TP) > bud + 1e-6).any(-1)
                self._put(unary, binary, ternary, a, b, None, np.moveaxis(bad, -1, 0))
            Xf = X * FLAT
            for c in self.names:
                if c in (a, b): continue
                PC, QC, RC = stk[c]
                p, k, si = cand(Xf[..., :2], np.zeros(Xf.shape[:2]), c, R_PIN + RC.max() + GAP)
                dd, s_, t_ = seg2(Xf[p, si], Xf[p, si], PC[k, p, si] * FLAT, QC[k, p, si] * FLAT)
                near = dd < R_PIN + RC[k] + GAP
                p, k, si, t_ = p[near], k[near], si[near], t_[near]
                lc_set = Lz if c not in FIXED else np.array([0.])
                hc = hh(c) + TP / 2 + GAP
                if len(p):
                    pc_, qc_ = PC[k, p, si], QC[k, p, si]
                    zc = pc_[:, 2] + t_ * (qc_[:, 2] - pc_[:, 2])
                    zcl = zc[:, None] + lc_set[None]                                             # (N, nc)
                    lo_e, hi_e = np.moveaxis(lo[:, :, p, si], -1, 0), np.moveaxis(hi[:, :, p, si], -1, 0)   # (N, na, nb)
                    bad = (lo_e[..., None] - hc + 1e-6 < zcl[:, None, None]) & (zcl[:, None, None] < hi_e[..., None] + hc - 1e-6)
                    bad = orby(p, bad)
                    self._put(unary, binary, ternary, a, b, c, bad)
        if CLIP: self.clip_tables(unary, binary, stk)
        self.unary, self.binary, self.ternary = unary, binary, ternary

    def clip_tables(self, unary, binary, stk):
        """2026-11-17 fix: a sloped rod (skewer rod 1, rod 4) against a flat plate, exact. The closest point in plan of a short
        piece is not where the height clash is worst (the skewer descends while it is still inside the plan band; seen as -0.3 to
        -0.9 mm in crossing_check). For every level offset o of the plate, the rod is clipped to the part whose height is within
        the clash window |z - o| < h1 + h2 + GAP (height is linear along it), and that piece's plan distance to the plate decides."""
        P = self.P; Lz = LV * T
        for a in ('rod1', 'rod4'):
            if a not in self.raw: continue
            for b in ('hum', 'uln', 'bc1', 'rod2', 'bc2', 'rod3'):
                if b not in self.names: continue
                PB, QB, RB = stk[b]; PBf, QBf = PB * FLAT, QB * FLAT
                thr = hh(a) + hh(b) + GAP - 1e-6
                O = Lz if a in FIXED else np.arange(-(NL - 1), NL) * T             # plate level lb, or lb - la
                bad = np.zeros((len(O), P), bool)
                Js = [X * FLAT for a_, b_, X, ball in self.J if {a_, b_} == {a, b}]   # shared joints: as crossing_check, the
                for pa, qa, ra in self.raw[a]:                                         # rod within JR (plan) of it is by design
                    z0, z1 = pa[..., 2], qa[..., 2]; dz = z1 - z0                       # (P,S)
                    o = O[:, None, None]
                    flat_ = np.abs(dz) < 1e-9
                    with np.errstate(divide='ignore', invalid='ignore'):
                        s1, s2 = (o - thr - z0) / dz, (o + thr - z0) / dz
                    lo_ = np.where(flat_, 0.0, np.clip(np.minimum(s1, s2), 0, 1)); hi_ = np.where(flat_, 1.0, np.clip(np.maximum(s1, s2), 0, 1))
                    ok = np.where(flat_, np.abs(z0 - o) < thr, lo_ < hi_)                # (nO,P,S)
                    pieces = [(lo_, hi_)]
                    for J in Js:                                                        # cut [e1, e2] (within JR of J) out
                        dv = ((qa - pa) * FLAT)[..., :2]; w = ((pa - J) * FLAT)[..., :2]
                        A_, B_, C_ = np.sum(dv * dv, -1), 2 * np.sum(dv * w, -1), np.sum(w * w, -1) - JR_CLIP ** 2
                        disc = B_ * B_ - 4 * A_ * C_; has = (disc > 0) & (A_ > 1e-12)
                        sq = np.sqrt(np.maximum(disc, 0)); Aq = np.maximum(A_, 1e-12)
                        e1 = np.where(has, (-B_ - sq) / (2 * Aq), 2.0); e2 = np.where(has, (-B_ + sq) / (2 * Aq), 2.0)
                        pieces = [pc for l_, h_ in pieces for pc in ((l_, np.minimum(h_, e1)), (np.maximum(l_, e2), h_))]
                    for l_, h_ in pieces:
                        A0 = pa[None] + l_[..., None] * (qa - pa)[None]; A1 = pa[None] + h_[..., None] * (qa - pa)[None]
                        dd, _, _ = seg2((A0 * FLAT)[:, None], (A1 * FLAT)[:, None], PBf[None], QBf[None])    # (nO,Kb,P,S)
                        hit = ok & (l_ < h_) & (dd < (ra + RB)[None, :, None, None] + GAP).any(1)
                        bad |= hit.any(-1)
                if a in FIXED:
                    unary[b] |= bad.T
                else:
                    la, lb = np.meshgrid(np.arange(NL), np.arange(NL), indexing='ij')
                    B = np.moveaxis(bad[(lb - la) + NL - 1], -1, 0)                       # (P, la, lb)
                    key = (a, b) if self.var.index(a) < self.var.index(b) else (b, a)
                    if key not in binary: binary[key] = np.zeros((P, NL, NL), bool)
                    binary[key] |= B if key == (a, b) else np.swapaxes(B, 1, 2)

    def tables_ref(self):
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
                        dd, s, t = seg2(pa * FLAT, qa * FLAT, pb * FLAT, qb * FLAT)        # overlap in plan
                        za = pa[..., 2] + s * (qa[..., 2] - pa[..., 2]); zb = pb[..., 2] + t * (qb[..., 2] - pb[..., 2])
                        xa = pa + s[..., None] * (qa - pa); xb = pb + t[..., None] * (qb - pb)
                        bud = np.where(self.wing((xa + xb) / 2), BUD_WING, BUD_BODY)
                        near = dd < ra + rb + GAP; tight = dd < ra + rb
                        if not (near.any()): continue
                        if a in FIXED or b in FIXED:          # one fixed: unary on the other
                            v, zv, zf = (b, zb, za) if a in FIXED else (a, za, zb)
                            dz = (zv[..., None] + Lz) - zf[..., None]                       # (P,S,NL)
                            bad = (near[..., None] & (np.abs(dz) < hh(a) + hh(b) + GAP - 1e-6))
                            unary[v] |= bad.any(1)
                        else:
                            key = bin_(a, b); sw = key != (a, b)
                            for x, la in enumerate(Lz):
                                dz = (za + la)[..., None] - (zb[..., None] + Lz)           # (P,S,NL) over lb
                                bad = near[..., None] & (np.abs(dz) < hh(a) + hh(b) + GAP - 1e-6)
                                if thick: bad |= tight[..., None] & (np.abs(dz) + TP > bud[..., None] + 1e-6)
                                bb = bad.any(1)
                                if sw: binary[key][:, :, x] |= bb
                                else: binary[key][:, x, :] |= bb
        # ---- pins and studs
        for a, b, X, ball in self.J:
            zX = X[..., 2]
            ba, bb_ = self.base(a, X), self.base(b, X)
            bud = np.where(self.wing(X), max(BUD_WING, BUD_JOINT), BUD_BODY)
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
                    dd, s, t = seg2(X * FLAT, X * FLAT, pc * FLAT, qc * FLAT)
                    near = dd < R_PIN + rc + GAP
                    if not near.any(): continue
                    zc = pc[..., 2] + t * (qc[..., 2] - pc[..., 2])
                    lc_set = Lz if c not in FIXED else np.array([0.])
                    zcl = zc[None] + lc_set[:, None, None]                    # (nc,P,S)
                    hc = hh(c) + TP / 2 + GAP                                          # pin / stud ends flush with plate faces
                    bad = near[None, None, None] & (lo[:, :, None] - hc + 1e-6 < zcl[None, None]) & (zcl[None, None] < hi[:, :, None] + hc - 1e-6)
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
        bud = np.where(g.wing(X), max(BUD_WING, BUD_JOINT), BUD_BODY).min(1)                  # (P,)
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
