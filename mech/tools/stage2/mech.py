"""Linkage configurations: synthesis from a target motion and forward kinematics (batched). See kin.py for frames/conventions.

A design = pose (folded humerus frame yaw/pitch/roll, folded elbow/wrist angles E1/M1) + free link dimensions per chain.
synthesise(): the last rod of each chain (its anchor on the bellcrank and its length) is fitted exactly (circle/sphere least squares)
to the target motion, as in v4. forward(): the real mechanism, given all dimensions (nominal or perturbed)."""
import numpy as np
from kin import *

# ---------------------------------------------------------------- design-variable layout
POSE_VARS = [('rx', -3.3, 3.3), ('ry', -3.3, 3.3), ('rz', -3.3, 3.3), ('E1', 10, 25), ('M1', 10, 50), ('fz', 0, 3)]   # rotation vector (rad) Ro -> folded
ELBOW_VARS = {
    'body': [('g1', -40, 40), ('g2', -40, 40), ('h1x', -6, 26), ('h1y', -14, 8), ('h1z', -6, 6), ('rb1', 4, 16), ('Lc1', 4, 40),
             ('wx', -12, 10), ('wy', -12, 8), ('br1', 0, 1)],
    'hum': [('Ax', -25, 10), ('Ay', -40, 15), ('Az', -20, 12), ('gx', -6, 26), ('gy', -16, 8), ('rb1', 4, 16), ('Lc1', 4, 40),
            ('wx', -12, 10), ('wy', -12, 8), ('br1', 0, 1)],
    'ulna': [('Ax', -25, 10), ('Ay', -40, 15), ('Az', -20, 12), ('gx', -6, 32), ('gy', -14, 10), ('rb1', 4, 16), ('Lc1', 4, 40),
             ('kx', 0, 30), ('ky', -14, 10), ('br1', 0, 1)],
}
WRIST_VARS = {
    'ulna': [('G2x', -4, 30), ('G2y', -14, 6), ('h3x', 6, 30), ('h3y', -16, 4), ('rb3', 4, 16), ('Lc3', 4, 40), ('qx', -12, 20), ('qy', -14, 6), ('br3', 0, 1)],
    'hum': [('G2x', -4, 26), ('G2y', -16, 6), ('h3x', -6, 20), ('h3y', -14, 6), ('rb3', 4, 16), ('Lc3', 4, 40), ('qx', -12, 20), ('qy', -14, 6), ('br3', 0, 1)],
}
# manufactured dimensions (perturbed in the sensitivity analysis) and which of them are rod lengths (+ joint clearance)
DIMS_E = {'body': [('G1w', 3), ('h1', 3), ('rb1', 1), ('Lc1', 1), ('b2', 3), ('L2', 1), ('w', 2)],
          'hum': [('A', 3), ('g', 2), ('rb1', 1), ('Lc1', 1), ('b2', 2), ('L2', 1), ('w', 2)],
          'ulna': [('A', 3), ('g', 2), ('rb1', 1), ('Lc1', 1), ('b2', 2), ('L2', 1), ('k', 2)]}
DIMS_W = [('G2', 2), ('h3', 2), ('rb3', 1), ('Lc3', 1), ('b4', 2), ('L4', 1), ('q', 2)]
DIMS_WT = [('G2', 2), ('h3', 2), ('rb3', 1), ('Lc3', 1), ('b4', 3), ('L4', 1), ('q', 3)]           # tilted wrist: rod 4 is spatial
# design B (tilted wrist): wrist-axis tilt tw (deg), primary root twist phi (deg), shift of the folded primary angles dl (deg),
# out-of-plane offsets of the hand lever (qz) and of the rod-4 ball on bellcrank 2 (b4z). Appended after the planar variables.
TILT_VARS = [('tw', -85, 85), ('phi', -20, 20), ('dl', -60, 10), ('qz', -8, 8), ('b4z', -12, 12)]
DIMS_B = [('LH', 1), ('LU', 1)]
RODS = ['Lc1', 'L2', 'Lc3', 'L4']


def var_list(ec, wc, tilt=False):
    return POSE_VARS + ELBOW_VARS[ec] + WRIST_VARS[wc] + (TILT_VARS if tilt else [])


def dim_list(ec, tilt=False):
    return DIMS_B + DIMS_E[ec] + (DIMS_WT if tilt else DIMS_W)


def flatten(d, ec, tilt=False):
    return np.concatenate([np.reshape(d[k], (d[k].shape[0], -1)) for k, n in dim_list(ec, tilt)], 1)


def unflatten(F, ec, template, tilt=False):
    out = dict(template); i = 0
    for k, n in dim_list(ec, tilt):
        v = F[:, i:i + n]; out[k] = v[:, 0] if n == 1 else v; i += n
    return out


def two_circle(c1, r1, c2, r2):
    """both intersections (2,...,2) and margin"""
    p1, m = cc2(c1, r1, c2, r2, np.ones(r1.shape, bool)); p2, _ = cc2(c1, r1, c2, r2, np.zeros(r1.shape, bool))
    return np.stack([p1, p2]), m


def ang2(v):
    return np.arctan2(v[..., 1], v[..., 0])


def pick_pt(cands, angle_of, ref):
    a0, a1 = angle_of(cands[0]), angle_of(cands[1])
    k = np.abs(angdiff(a0, ref)) <= np.abs(angdiff(a1, ref))
    a = np.where(k, a0, a1); return ref + angdiff(a, ref), np.where(k[..., None], cands[0], cands[1])


# ---------------------------------------------------------------- synthesis
def fit_fixed_z(Qb, bz):
    """sphere through the points Qb (P,N,3) with the centre's z fixed to bz (P,): centre (P,3), radius (P), residual (P,N)"""
    X2 = Qb[..., :2]; rhs = np.sum(X2 ** 2, -1) + (Qb[..., 2] - bz[:, None]) ** 2
    A = np.concatenate([2 * X2, np.ones(X2.shape[:-1] + (1,))], -1)
    AtA = np.einsum('...ni,...nj->...ij', A, A) + 1e-9 * np.eye(3); Atb = np.einsum('...ni,...n->...i', A, rhs)
    s_ = np.linalg.solve(AtA, Atb[..., None])[..., 0]
    c = np.c_[s_[:, :2], bz]; r = np.sqrt(np.maximum(s_[:, 2] + np.sum(s_[:, :2] ** 2, -1), 1e-12))
    return c, r, nrm(Qb - c[:, None]) - r[:, None]


def roots3(c0, c1, c2, L):
    """|c0 + cos(phi) c1 + sin(phi) c2| = L with c1 _|_ c2, |c1| = |c2|: roots (2,...), margin (mm)"""
    Pp, Qq = 2 * np.sum(c0 * c1, -1), 2 * np.sum(c0 * c2, -1)
    S = np.sum(c0 * c0, -1) + np.sum(c1 * c1, -1)
    amp = np.maximum(np.hypot(Pp, Qq), 1e-12); r = (L ** 2 - S) / amp
    base, dd = np.arctan2(Qq, Pp), np.arccos(np.clip(r, -1, 1))
    margin = np.minimum(L - np.sqrt(np.maximum(S - amp, 0)), np.sqrt(S + amp) - L)
    return np.stack([base + dd, base - dd]), margin


def to_ulna3(X2, pe, wc, z=None):
    """wrist-chain 2D points (ulna frame for wc 'ulna', humerus frame for 'hum') -> ulna frame 3D (origin at the elbow)"""
    if wc == 'hum': X2 = rot2(-pe, X2 - np.array([LH, 0.0]))
    return np.concatenate([X2, (np.zeros(X2.shape[:-1]) if z is None else z)[..., None]], -1)


def rotz_apply(t, v):
    """rotate 3D vectors v (...,3) about z by t (...)"""
    return np.concatenate([rot2(t, v[..., :2]), v[..., 2:]], -1)


def synthesise(ec, wc, X, u, tilt=False):
    """X (P, nvar) -> dict with motion frames, target angles, dimensions, fit residuals"""
    names = [v[0] for v in var_list(ec, wc, tilt)]; x = {n: X[:, i] for i, n in enumerate(names)}
    P, N = X.shape[0], len(u)
    axis, th, Rf = shoulder(np.c_[x['rx'], x['ry'], x['rz']])
    Rh = humerus_frames(axis, th, u)
    Et = E0 + (x['E1'][:, None] - E0) * u; Mt = M0 + (x['M1'][:, None] - M0) * u
    pe_t, psi_t = np.pi - Et * D, -(np.pi - Mt * D)
    d = dict(LH=np.full(P, LH), LU=np.full(P, LU), axis=axis, th=th, Rf=Rf, E1=x['E1'], M1=x['M1'], fz=x['fz'], ec=ec, wc=wc, tilt=tilt)
    if tilt: d.update(tw=np.radians(x['tw']), phi=np.radians(x['phi']), dl=x['dl'])
    ref = dict(pe=pe_t, psi=psi_t)
    E = mv(Rh, np.array([LH, 0, 0])); Ru = Rh @ rz3(pe_t)
    ok = np.ones(P, bool); res_e = np.zeros(P)
    # ---- elbow chain
    if ec == 'body':
        Ab = hinge_frame(axis); h1 = np.c_[x['h1x'], x['h1y'], x['h1z']]
        H1 = mv(Rh, h1[:, None]); axial = np.sum(H1[:, 0] * axis, -1)
        H2 = np.stack([np.sum(H1 * Ab[:, None, :, 0], -1), np.sum(H1 * Ab[:, None, :, 1], -1)], -1)
        g = np.c_[x['g1'], x['g2']]
        B1, m1 = cc2(np.broadcast_to(g[:, None], H2.shape), x['rb1'][:, None], H2, x['Lc1'][:, None], (x['br1'] > 0.5)[:, None])
        beta = np.unwrap(ang2(B1 - g[:, None]), axis=1)
        G1w = mv(Ab, np.c_[g, axial])
        Fb = Ab[:, None] @ rz3(beta)
        U = E + mv(Ru, to3(np.c_[x['wx'], x['wy']])[:, None])
        Ub = mtv(Fb, U - G1w[:, None])
        c, L2, res = sphere_fit(Ub)
        d.update(G1w=G1w, h1=h1, rb1=x['rb1'], Lc1=x['Lc1'], b2=c, L2=L2, w=np.c_[x['wx'], x['wy']], Ab=Ab)
        ref['b1'] = beta; ok &= m1.min(1) > 0; res_e = np.abs(res).max(1)
    else:
        A = np.c_[x['Ax'], x['Ay'], x['Az']]; g = np.c_[x['gx'], x['gy']]
        if ec == 'hum':
            Fr, C = Rh, mv(Rh, to3(g)[:, None])
        else:
            Fr, C = Ru, E + mv(Ru, to3(g)[:, None])
        roots, m1 = loop_roots(Fr, C, np.broadcast_to(np.c_[x['rb1'], 0 * x['rb1']][:, None], (P, N, 2)), A[:, None], x['Lc1'][:, None])
        b = np.empty((P, N)); b[:, 0] = np.where(x['br1'] > 0.5, roots[0][:, 0], roots[1][:, 0])
        for i in range(1, N): b[:, i] = pick(roots[:, :, i], b[:, i - 1])
        ok &= m1.min(1) > 0
        if ec == 'hum':
            Uh = np.array([LH, 0]) + rot2(pe_t, np.c_[x['wx'], x['wy']][:, None])        # humerus 2D
            Ub = rot2(-b, Uh - g[:, None])
            c, L2, res = circle_fit2(Ub)
            d.update(A=A, g=g, rb1=x['rb1'], Lc1=x['Lc1'], b2=c, L2=L2, w=np.c_[x['wx'], x['wy']])
        else:
            k = np.c_[x['kx'], x['ky']]
            Ku = rot2(-pe_t, k[:, None] - np.array([LH, 0]))                              # ulna 2D
            Kb = rot2(-b, Ku - g[:, None])
            c, L2, res = circle_fit2(Kb)
            d.update(A=A, g=g, rb1=x['rb1'], Lc1=x['Lc1'], b2=c, L2=L2, k=k)
        ref['b1'] = b; res_e = np.abs(res).max(1)
    # ---- wrist chain (2D, humerus plane)
    G2, h3 = np.c_[x['G2x'], x['G2y']], np.c_[x['h3x'], x['h3y']]; q = np.c_[x['qx'], x['qy']]
    br3 = (x['br3'] > 0.5)[:, None]
    if wc == 'ulna':
        Cu = rot2(-pe_t, h3[:, None] - np.array([LH, 0]))
        B3, m3 = cc2(np.broadcast_to(G2[:, None], Cu.shape), x['rb3'][:, None], Cu, x['Lc3'][:, None], br3)
        Qf = np.array([LU, 0]) + rot2(psi_t, q[:, None])
    else:
        Ch = np.array([LH, 0]) + rot2(pe_t, h3[:, None])
        B3, m3 = cc2(np.broadcast_to(G2[:, None], Ch.shape), x['rb3'][:, None], Ch, x['Lc3'][:, None], br3)
        Wh = np.array([LH, 0]) + rot2(pe_t, np.array([LU, 0.0]))
        Qf = Wh + rot2(pe_t + psi_t, q[:, None])
    beta3 = np.unwrap(ang2(B3 - G2[:, None]), axis=1)
    if not tilt:
        c4, L4, res4 = circle_fit2(rot2(-beta3, Qf - G2[:, None]))
    else:
        # hand lever Q (3D) turns about the tilted wrist axis; rod 4 is spatial -> sphere fit in the bellcrank-2 frame, centre z = b4z
        q = np.c_[x['qx'], x['qy'], x['qz']]
        Qu = np.array([LU, 0, 0.0]) + mv(hand_rot(psi_t, d['tw']), q[:, None])          # ulna frame, origin at the elbow
        G2u = to_ulna3(np.broadcast_to(G2[:, None], B3.shape), pe_t, wc); bu = beta3 - (pe_t if wc == 'hum' else 0)
        c4, L4, res4 = fit_fixed_z(rotz_apply(-bu, Qu - G2u), x['b4z'])
    d.update(G2=G2, h3=h3, rb3=x['rb3'], Lc3=x['Lc3'], b4=c4, L4=L4, q=q, br3=x['br3'] > 0.5)
    ref['b3'] = beta3; ok &= m3.min(1) > 0
    return d, ref, dict(ok=ok, res_e=res_e, res_w=np.abs(res4).max(1), Rh=Rh)


# ---------------------------------------------------------------- forward kinematics
def forward(d, ref, u, Rh=None, detail=False):
    """d: dims (P,...), ref: reference angles (P,N) used to pick the assembly branch at each step."""
    ec, wc = d['ec'], d['wc']
    P, N = d['LH'].shape[0], len(u)
    if Rh is None: Rh = humerus_frames(d['axis'], d['th'], u)
    LHv, LUv = d['LH'][:, None], d['LU'][:, None]
    E = Rh[..., :, 0] * LHv[..., None]
    zh = Rh[..., :, 2]
    out = dict(margins=[], trans=[], swivel=[], pts=[])
    # ---- elbow
    if ec == 'body':
        a = d['axis']; Ab = hinge_frame(a); G = d['G1w'][:, None]
        H1 = mv(Rh, d['h1'][:, None])
        roots, m1 = loop_roots(np.broadcast_to(Ab[:, None], Rh.shape), np.broadcast_to(G, H1.shape),
                               np.broadcast_to(np.c_[d['rb1'], 0 * d['rb1']][:, None], (P, N, 2)), H1, d['Lc1'][:, None])
        b = pick(roots, ref['b1'])
        Rb = Ab[:, None] @ rz3(b); B1 = G + Rb[..., :, 0] * d['rb1'][:, None, None]; B2 = G + mv(Rb, d['b2'][:, None])
        w3 = to3(d['w'])[:, None]
        roots2, m2 = loop_roots(Rh, E, np.broadcast_to(d['w'][:, None], (P, N, 2)), B2, d['L2'][:, None])
        pe = pick(roots2, ref['pe'])
        Ru = Rh @ rz3(pe); U = E + mv(Ru, w3)
        out['trans'] += [trans_angle(B1 - H1, np.cross(a[:, None], B1 - G)), trans_angle(U - B2, np.cross(zh, U - E))]
        out['margins'] += [m1, m2]
        if detail:
            out['swivel'] = [cone_half_angle(unit(mtv(Rb, U - B2))), cone_half_angle(unit(mtv(Ru, B2 - U)))]
            hb = Rh[..., :, 0] * d['h1'][:, None, 0:1]
            out['pts'] += [G + 0 * B1, B1, B2, H1, U] + [H1 + (B1 - H1) * t for t in (.25, .5, .75)] + [B2 + (U - B2) * t for t in (.25, .5, .75)] \
                + [(G + B1) / 2, (B1 + B2) / 2, (G + B2) / 2, (H1 + hb) / 2, (U + E + mv(Ru, np.c_[d['w'][:, 0], 0 * d['w'][:, 0], 0 * d['w'][:, 0]][:, None])) / 2]
            out['elbow_pts'] = dict(G1=G + 0 * B1, B1=B1, B2=B2, H1=H1, U=U)
    elif ec == 'hum':
        A = d['A'][:, None]; Gh = d['g'][:, None]; G = mv(Rh, to3(d['g'])[:, None])
        roots, m1 = loop_roots(Rh, G, np.broadcast_to(np.c_[d['rb1'], 0 * d['rb1']][:, None], (P, N, 2)), np.broadcast_to(A, G.shape), d['Lc1'][:, None])
        b = pick(roots, ref['b1'])
        B1h = Gh + rot2(b, np.c_[d['rb1'], 0 * d['rb1']][:, None]); B2h = Gh + rot2(b, d['b2'][:, None])
        cands, m2 = two_circle(np.broadcast_to(np.array([1.0, 0]) * LHv[..., None], B2h.shape), np.broadcast_to(nrm(d['w'])[:, None], b.shape), B2h,
                               np.broadcast_to(d['L2'][:, None], b.shape))
        aw = ang2(d['w'])[:, None]
        pe, Uh = pick_pt(cands, lambda p: ang2(p - np.array([1.0, 0]) * LHv[..., None]) - aw, ref['pe'])
        Ru = Rh @ rz3(pe); B1 = mv(Rh, to3(B1h)); B2 = mv(Rh, to3(B2h)); U = mv(Rh, to3(Uh))
        out['trans'] += [trans_angle(B1 - A, np.cross(zh, B1 - G)), trans_angle(U - B2, np.cross(zh, U - E))]
        out['margins'] += [m1, m2]
        if detail:
            Rb = Rh @ rz3(b)
            out['swivel'] = [cone_half_angle(unit(B1 - A)), cone_half_angle(unit(mtv(Rb, A - B1)))]
            out['pts'] += [G, B1, B2, U, A + 0 * B1] + [A + (B1 - A) * t for t in (.25, .5, .75)] + [B2 + (U - B2) * t for t in (.25, .5, .75)] \
                + [(G + B1) / 2, (B1 + B2) / 2, (G + B2) / 2, (U + E) / 2]
            out['elbow_pts'] = dict(G1=G, B1=B1, B2=B2, A=A + 0 * B1, U=U)
    else:   # ulna: unknowns pe, b solved together (Newton from the reference)
        A = d['A'][:, None]; g = d['g'][:, None]; r1 = np.c_[d['rb1'], 0 * d['rb1']][:, None]; b2 = d['b2'][:, None]; k = d['k'][:, None]
        LHv2 = np.array([1.0, 0]) * LHv[..., None]

        def F(pe, b):
            Ru = Rh @ rz3(pe); B1 = E + mv(Ru, to3(g + rot2(b, r1)))
            B2h = LHv2 + rot2(pe, g + rot2(b, b2))
            return np.stack([nrm(B1 - A) - d['Lc1'][:, None], nrm(B2h - k) - d['L2'][:, None]], -1)
        pe, b = ref['pe'].copy(), ref['b1'].copy()
        for _ in range(12):
            f = F(pe, b); h = 1e-6
            J = np.stack([(F(pe + h, b) - f) / h, (F(pe, b + h) - f) / h], -1)
            det = J[..., 0, 0] * J[..., 1, 1] - J[..., 0, 1] * J[..., 1, 0]
            det = np.where(np.abs(det) < 1e-12, 1e-12, det)
            dpe = (J[..., 1, 1] * f[..., 0] - J[..., 0, 1] * f[..., 1]) / det
            db = (-J[..., 1, 0] * f[..., 0] + J[..., 0, 0] * f[..., 1]) / det
            pe, b = pe - np.clip(dpe, -0.3, 0.3), b - np.clip(db, -0.3, 0.3)
        resid = np.abs(F(pe, b)).max(-1)
        Ru = Rh @ rz3(pe); Gw = E + mv(Ru, to3(g)); B1 = Gw + mv(Ru, to3(rot2(b, r1)))
        B2h = LHv2 + rot2(pe, g + rot2(b, b2)); B2 = mv(Rh, to3(B2h)); K = mv(Rh, to3(np.broadcast_to(k, B2h.shape)))
        _, m1 = loop_roots(Ru, Gw, np.broadcast_to(r1, (P, N, 2)), np.broadcast_to(A, Gw.shape), d['Lc1'][:, None])
        Gh = LHv2 + rot2(pe, g)
        _, m2 = two_circle(Gh, np.broadcast_to(nrm(d['b2'])[:, None], pe.shape), np.broadcast_to(k, Gh.shape), np.broadcast_to(d['L2'][:, None], pe.shape))
        m1 = np.where(resid > 1e-3, -1.0, m1)
        out['trans'] += [trans_angle(B1 - A, np.cross(Ru[..., :, 2], B1 - Gw)), trans_angle(to3(k - B2h), to3(rot2(np.pi / 2 + 0 * pe, k - LHv2)))]
        out['margins'] += [m1, m2]
        if detail:
            Rb = Ru @ rz3(b)
            out['swivel'] = [cone_half_angle(unit(B1 - A)), cone_half_angle(unit(mtv(Rb, A - B1)))]
            out['pts'] += [Gw, B1, B2, K, A + 0 * B1] + [A + (B1 - A) * t for t in (.25, .5, .75)] + [B2 + (K - B2) * t for t in (.25, .5, .75)] \
                + [(Gw + B1) / 2, (B1 + B2) / 2, (Gw + B2) / 2, (K + E) / 2]
            out['elbow_pts'] = dict(G1=Gw, B1=B1, B2=B2, A=A + 0 * B1, K=K)
    # ---- wrist (2D in the humerus plane)
    G2, rb3 = d['G2'][:, None], d['rb3'][:, None]
    LHv2 = np.array([1.0, 0]) * LHv[..., None]
    if wc == 'ulna':
        Cu = rot2(-pe, d['h3'][:, None] - LHv2)
        B3, m3 = cc2(np.broadcast_to(G2, Cu.shape), np.broadcast_to(rb3, pe.shape), Cu, np.broadcast_to(d['Lc3'][:, None], pe.shape), d['br3'][:, None])
        Wc = np.array([1.0, 0]) * LUv[..., None]
    else:
        Cu = LHv2 + rot2(pe, d['h3'][:, None])
        B3, m3 = cc2(np.broadcast_to(G2, Cu.shape), np.broadcast_to(rb3, pe.shape), Cu, np.broadcast_to(d['Lc3'][:, None], pe.shape), d['br3'][:, None])
        Wc = LHv2 + rot2(pe, np.array([1.0, 0]) * LUv[..., None])
    beta3 = ang2(B3 - G2)
    if d.get('tilt'):
        return _wrist_tilt(d, ref, out, Rh, E, pe, B3, Cu, G2, beta3, m3, detail, locals().get('b'))
    B4 = G2 + rot2(beta3, d['b4'][:, None])
    cands, m4 = two_circle(np.broadcast_to(Wc, B4.shape), np.broadcast_to(nrm(d['q'])[:, None], pe.shape), B4, np.broadcast_to(d['L4'][:, None], pe.shape))
    aq = ang2(d['q'])[:, None]
    base = (0 if wc == 'ulna' else pe)
    psi, Q = pick_pt(cands, lambda p: ang2(p - Wc) - aq - base, ref['psi'])
    out['trans'] += [trans_angle(to3(B3 - Cu), to3(rot2(np.pi / 2 + 0 * pe, B3 - G2))), trans_angle(to3(Q - B4), to3(rot2(np.pi / 2 + 0 * pe, Q - Wc)))]
    out['margins'] += [m3, m4]
    out.update(pe=pe, psi=psi, E=180 - np.degrees(pe), M=180 + np.degrees(psi), b1=locals().get('b'), b3=beta3)
    if detail:
        tohum = (lambda v: LHv2 + rot2(pe, v)) if wc == 'ulna' else (lambda v: v)
        W2 = tohum(Wc) if wc == 'ulna' else Wc
        wp = dict(G2=tohum(np.broadcast_to(G2, B3.shape)), B3=tohum(B3), B4=tohum(B4), C3=tohum(Cu), Q=tohum(Q))
        w3 = {k_: mv(Rh, to3(v)) for k_, v in wp.items()}
        out['wrist_pts'] = w3
        X = [w3['G2'], w3['B3'], w3['B4'], w3['C3'], w3['Q']] + [w3['C3'] + (w3['B3'] - w3['C3']) * t for t in (.25, .5, .75)] \
            + [w3['B4'] + (w3['Q'] - w3['B4']) * t for t in (.25, .5, .75)] + [(w3['G2'] + w3['B3']) / 2, (w3['B3'] + w3['B4']) / 2, (w3['G2'] + w3['B4']) / 2,
                                                                                    (w3['Q'] + mv(Rh, to3(W2))) / 2]
        out['pts'] += X
        out['Rh'] = Rh; out['E_w'] = E
        out['W_w'] = mv(Rh, to3(W2))
        out['T_w'] = out['W_w'] + mv(Rh, to3(rot2(pe + psi, np.array([LMAN, 0.0]))))
    return out


def _wrist_tilt(d, ref, out, Rh, E, pe, B3, Cu, G2, beta3, m3, detail, b1):
    """wrist chain of design B, worked in the ulna frame (3D): bellcrank 2 planar (pins on rod 3), rod 4 spatial (balls)."""
    wc = d['wc']; P, N = pe.shape
    G2u = to_ulna3(np.broadcast_to(G2, B3.shape), pe, wc); bu = beta3 - (pe if wc == 'hum' else 0)
    B3u = to_ulna3(B3, pe, wc); Cuu = to_ulna3(Cu, pe, wc)
    B4u = G2u + rotz_apply(bu, np.broadcast_to(d['b4'][:, None], B3u.shape))
    a = wrist_axis(d['tw'])[:, None]; q0 = mv(rz3(np.full(P, PSI0)), d['q'])[:, None]            # lever at the open pose (ulna frame)
    Wu = np.zeros_like(B4u); Wu[..., 0] = d['LU'][:, None]
    qa = np.sum(q0 * a, -1, keepdims=True) * a; c1 = q0 - qa; c2 = np.cross(a, q0)
    roots, m4 = roots3(Wu + qa - B4u, np.broadcast_to(c1, B4u.shape), np.broadcast_to(c2, B4u.shape), d['L4'][:, None])
    dpsi = pick(roots, ref['psi'] - PSI0); psi = PSI0 + dpsi
    Rw = hand_rot(psi, d['tw']); Q = Wu + mv(Rw, d['q'][:, None])
    out['trans'] += [trans_angle(to3(B3 - Cu), to3(rot2(np.pi / 2 + 0 * pe, B3 - G2))), trans_angle(Q - B4u, np.cross(np.broadcast_to(a, Q.shape), Q - Wu))]
    out['margins'] += [m3, m4]
    out.update(pe=pe, psi=psi, E=180 - np.degrees(pe), M=180 + np.degrees(psi), b1=b1, b3=beta3)
    if detail:
        Rb = rz3(bu)                                                                              # bellcrank-2 frame in ulna coords
        out['swivel'] = list(out['swivel']) + [rod_end_tilt(unit(mtv(Rb, Q - B4u))), rod_end_tilt(unit(mtv(Rw, B4u - Q)))]   # rod 4: rod-end bearings
        Ru = Rh @ rz3(pe)
        tow = lambda v: E + mv(Ru, v)
        w3 = dict(G2=tow(G2u), B3=tow(B3u), B4=tow(B4u), C3=tow(Cuu), Q=tow(Q))
        out['wrist_pts'] = w3
        out['pts'] += [w3['G2'], w3['B3'], w3['B4'], w3['C3'], w3['Q']] + [w3['C3'] + (w3['B3'] - w3['C3']) * t for t in (.25, .5, .75)] \
            + [w3['B4'] + (w3['Q'] - w3['B4']) * t for t in (.25, .5, .75)] + [(w3['G2'] + w3['B3']) / 2, (w3['B3'] + w3['B4']) / 2,
                                                                                    (w3['G2'] + w3['B4']) / 2, (w3['Q'] + tow(Wu)) / 2]
        out['Rh'] = Rh; out['E_w'] = E; out['Ru'] = Ru; out['Rm'] = Ru @ Rw
        out['W_w'] = tow(Wu); out['T_w'] = out['W_w'] + mv(out['Rm'], np.array([LMAN, 0, 0.0]))
    return out


def wing_region_dist(Xh, pe, psi, fz=0.0, hand=True):
    """hidden-ness rule of the simulator (outsideDist3 wing part), vectorised. Xh: humerus-frame points (...,3).
    The wing occupies |z| <= 4 around the bone plane, extended on the dorsal side by the feather standoff fz."""
    E, W, T = wing_points_2d(pe, psi)
    S = np.zeros_like(E)
    z = np.abs(Xh[..., 2] - fz / 2) - fz / 2          # |z| measured from the slab [0, fz]; wing = this <= 4
    x2 = Xh[..., :2]
    tri = [S, E, W]
    sg = np.sign((E[..., 0] - S[..., 0]) * (W[..., 1] - S[..., 1]) - (E[..., 1] - S[..., 1]) * (W[..., 0] - S[..., 0])); sg = np.where(sg == 0, 1, sg)
    dmax = np.zeros(x2.shape[:-1])
    for i in range(3):
        a, b = tri[i], tri[(i + 1) % 3]; ex, ey = b[..., 0] - a[..., 0], b[..., 1] - a[..., 1]; Ln = np.maximum(np.hypot(ex, ey), 1e-9)
        dmax = np.maximum(dmax, -((x2[..., 0] - a[..., 0]) * (-ey) + (x2[..., 1] - a[..., 1]) * ex) * sg / Ln)
    best = np.maximum(0, dmax)
    for o, dd, Lb, ymin in ((S, E, LH, -12), (E, W, LU, -38), (W, T, LMAN, -45))[:3 if hand else 2]:
        ang = np.arctan2(dd[..., 1] - o[..., 1], dd[..., 0] - o[..., 0]); loc = rot2(-ang, x2 - o)
        best = np.minimum(best, np.maximum.reduce([np.zeros_like(z), loc[..., 1] - 1, ymin - loc[..., 1], -2 - loc[..., 0], loc[..., 0] - Lb - 2]))
    return np.where(z > 4, z - 4, best)


def wing_region_tilt(X, Rh, pe, psi, Rp, Ww, fz):
    """design B: arm part (shoulder-elbow-wrist) in the humerus plane as before; the hand part in the primaries' own plane"""
    Xh = np.einsum('pnji,pnkj->pnki', Rh, X)
    arm = wing_region_dist(Xh, pe, psi, fz, hand=False)
    Xp = np.einsum('pnji,pnkj->pnki', Rp, X - Ww[:, :, None])
    zz = np.abs(Xp[..., 2] - fz / 2) - fz / 2
    box = np.maximum.reduce([np.zeros_like(zz), Xp[..., 1] - 1, -45 - Xp[..., 1], -2 - Xp[..., 0], Xp[..., 0] - LMAN - 2])
    return np.minimum(arm, np.where(zz > 4, zz - 4, box))
