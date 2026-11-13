"""Batched kinematics for the wing-fold linkage (stage 2). All lengths mm, kingfisher 1:1, v4 shoulder frame
(origin right shoulder, x lateral, y cranial, z dorsal). Arrays: P individuals x N motion steps.

Motion: humerus frame Rh(u) = Rot(axis, theta*u) @ Ro, single oblique shoulder hinge (axis, theta from the folded frame).
Elbow: ulna frame Ru = Rh Rz(pe), pe = pi - E.  Wrist: hand frame Rm = Ru Rz(psi), psi = -(pi - M).  (same as the simulator)

Elbow-chain configurations (where bellcrank 1 sits):
  'body' : BC1 pivots on the body, axis || shoulder hinge, in the plane of the humerus crank H1 -> rod1 planar (pins);
           rod2 BC1 -> ulna lever U is spatial (ball ends).  Stephenson-type six-bar.          (v4)
  'hum'  : BC1 pivots on the humerus (axis = wing-plane normal); rod1 from a body anchor A is spatial (balls);
           rod2 BC1 -> ulna lever is planar (pins).  Watt-II-type six-bar.                     (r4)
  'ulna' : BC1 pivots on the ulna; rod1 from a body anchor A is spatial (balls); rod2 BC1 -> humerus point K is planar.
Wrist-chain configurations (all planar, in the wing plane):
  'ulna' : BC2 on the ulna, driven by a humerus crank C3; rod4 -> hand lever Q.               (v4)
  'hum'  : BC2 on the humerus, driven by an ulna crank C3; rod4 -> hand lever Q.              (r3)
"""
import numpy as np
from scipy.spatial.transform import Rotation as Rot

D = np.pi / 180
LH, LU, LCMC, LDIG = 25.7, 32.0, 13.3, 7.0
LMAN = LCMC + LDIG
E0, M0 = 150.0, 160.0
Ro = Rot.from_euler('z', -15, degrees=True).as_matrix()


# ---------------------------------------------------------------- small helpers
def nrm(v):
    return np.linalg.norm(v, axis=-1)


def unit(v):
    return v / np.maximum(nrm(v), 1e-12)[..., None]


def rz3(t):
    c, s = np.cos(t), np.sin(t); z, o = np.zeros_like(t), np.ones_like(t)
    return np.stack([np.stack([c, -s, z], -1), np.stack([s, c, z], -1), np.stack([z, z, o], -1)], -2)


def rot2(t, v):
    """rotate 2D vectors v (...,2) by angle t (...)"""
    c, s = np.cos(t), np.sin(t)
    return np.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1]], -1)


def rodrigues(a, th):
    """a (...,3) unit axis, th (...) -> (...,3,3)"""
    a = np.broadcast_to(a, th.shape + (3,)); c, s = np.cos(th)[..., None, None], np.sin(th)[..., None, None]
    K = np.zeros(th.shape + (3, 3)); K[..., 0, 1], K[..., 0, 2], K[..., 1, 0] = -a[..., 2], a[..., 1], a[..., 2]
    K[..., 1, 2], K[..., 2, 0], K[..., 2, 1] = -a[..., 0], -a[..., 1], a[..., 0]
    return np.eye(3) * c + s * K + (1 - c) * a[..., :, None] * a[..., None, :]


def mv(M, v):
    return np.einsum('...ij,...j->...i', M, v)


def mtv(M, v):
    return np.einsum('...ji,...j->...i', M, v)


def angdiff(a, b):
    return np.arctan2(np.sin(a - b), np.cos(a - b))


def cc2(c1, r1, c2, r2, br):
    """circle-circle intersection in 2D (batched). br (...) bool selects the side. returns point, margin (mm, >0 = closes)"""
    d = c2 - c1; L = np.maximum(nrm(d), 1e-12)
    margin = np.minimum(r1 + r2 - L, L - np.abs(r1 - r2))
    a = (r1 ** 2 - r2 ** 2 + L ** 2) / (2 * L)
    h = np.sqrt(np.maximum(r1 ** 2 - a ** 2, 0)); ex = d / L[..., None]; perp = np.stack([-ex[..., 1], ex[..., 0]], -1)
    sg = np.where(br, 1.0, -1.0)
    return c1 + a[..., None] * ex + (h * sg)[..., None] * perp, margin


def loop_roots(Rf, C, w, F, L):
    """X(phi) = C + Rf Rz(phi) w (w in the frame's xy-plane), |X - F| = L. Returns roots (2,...), margin (mm)."""
    w3 = np.stack([w[..., 0], w[..., 1], np.zeros_like(w[..., 0])], -1)
    c0 = C - F; c1 = mv(Rf, w3); c2 = mv(Rf, np.stack([-w[..., 1], w[..., 0], np.zeros_like(w[..., 0])], -1))
    Pp, Qq = 2 * np.sum(c0 * c1, -1), 2 * np.sum(c0 * c2, -1)
    S = np.sum(c0 * c0, -1) + np.sum(c1 * c1, -1)
    amp = np.maximum(np.hypot(Pp, Qq), 1e-12); r = (L ** 2 - S) / amp
    base, dd = np.arctan2(Qq, Pp), np.arccos(np.clip(r, -1, 1))
    margin = np.minimum(L - np.sqrt(np.maximum(S - amp, 0)), np.sqrt(S + amp) - L)
    return np.stack([base + dd, base - dd]), margin


def pick(roots, ref):
    k = np.abs(angdiff(roots[0], ref)) <= np.abs(angdiff(roots[1], ref))
    r = np.where(k, roots[0], roots[1]); return ref + angdiff(r, ref)


def circle_fit2(X):
    """X (...,N,2) -> centre (...,2), radius (...), residual (...,N)"""
    A = np.concatenate([2 * X, np.ones(X.shape[:-1] + (1,))], -1); b = np.sum(X ** 2, -1)
    AtA = np.einsum('...ni,...nj->...ij', A, A) + 1e-9 * np.eye(3); Atb = np.einsum('...ni,...n->...i', A, b)
    s = np.linalg.solve(AtA, Atb[..., None])[..., 0]
    c = s[..., :2]; r = np.sqrt(np.maximum(s[..., 2] + np.sum(c ** 2, -1), 1e-12))
    return c, r, nrm(X - c[..., None, :]) - r[..., None]


def sphere_fit(X):
    A = np.concatenate([2 * X, np.ones(X.shape[:-1] + (1,))], -1); b = np.sum(X ** 2, -1)
    AtA = np.einsum('...ni,...nj->...ij', A, A) + 1e-9 * np.eye(4); Atb = np.einsum('...ni,...n->...i', A, b)
    s = np.linalg.solve(AtA, Atb[..., None])[..., 0]
    c = s[..., :3]; r = np.sqrt(np.maximum(s[..., 3] + np.sum(c ** 2, -1), 1e-12))
    return c, r, nrm(X - c[..., None, :]) - r[..., None]


def trans_angle(rod, vel):
    """transmission quality: 90 deg = rod along the driven point's velocity, 0 = dead point"""
    return np.degrees(np.arcsin(np.clip(np.abs(np.sum(unit(rod) * unit(vel), -1)), 0, 1)))


def cone_half_angle(d):
    """d (...,N,3) unit directions -> half-angle (deg) of the cone around their mean (ball-joint swivel needed)"""
    m = unit(d.mean(-2)); return np.degrees(np.arccos(np.clip(np.sum(d * m[..., None, :], -1), -1, 1))).max(-1)


# ---------------------------------------------------------------- shoulder / motion
def shoulder(rv):
    """rv (P,3) rad: rotation vector carrying the open humerus frame Ro to the folded one (= shoulder hinge axis * motor rotation)"""
    th = nrm(rv); axis = rv / np.maximum(th, 1e-12)[:, None]
    Rf = Rot.from_rotvec(rv).as_matrix() @ Ro
    return axis, th, Rf


def ypr_to_rv(ypr):
    Rf = Rot.from_euler('zyx', np.c_[ypr[:, 0], -ypr[:, 1], ypr[:, 2]], degrees=True).as_matrix()
    return Rot.from_matrix(Rf @ Ro.T).as_rotvec()


def shoulder_ypr(ypr):
    """ypr (P,3) deg: folded humerus frame (yaw about z from +x, pitch down negative, roll about the bone). -> axis (P,3), theta (P), Rf (P,3,3)"""
    Rf = Rot.from_euler('zyx', np.c_[ypr[:, 0], -ypr[:, 1], ypr[:, 2]], degrees=True).as_matrix()
    rv = Rot.from_matrix(Rf @ Ro.T).as_rotvec(); th = nrm(rv)
    return rv / np.maximum(th, 1e-12)[:, None], th, Rf


def humerus_frames(axis, th, u):
    return rodrigues(axis[:, None, :], th[:, None] * u[None, :]) @ Ro            # (P,N,3,3)


def hinge_frame(a):
    e1 = unit(np.cross(a, [0, 0, 1.0])); e2 = np.cross(a, e1)
    return np.stack([e1, e2, a], -1)                                               # columns


def wing_points_2d(pe, psi):
    """humerus-frame 2D positions of E, W, T"""
    E = np.zeros(pe.shape + (2,)); E[..., 0] = LH
    W = E + LU * np.stack([np.cos(pe), np.sin(pe)], -1)
    T = W + LMAN * np.stack([np.cos(pe + psi), np.sin(pe + psi)], -1)
    return E, W, T


def to3(v2):
    return np.concatenate([v2, np.zeros(v2.shape[:-1] + (1,))], -1)


# ---------------------------------------------------------------- feathers (same layout as the simulator)
PRIM = [[0.05, 40, -82, -8], [0.15, 42, -75, -7], [0.25, 44, -68, -6], [0.35, 46, -61, -5], [0.45, 48, -54, -4],
        [0.55, 50, -46, -3.5], [0.68, 53, -38, -3], [0.80, 56, -29, -2.5], [0.92, 57, -20, -2], [1.00, 53, -11, -1.5]]
SEC = [[0.96 - i * 0.095, 40 - i * 0.9, -100 - i * 3.0, -170 + i * 0.8] for i in range(10)]


def feather_local(spread, per=5):
    """feather sample points: secondaries in the ulna frame (rel. elbow), primaries in the hand frame (rel. wrist). spread 0 = folded, 1 = open"""
    sec, pri = [], []
    for i, (f, L, a0, a1) in enumerate(SEC):
        a = np.radians(a1 + (a0 - a1) * spread); d = np.array([np.cos(a), np.sin(a), 0]); s = np.array([-np.sin(a), np.cos(a), 0])
        base = np.array([LU * f, 0, -0.35 * i * 0.2])
        sec += [base + d * L * t for t in np.linspace(0.15, 1, per)] + [base + d * L * 0.82 + s * 3.7, base + d * L * 0.82 - s * 3.7]
    for i, (f, L, a0, a1) in enumerate(PRIM):
        a = np.radians(a1 + (a0 - a1) * spread); d = np.array([np.cos(a), np.sin(a), 0]); s = np.array([-np.sin(a), np.cos(a), 0])
        base = np.array([LMAN * f, 0, -0.35 * (2 + i * 0.25)])
        pri += [base + d * L * t for t in np.linspace(0.15, 1, per)] + [base + d * L * 0.82 + s * 3.5, base + d * L * 0.82 - s * 3.5]
    return np.array(sec), np.array(pri)


SEC_F, PRI_F = feather_local(0.0)
TIP_IDX = None


def folded_wing(Rf, E1, M1, fz=None):
    """folded pose (P,...) -> dict of world points. fz (P,): feather standoff on the dorsal side of the bone plane (mm)"""
    pe, psi = np.pi - E1 * D, -(np.pi - M1 * D)
    Ru = Rf @ rz3(pe); Rm = Ru @ rz3(psi)
    E = mv(Rf, np.array([LH, 0, 0])); W = E + mv(Ru, np.array([LU, 0, 0])); T = W + mv(Rm, np.array([LMAN, 0, 0]))
    off = np.zeros((len(E1), 1, 3)) if fz is None else np.stack([0 * fz, 0 * fz, fz], -1)[:, None]
    sec = E[:, None] + np.einsum('pij,pnj->pni', Ru, SEC_F[None] + off); pri = W[:, None] + np.einsum('pij,pnj->pni', Rm, PRI_F[None] + off)
    t = np.linspace(0, 1, 6)[None, :, None]
    bones = np.concatenate([E[:, None] * t[:, 2:], E[:, None] + (W - E)[:, None] * t, W[:, None] + (T - W)[:, None] * t], 1)
    return dict(E=E, W=W, T=T, Ru=Ru, Rm=Rm, sec=sec, pri=pri, bones=bones)


def wing_along_path(axis, th, Ro_, E1, M1, us):
    """bones + feathers at intermediate fractions (targets linear in u) for the sweep-through-body check"""
    out = []
    for u in us:
        Rh = rodrigues(axis, th * u) @ Ro_
        E, M = E0 + (E1 - E0) * u, M0 + (M1 - M0) * u
        pe, psi = np.pi - E * D, -(np.pi - M * D)
        Ru = Rh @ rz3(pe); Rm = Ru @ rz3(psi)
        Ep = mv(Rh, np.array([LH, 0, 0])); W = Ep + mv(Ru, np.array([LU, 0, 0])); T = W + mv(Rm, np.array([LMAN, 0, 0]))
        s, p = feather_local(1 - u, per=3)
        t = np.linspace(0, 1, 5)[None, :, None]
        pts = np.concatenate([Ep[:, None] * t[:, 3:], Ep[:, None] + (W - Ep)[:, None] * t, W[:, None] + (T - W)[:, None] * t,
                              Ep[:, None] + np.einsum('pij,nj->pni', Ru, s), W[:, None] + np.einsum('pij,nj->pni', Rm, p)], 1)
        out.append(pts)
    return np.concatenate(out, 1)
