"""Objectives and constraints for stage 2 (pymoo). All 1:1 kingfisher mm.

Objectives (minimise):
  f1 joint error   : max |E - E_target|, |M - M_target| over the motion (deg); targets linear in motor rotation
  f2 -transmission : -(minimum transmission angle over all four driven joints and the motion) (deg)
  f3 protrusion    : max distance of linkage sample points outside body (CG SDF) and wing region (simulator rule) (mm)
  f4 sensitivity   : 2 sigma of the joint-angle error from +-0.05 mm on every manufactured dimension (uniform) plus
                     joint clearance (radial 0.01 mm per pin -> rod length +-0.02 mm) plus motor quantisation (0.5 deg steps),
                     linearised (Jacobian), max over motion (deg)
  f5 fold fit      : how well the folded wing lies on the body (feathers buried / floating, bones, wing tip, covering the
                     CG's folded wing, wrist position, and the folded elbow / wrist / hand direction measured on the
                     Alcedo cristata CT (ct_frame.py; dead band 3 mm / 10 deg for the CT's uncertainty)) (mm-like score)
  f6 part size     : largest bellcrank / lever dimension (mm)
Constraints (<= 0):
  all loops close along the whole motion; dead-point margin (rod-length change that breaks a loop) >= 0.3 mm;
  every lever / bellcrank arm / crank radius >= LMIN (8 mm, 1:1; was 4); rods 4..40 mm; ball-joint swivel half-angle <= SWIVEL (25 deg);
  minimum transmission angle >= TMIN_MIN (35 deg);
  joint-angle gain |dE/dmotor|, |dM/dmotor| <= 2.5 deg/deg everywhere (controllability);
  feathers never sweep > 4 mm into the body during the motion; fold-fit <= 16 (prunes hopeless poses); fitted last-rod residual <= 0.3 mm.
"""
import json, pathlib, numpy as np
from pymoo.core.problem import Problem
from mech import *
from body import Body

LMIN, SWIVEL, MARGIN_MIN, SWEEP_MIN, FIT_MAX = 8.0, 25.0, 0.3, -4.0, 20.0
TMIN_MIN = 35.0
CT = json.load(open(pathlib.Path(__file__).resolve().parents[2] / 'results' / 'stage2' / 'ct_target.json'))['target']
CT_E, CT_W, CT_H = (np.array(CT[k]) for k in ('elbow', 'wrist', 'hand_dir'))
CT_HAND_W = 0.0                       # hand-direction term off: a planar wing cannot follow the CT hand (43 deg out of the arm plane)
GAIN_MAX = 2.5                        # max joint-angle change per degree of motor rotation (deg/deg), anywhere in the motion
SIG_MOTOR = 0.5 / np.sqrt(12)         # motor commanded in 0.5 deg steps -> uniform quantisation error, sigma (deg)
SIG_DIM = 0.05 / np.sqrt(3); CLEAR = 0.01; SIG_CL = 2 * CLEAR / np.sqrt(3)
U = np.linspace(0, 1, 41); SUB = np.arange(0, 41, 4)


class Evaluator:
    def __init__(self, ec, wc, width=45, nvis=100):
        self.ec, self.wc = ec, wc
        self.body = Body(width)
        rng = np.random.default_rng(0)
        v = self.body.vis; self.vis = v[rng.choice(len(v), min(nvis, len(v)), replace=False)]
        self.vars = var_list(ec, wc)
        self.dims = dim_list(ec)
        sig = []
        for k, n in self.dims:
            s = SIG_DIM ** 2 + (SIG_CL ** 2 if k in RODS else 0); sig += [np.sqrt(s)] * n
        self.sig = np.array(sig)

    # ------------------------------------------------------------ pieces
    def fold_fit(self, d):
        b = self.body
        fw = folded_wing(d['Rf'], d['E1'], d['M1'], d.get('fz'))
        feath = np.concatenate([fw['sec'], fw['pri']], 1)
        s = b.sd(feath)
        bury, flt = np.maximum(0, -1 - s), np.maximum(0, s - 4)
        sb = b.sd(fw['bones'][:, 3:])               # skip the proximal humerus
        bone = np.maximum(0, sb - 1) + np.maximum(0, -8 - sb)
        tips = fw['pri'][:, 4::7]                    # feather tips (per=5 + 2 side points)
        tip = tips[np.arange(len(tips)), np.argmin(tips[..., 1], 1)]
        tipd = nrm(tip - b.tip)
        dd = np.sqrt(((self.vis[None, :, None, :] - feath[:, None, :, :]) ** 2).sum(-1)).min(2)
        cover = np.maximum(0, dd - 3).mean(1)
        W = fw['W']; sw = b.sd(W)
        wr = np.maximum(0, -4 - W[:, 1]) + np.maximum(0, W[:, 1] - 10) + np.maximum(0, sw - 1) + np.maximum(0, -8 - sw)
        # the folded wing must show its dorsal (upper) surface: wing-plane normal (+z of the humerus frame) pointing away from the body
        cen = feath.mean(1); ax_pt = np.array([-9.0, 0, -5.0]); outward = unit(cen - np.c_[ax_pt[0] + 0 * cen[:, 0], cen[:, 1], ax_pt[2] + 0 * cen[:, 0]])
        dors = np.sum(d['Rf'][:, :, 2] * outward, -1)
        flip = 10 * np.maximum(0, 0.3 - dors)
        # CT: folded elbow / wrist positions and hand direction (Alcedo cristata, scaled to the ulna)
        ce = np.sqrt((nrm(fw['E'] - CT_E) ** 2 + nrm(fw['W'] - CT_W) ** 2) / 2)
        ch = np.degrees(np.arccos(np.clip(np.sum(unit(fw['T'] - fw['W']) * CT_H, -1), -1, 1)))
        ct = np.maximum(0, ce - 3) + CT_HAND_W * np.maximum(0, ch - 10) / 5
        parts = np.c_[np.sqrt((bury ** 2).mean(1)), np.sqrt((flt ** 2).mean(1)), np.sqrt((bone ** 2).mean(1)), np.maximum(0, tipd - 4), cover, wr, flip, ct]
        F = parts.sum(1)
        return F, dict(parts=parts, ct_pos=ce, ct_hand=ch, dorsal=dors, tip=tip, tipd=tipd, bury_max=bury.max(1), float_max=flt.max(1), cover=cover, W=W, feath_sd_min=s.min(1))

    def sweep(self, d):
        pts = wing_along_path(d['axis'], d['th'], Ro, d['E1'][:, None] * 0 + d['E1'][:, None] if False else d['E1'], d['M1'], (0.2, 0.4, 0.6, 0.8)) \
            if False else None
        out = np.full(len(d['E1']), np.inf)
        for u in (0.15, 0.3, 0.45, 0.6, 0.75, 0.9):
            Rh = rodrigues(d['axis'], d['th'] * u) @ Ro
            E = E0 + (d['E1'] - E0) * u; M = M0 + (d['M1'] - M0) * u
            pe, psi = np.pi - E * D, -(np.pi - M * D)
            Ru = Rh @ rz3(pe); Rm = Ru @ rz3(psi)
            Ep = mv(Rh, np.array([LH, 0, 0])); W = Ep + mv(Ru, np.array([LU, 0, 0]))
            s_, p_ = feather_local(1 - u, per=3)
            off = np.stack([0 * d['fz'], 0 * d['fz'], d['fz']], -1)[:, None] if 'fz' in d else 0
            pts = np.concatenate([Ep[:, None] + np.einsum('pij,pnj->pni', Ru, s_[None] + off), W[:, None] + np.einsum('pij,pnj->pni', Rm, p_[None] + off)], 1)
            out = np.minimum(out, self.body.sd(pts).min(1))
        return out

    def levers(self, d, f):
        ec, wc = self.ec, self.wc
        L = [d['rb1'], nrm(d['b2']), d['rb3'], nrm(d['b4']), nrm(d['q'])]
        a = d['axis']
        if ec == 'body':
            H1w = mv(d['Rf'], d['h1']); L += [nrm(d['w']), nrm(H1w - np.sum(H1w * a, -1)[:, None] * a)]
            part = [np.maximum.reduce([d['rb1'], nrm(d['b2']), nrm(d['b2'] - np.c_[d['rb1'], 0 * d['rb1'], 0 * d['rb1']])]),
                    np.hypot(d['h1'][:, 1], d['h1'][:, 2]), nrm(d['w'])]
        elif ec == 'hum':
            L += [nrm(d['w']), nrm(d['A'] - np.sum(d['A'] * a, -1)[:, None] * a)]
            part = [np.maximum.reduce([d['rb1'], nrm(d['b2']), nrm(d['b2'] - np.c_[d['rb1'], 0 * d['rb1']])]), nrm(d['w']), np.abs(d['g'][:, 1])]
        else:
            L += [nrm(d['k'] - np.array([LH, 0]))]
            part = [np.maximum.reduce([d['rb1'], nrm(d['b2']), nrm(d['b2'] - np.c_[d['rb1'], 0 * d['rb1']])]), np.abs(d['k'][:, 1]), np.abs(d['g'][:, 1])]
        if wc == 'ulna':
            L += [nrm(d['h3'] - np.array([LH, 0]))]; part += [np.abs(d['h3'][:, 1])]
        else:
            L += [nrm(d['h3'])]; part += [np.abs(d['h3'][:, 1])]
        part += [np.maximum.reduce([d['rb3'], nrm(d['b4']), nrm(d['b4'] - np.c_[d['rb3'], 0 * d['rb3']])]), nrm(d['q'])]
        return np.min(np.stack(L), 0), np.max(np.stack(part), 0)

    def sensitivity(self, d, f):
        """Jacobian of E(u), M(u) w.r.t. every manufactured dimension at the SUB steps"""
        P = len(d['E1']); flat = flatten(d, self.ec); Dn = flat.shape[1]; h = 1e-3
        Pert = np.repeat(flat, Dn, 0) + h * np.tile(np.eye(Dn), (P, 1))
        rep = lambda a: np.repeat(a, Dn, 0)
        dd = {k: (rep(v) if isinstance(v, np.ndarray) and v.shape[:1] == (P,) else v) for k, v in d.items()}
        dd = unflatten(Pert, self.ec, dd)
        ref = dict(pe=rep(f['pe'][:, SUB]), psi=rep(f['psi'][:, SUB]), b3=rep(f['b3'][:, SUB]))
        if f.get('b1') is not None: ref['b1'] = rep(f['b1'][:, SUB])
        Rh = humerus_frames(dd['axis'], dd['th'], U[SUB])
        g = forward(dd, ref, U[SUB], Rh=Rh)
        JE = ((g['E'] - rep(f['E'][:, SUB])) / h).reshape(P, Dn, -1)
        JM = ((g['M'] - rep(f['M'][:, SUB])) / h).reshape(P, Dn, -1)
        gE, gM = self.gain(d, f)
        sE = np.sqrt(np.einsum('pdn,d->pn', JE ** 2, self.sig ** 2) + (gE[:, SUB] * SIG_MOTOR) ** 2)
        sM = np.sqrt(np.einsum('pdn,d->pn', JM ** 2, self.sig ** 2) + (gM[:, SUB] * SIG_MOTOR) ** 2)
        # rod forces per unit output torque: |d(theta)/d(rod length)| rad/mm == N per N*mm
        names = [k for k, n in self.dims for _ in range(n)]
        ridx = [i for i, k in enumerate(names) if k in RODS]
        rf = np.radians(np.maximum(np.abs(JE[:, ridx]), np.abs(JM[:, ridx]))).max((1, 2))
        worst = np.einsum('pdn,d->pn', np.abs(JE), np.full(Dn, 0.05)).max(1), np.einsum('pdn,d->pn', np.abs(JM), np.full(Dn, 0.05)).max(1)
        return 2 * np.maximum(sE, sM).max(1), dict(rodforce=rf, worst005=np.maximum(*worst), JE=JE, JM=JM)

    def gain(self, d, f):
        """local joint-angle gain per degree of motor rotation, at each step (central differences)"""
        th = np.degrees(d['th'])[:, None]
        gE = np.abs(np.gradient(f['E'], U, axis=1)) / th; gM = np.abs(np.gradient(f['M'], U, axis=1)) / th
        return gE, gM

    def protrusion(self, d, f):
        X = np.stack(f['pts'], 2)[:, SUB]                     # (P,n,k,3)
        Rh = f['Rh'][:, SUB]
        Xh = np.einsum('pnji,pnkj->pnki', Rh, X)
        wr = wing_region_dist(Xh, f['pe'][:, SUB, None], f['psi'][:, SUB, None], d['fz'][:, None, None])
        sd = self.body.sd(X)
        o = np.where(sd <= -0.5, 0, np.minimum(np.maximum(0, sd + 0.5), wr))
        return o.max((1, 2))

    # ------------------------------------------------------------ full evaluation
    def __call__(self, X, extra=False):
        P = X.shape[0]
        BIG = 1e3
        d, ref, info = synthesise(self.ec, self.wc, X, U)
        with np.errstate(all='ignore'):
            f = forward(d, ref, U, Rh=info['Rh'], detail=True)
            Et = E0 + (d['E1'][:, None] - E0) * U; Mt = M0 + (d['M1'][:, None] - M0) * U
            err = np.maximum(np.abs(f['E'] - Et).max(1), np.abs(f['M'] - Mt).max(1))
            tmin = np.min([t.min(1) for t in f['trans']], 0)
            marg = np.min([m.min(1) for m in f['margins']], 0)
            valid = info['ok'] & np.isfinite(err) & (marg > 0)
            prot = self.protrusion(d, f)
            sens, sx = self.sensitivity(d, f)
            fit, fx = self.fold_fit(d)
            lev, part = self.levers(d, f)
            swiv = np.max(np.stack(f['swivel']), 0)
            sweep = self.sweep(d)
            rods = np.stack([d['Lc1'], d['L2'], d['Lc3'], d['L4']])
            gE, gM = self.gain(d, f); gmax = np.maximum(gE.max(1), gM.max(1))
        F = np.c_[err, -tmin, prot, sens, fit, part]
        F = np.where(valid[:, None] & np.isfinite(F), F, BIG)
        G = np.c_[np.where(valid, 0, 1) * BIG + 0 * err,
                  MARGIN_MIN - marg, LMIN - lev, 4 - rods.min(0), rods.max(0) - 40, swiv - SWIVEL, SWEEP_MIN - sweep, fit - FIT_MAX,
                  np.maximum(info['res_e'], info['res_w']) - 0.3, gmax - GAIN_MAX, TMIN_MIN - tmin]
        G = np.where(np.isfinite(G), G, BIG)
        if not extra: return F, G
        return F, G, dict(d=d, f=f, err=err, tmin=tmin, marg=marg, prot=prot, sens=sens, fit=fit, part=part, lev=lev, swivel=swiv,
                          sweep=sweep, rods=rods, valid=valid, gain=gmax, res_e=info['res_e'], res_w=info['res_w'], rodforce=sx['rodforce'],
                          worst005=sx['worst005'], fitx=fx, trans=np.stack([t.min(1) for t in f['trans']], 1),
                          margins=np.stack([m.min(1) for m in f['margins']], 1))


class LinkProblem(Problem):
    def __init__(self, ec, wc, width=45):
        self.ev = Evaluator(ec, wc, width)
        xl = np.array([v[1] for v in self.ev.vars], float); xu = np.array([v[2] for v in self.ev.vars], float)
        super().__init__(n_var=len(xl), n_obj=6, n_ieq_constr=11, xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        F, G = self.ev(X); out['F'] = F; out['G'] = G
