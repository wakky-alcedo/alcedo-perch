"""Stage-2d check: do the rod forces change sign during the fold (then joint play shows up as backlash), and how much
spring preload at the elbow and wrist would keep every rod loaded one way, for any direction of gravity.

Energy method with the motor locked: the force in rod k is t_k = dV/dL_k (V = gravity + spring potential), by finite
differences of the forward kinematics with that rod's length changed. Springs: constant torque tau at the elbow and the
wrist (signs chosen for the best result). Masses are ESTIMATES: forearm + secondaries M_U at the middle of the ulna,
hand + primaries M_H at W + 1.2 (T - W). Forces scale linearly with them.
usage: loads.py   -> results/stage2/2d/loads.json"""
import json, pathlib, itertools, numpy as np
import sys
from problem import Evaluator, U
from mech import forward, RODS

ROOT = pathlib.Path(__file__).resolve().parents[2]
from problem import STAGE, TAG
OUT = ROOT / 'results' / 'stage2' / (STAGE + TAG if '2e' in sys.argv else '2d')
M_U, M_H = 1.5e-3, 1.0e-3          # kg (estimates)
G0 = 9.81
H = 1e-3                            # mm, finite-difference step on a rod length


def gravity_dirs(n=200):
    i = np.arange(n) + 0.5; phi = np.arccos(1 - 2 * i / n); th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


def run(c):
    ev = Evaluator(c['ec'], c['wc'], 45, tilt=('A' if c.get('dsg') == 'A' else True))   # no stack needed here
    F, Gc, e = ev(np.array([c['x']]), extra=True); d, f = e['d'], e['f']
    rods = [k for k in RODS if k in d]
    rows = []
    for k in rods:
        for s in (+1, -1):
            dd = {kk: (v.copy() if isinstance(v, np.ndarray) else v) for kk, v in d.items()}
            dd[k] = dd[k] + s * H; rows.append(dd)
    ref = dict(pe=f['pe'], psi=f['psi'], b3=f['b3'])
    if f.get('b1') is not None: ref['b1'] = f['b1']
    res = []
    with np.errstate(all='ignore'):
        for dd in rows:
            g = forward(dd, ref, U, Rh=f['Rh'], detail=True)
            cu = (g['E_w'] + g['W_w'])[0] / 2; ch = (g['W_w'] + 1.2 * (g['T_w'] - g['W_w']))[0]
            res.append(dict(E=np.radians(g['E'][0]), M=np.radians(g['M'][0]), cu=cu, ch=ch))
    dirs = gravity_dirs()
    out = dict(id=c['id'], ec=c['ec'], wc=c['wc'], rods={})
    # gravity part of the rod force (N), (rod, dir, step); spring part per 1 N mm at elbow / wrist, (rod, step)
    Gk, SE, SW = [], [], []
    for j, k in enumerate(rods):
        p, m = res[2 * j], res[2 * j + 1]
        dV = (M_U * (p['cu'] - m['cu']) + M_H * (p['ch'] - m['ch'])) / (2 * H)          # kg mm / mm  per unit g direction
        Gk.append(-G0 * np.einsum('sx,gx->gs', dV, dirs) * 1e-3 * 1e3)                    # V = -m g.r  -> N (mm cancels)
        SE.append(-(p['E'] - m['E']) / (2 * H)); SW.append(-(p['M'] - m['M']) / (2 * H))  # V_s = -tau E -> t = -tau dE/dL  (N per N mm)
    Gk, SE, SW = np.array(Gk), np.array(SE), np.array(SW)
    out['gravity_max_N'] = {k: float(np.abs(Gk[j]).max()) for j, k in enumerate(rods)}
    best = None
    for sE, sW in itertools.product((1, -1), (1, -1)):
        for ratio in np.linspace(0, 1, 21):                     # share of the spring torque at the elbow (total 1 N mm)
            S = sE * ratio * SE + sW * (1 - ratio) * SW          # (rod, step) per N mm
            smin = np.abs(S).min(1); sgn_ok = np.all(np.sign(S) == np.sign(S[:, :1]), 1)
            if not sgn_ok.all(): continue
            need = float(np.max(np.abs(Gk).max(1).max(1) / smin))     # N mm of spring so that gravity never wins
            if best is None or need < best['tau_Nmm']:
                best = dict(tau_Nmm=need, elbow_share=float(ratio), signs=(sE, sW),
                            tau_E=need * ratio, tau_W=need * (1 - ratio), Smin={k: float(smin[j]) for j, k in enumerate(rods)})
    # without springs: does any rod force change sign during the fold for some gravity direction?
    flips = {k: float(np.mean(np.any(np.sign(Gk[j]) != np.sign(Gk[j][:, :1]), 1))) for j, k in enumerate(rods)}
    out['share_of_dirs_with_sign_change_no_spring'] = flips
    out['spring'] = best
    # angle change per mm of rod length (elbow E, wrist M), max over the fold -> how much a rod's elastic stretch costs
    out['dangle_dL_deg_per_mm'] = {k: dict(E=float(np.degrees(np.abs(res[2 * j]['E'] - res[2 * j + 1]['E']).max() / (2 * H))),
                                           M=float(np.degrees(np.abs(res[2 * j]['M'] - res[2 * j + 1]['M']).max() / (2 * H))))
                                   for j, k in enumerate(rods)}
    if best:                                   # rod forces with the spring at twice the minimum preload (margin), worst gravity direction
        sE, sW = best['signs']; tE, tW = 2 * best['tau_E'], 2 * best['tau_W']
        out['force_max_with_spring_N'] = {k: float(np.abs(Gk[j] + (sE * tE * SE[j] + sW * tW * SW[j])[None]).max()) for j, k in enumerate(rods)}
    if best:
        th = np.radians(c['theta'])
        dEdth = np.abs(np.gradient(np.radians(f['E'][0]), U)) / th; dMdth = np.abs(np.gradient(np.radians(f['M'][0]), U)) / th
        out['spring']['motor_extra_Nmm'] = float(np.max(best['tau_E'] * dEdth + best['tau_W'] * dMdth))
    return out


def main():
    cands = json.load(open(OUT / 'candidates.json', encoding='utf-8'))
    out = {}
    for c in cands:
        r = run(c); out[c['id']] = r
        sp = r['spring']
        print(c['id'], c['ec'], c['wc'], 'gravity max N', {k: round(v, 4) for k, v in r['gravity_max_N'].items()},
              'flip share', {k: round(v, 2) for k, v in r['share_of_dirs_with_sign_change_no_spring'].items()},
              'spring', None if not sp else dict(tau=round(sp['tau_Nmm'], 2), tauE=round(sp['tau_E'], 2), tauW=round(sp['tau_W'], 2), motor=round(sp['motor_extra_Nmm'], 2)), flush=True)
    json.dump(out, open(OUT / 'loads.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
