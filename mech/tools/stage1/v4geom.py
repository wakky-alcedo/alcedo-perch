"""Rebuild v4 geometry (bones, feather quads, mechanism samples) from the simulator dump, in the v4 shoulder frame (mm)."""
import json, pathlib, numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]
LH, LU, LCMC, LDIG = 25.7, 32.0, 13.3, 7.0
LMAN = LCMC + LDIG
PRIM = [[0.05, 40, -82, -8], [0.15, 42, -75, -7], [0.25, 44, -68, -6], [0.35, 46, -61, -5], [0.45, 48, -54, -4],
        [0.55, 50, -46, -3.5], [0.68, 53, -38, -3], [0.80, 56, -29, -2.5], [0.92, 57, -20, -2], [1.00, 53, -11, -1.5]]
SEC = [[0.96 - i * 0.095, 40 - i * 0.9, -100 - i * 3.0, -170 + i * 0.8] for i in range(10)]


def load(id='v4'):
    return json.load(open(ROOT / 'results' / 'stage1' / f'{id}_path.json'))


def feathers(q, target):
    """feather quads exactly as featherQuads() in the simulator. returns list of (kind, 5x3 polygon)"""
    E, W, Ru, Rm = map(np.array, (q['E'], q['W'], q['Ru'], q['Rm']))
    fw = np.clip((q['Mdeg'] - target['M1']) / (target['M0'] - target['M1']), 0, 1)
    fe = np.clip((q['Edeg'] - target['E1']) / (target['E0'] - target['E1']), 0, 1)
    out = []

    def make(base, F, ang, L, w, kind, order):
        a = np.radians(ang); d = F @ [np.cos(a), np.sin(a), 0]; s = F @ [-np.sin(a), np.cos(a), 0]
        tip = base + d * L; sh = base + d * L * 0.82; drop = F @ [0, 0, -0.35 * order]
        out.append((kind, np.array([base + s * w * 0.22, sh + s * w * 0.5, tip, sh - s * w * 0.5, base - s * w * 0.22]) + drop))
    for i, (f, L, a0, a1) in enumerate(SEC): make(E + Ru @ [LU * f, 0, 0], Ru, a1 + (a0 - a1) * fe, L, 7.5, 's', i * 0.2)
    for i, (f, L, a0, a1) in enumerate(PRIM): make(W + Rm @ [LMAN * f, 0, 0], Rm, a1 + (a0 - a1) * fw, L, 7, 'p', 2 + i * 0.25)
    return out


def bones(q, n=6):
    O = np.zeros(3); E, W, T = map(np.array, (q['E'], q['W'], q['T']))
    return np.vstack([O + (E - O) * t for t in np.linspace(0, 1, n)] + [E + (W - E) * t for t in np.linspace(0, 1, n)] + [W + (T - W) * t for t in np.linspace(0, 1, n)])


def feather_points(q, target, per=6):
    pts = []
    for kind, P in feathers(q, target):
        base = (P[0] + P[4]) / 2
        pts += [base + (P[2] - base) * t for t in np.linspace(0, 1, per)] + [P[1], P[3]]
    return np.array(pts)


if __name__ == '__main__':
    d = load(); q = d['steps'][-1]
    print('folded: E', np.round(q['E'], 1), 'W', np.round(q['W'], 1), 'T', np.round(q['T'], 1), 'Edeg %.1f Mdeg %.1f' % (q['Edeg'], q['Mdeg']))
    tips = np.array([P[2] for k, P in feathers(q, d['target']) if k == 'p'])
    print('primary tips', np.round(tips[[0, 5, 8, 9]], 1))
    fp = feather_points(q, d['target']); print('folded wing bbox', np.round(fp.min(0), 1), np.round(fp.max(0), 1))
    q0 = d['steps'][0]; print('open: E', np.round(q0['E'], 1), 'W', np.round(q0['W'], 1), 'T', np.round(q0['T'], 1))
