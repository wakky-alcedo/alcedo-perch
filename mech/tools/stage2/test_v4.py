"""Check: v4 through synthesise/forward vs the simulator dump."""
import json, sys, pathlib, numpy as np
from mech import *
ROOT = pathlib.Path(__file__).resolve().parents[2]
P = json.load(open(ROOT / 'results' / 'v4_pose.json'))
sim = json.load(open(ROOT / 'results' / 'stage1' / 'v4_path.json'))
p = sim['params']
# reconstruct the v4 design vector (elbow 'body', wrist 'ulna')
axis, th, Rf = shoulder(ypr_to_rv(np.array([P['x'][:3]])))
print('axis', axis.round(4), 'theta', np.degrees(th).round(2), 'vs', P['axis'], P['theta'])
Ab = hinge_frame(axis)[0]; G1 = np.array([p['G1x'], p['G1y'], p['G1z']])
g = Ab[:, :2].T @ G1
# branches: try all and keep the one matching the simulator
best = None
for br1 in (0, 1):
    for br3 in (0, 1):
        X = np.array([[*ypr_to_rv(np.array([P['x'][:3]]))[0], P['E1'], P['M1'], g[0], g[1], p['h1x'], p['h1y'], p['h1z'], p['rb1'], p['Lc1'], p['wx'], p['wy'], br1,
                       p['G2x'], p['G2y'], p['h3x'], p['h3y'], p['rb3'], p['Lc3'], p['qx'], p['qy'], br3]])
        u = np.linspace(0, 1, 161)
        d, ref, info = synthesise('body', 'ulna', X, u)
        if not info['ok'][0]: continue
        f = forward(d, ref, u)
        Es = np.array([s['Edeg'] for s in sim['steps']]); Ms = np.array([s['Mdeg'] for s in sim['steps']])
        err = max(np.abs(f['E'][0] - Es).max(), np.abs(f['M'][0] - Ms).max())
        print('br', br1, br3, 'axial G1 check', (G1 @ axis[0]).round(3), 'vs H1 axial', 'b2', d['b2'][0].round(3), 'L2 %.3f' % d['L2'][0], 'b4', d['b4'][0].round(3), 'L4 %.3f' % d['L4'][0],
              'max |E-sim|,|M-sim| %.3f' % err, 'tmin %.1f' % min(t.min() for t in f['trans']))
print('sim dims b2', p['b2x'], p['b2y'], p['b2z'], 'L2', p['L2'], 'b4', p['b4x'], p['b4y'], 'L4', p['L4'])
# exact simulator dimensions into forward()
X = np.array([[*ypr_to_rv(np.array([P['x'][:3]]))[0], P['E1'], P['M1'], g[0], g[1], p['h1x'], p['h1y'], p['h1z'], p['rb1'], p['Lc1'], p['wx'], p['wy'], 1,
               p['G2x'], p['G2y'], p['h3x'], p['h3y'], p['rb3'], p['Lc3'], p['qx'], p['qy'], 0]])
u = np.linspace(0, 1, 161); d, ref, info = synthesise('body', 'ulna', X, u)
d['G1w'] = G1[None]; d['b2'] = np.array([[p['b2x'], p['b2y'], p['b2z']]]); d['L2'] = np.array([p['L2']]); d['b4'] = np.array([[p['b4x'], p['b4y']]]); d['L4'] = np.array([p['L4']])
f = forward(d, ref, u, detail=True)
Es = np.array([s['Edeg'] for s in sim['steps']]); Ms = np.array([s['Mdeg'] for s in sim['steps']])
print('exact dims: max |E-sim| %.4f |M-sim| %.4f' % (np.abs(f['E'][0] - Es).max(), np.abs(f['M'][0] - Ms).max()))
print('tmin', min(t.min() for t in f['trans']).round(2), 'sim trans min', min(min(s['trans']) for s in sim['steps']))
print('swivel rod2 (deg)', [s.round(1) for s in f['swivel']], 'margins (mm)', [m.min().round(2) for m in f['margins']])
d['th'] = np.radians([108.0]); f = forward(d, ref, u)
print('theta=108 (simulator slider): max |E-sim| %.5f |M-sim| %.5f' % (np.abs(f['E'][0] - Es).max(), np.abs(f['M'][0] - Ms).max()))
