import json, sys, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
sys.argv = ['v4pose.py']; exec(open('v4pose.py').read().split("if __name__ == '__main__':")[0])
def show(x, fn, title):
    Rf = folded_frame(x); g = wing_points(Rf, x[3], x[4], 0, 0)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for k, (i, j, lab) in enumerate([(0, 1, 'top (x,y)'), (1, 2, 'side (y,z)'), (0, 2, 'back (x,z)')]):
        a = ax[k]; a.set_title(title + ' ' + lab); a.set_aspect('equal')
        a.add_patch(Ellipse((BC[i], BC[j]), 2 * BR[i], 2 * BR[j], fc='#cde', ec='k'))
        for side in (1, -1):
            f = (lambda P: P) if side == 1 else (lambda P: np.c_[2 * BC[0] - P[:, 0], P[:, 1], P[:, 2]] if P.ndim == 2 else np.array([2 * BC[0] - P[0], P[1], P[2]]))
            sk = np.array([[0, 0, 0], g['E'], g['W'], g['T']]); sk = f(sk)
            a.plot(sk[:, i], sk[:, j], 'o-', c='C0' if side == 1 else 'C1')
            fe = f(g['feath']); a.plot(fe[:, i], fe[:, j], '.', ms=3, c='C2')
        a.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(fn, dpi=70)
d = json.load(open('v4_pose.json'))
show(np.array(d['x']), 'v4pose_a.png', 'opt')
