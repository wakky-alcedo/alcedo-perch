"""Layer plan of each stage-2d candidate: which part sits in which layer, drawn in the arm plane (humerus frame) at the open,
middle and folded pose. No reference-model content -> results/stage2/2d/layers_<id>.png (in git).
usage: layer_figs.py"""
import json, pathlib, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from problem import Evaluator, LAYER_T, R_BOSS

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / 'results' / 'stage2' / '2d'
COL = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']
NAME = dict(hum='humerus', uln='ulna', hand='hand', bc1='bellcrank 1', bc2='bellcrank 2', rod1='rod 1 (elbow)', rod2='rod 2', rod3='rod 3', rod4='rod 4 (wrist)')


def colouring(adj, n):
    """minimum proper colouring by backtracking (n <= 9); adj: list of int bitmasks"""
    for k in range(1, n + 1):
        col = [-1] * n

        def bt(i):
            if i == n: return True
            for c in range(k):
                if all(not (adj[i] >> j & 1) or col[j] != c for j in range(i)):
                    col[i] = c
                    if bt(i + 1): return True
            col[i] = -1; return False
        if bt(0): return col


def main():
    cands = json.load(open(OUT / 'candidates.json', encoding='utf-8'))
    for c in cands:
        ev = Evaluator(c['ec'], c['wc'], 45, tilt=True)
        F, G, e = ev(np.array([c['x']]), extra=True); f = e['f']; lx = e['layx']
        names = lx['names']; n = len(names)
        col = colouring([int(v) for v in lx['A_all'][0]], n)
        steps = [0, 20, 40]
        P_, adj, _ = ev.parts(f, k=steps)
        Rh = f['Rh'][0, steps]
        fig, axs = plt.subplots(1, 3, figsize=(18, 6.5))
        for a, (ax, k) in enumerate(zip(axs, steps)):
            for nm in names:
                for p, q, r in P_[nm]:
                    ph, qh = Rh[a].T @ p[0, a], Rh[a].T @ q[0, a]
                    lw = max(1.5, r * 2 * 72 / 25.4 * 0.55)
                    if np.allclose(ph, qh): ax.add_patch(plt.Circle(ph[:2], r, color=COL[col[names.index(nm)]], alpha=0.35))
                    else: ax.plot([ph[0], qh[0]], [ph[1], qh[1]], color=COL[col[names.index(nm)]], lw=lw, alpha=0.55, solid_capstyle='round')
            ax.set_aspect('equal'); ax.grid(alpha=.3); ax.set_title(f"u = {k / 40:.1f}  (" + ('open' if k == 0 else 'folded' if k == 40 else 'middle') + ')')
            ax.set_xlabel('x along humerus (mm)'); ax.set_ylabel('y (mm)')
        hand = [plt.Line2D([0], [0], color=COL[col[i]], lw=6, alpha=0.6, label=f'layer {col[i] + 1}: {NAME.get(nm, nm)}') for i, nm in sorted(enumerate(names), key=lambda t: col[t[0]])]
        axs[2].legend(handles=hand, fontsize=9, loc='best')
        fig.suptitle(f"{c['id']} {c['ec']}/{c['wc']}: {max(col) + 1} layers x {LAYER_T} mm = {(max(col) + 1) * LAYER_T:.1f} mm (wing part {c['layers_wing']} layers); "
                     f"drawn in the humerus frame, line width ~ part envelope", fontsize=12)
        plt.tight_layout(); fig.subplots_adjust(top=0.86); plt.savefig(OUT / f"layers_{c['id']}.png", dpi=70); plt.close(fig)
        c['layer_of'] = {nm: col[i] + 1 for i, nm in enumerate(names)}
        print(c['id'], c['layer_of'])
    json.dump(cands, open(OUT / 'candidates.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
