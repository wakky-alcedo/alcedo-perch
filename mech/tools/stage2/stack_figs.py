"""Stage 2e: real-height layer plan of each candidate (stack.full_check plan), top view (humerus frame) at the open, middle and
folded pose coloured by height, and a side view (x vs height) at the folded pose with the pins / ball studs.
No reference-model content -> results/stage2/2e/stack_<id>.png (in git).   usage: stack_figs.py"""
import json, pathlib, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from analyze import make_ev
from problem import U
from stack import Geometry, T
ROOT = pathlib.Path(__file__).resolve().parents[2]
from problem import STAGE
OUT = ROOT / 'results' / 'stage2' / STAGE
NAME = dict(hum='humerus', uln='ulna', hand='hand', bc1='bellcrank 1', rod1='rod 1 (skewer)', rod2='rod 2', rod4='rod 4 (wrist)')


def main():
    cands = json.load(open(OUT / 'candidates.json', encoding='utf-8'))
    chk = json.load(open(OUT / 'check_2e.json', encoding='utf-8'))
    cm = plt.get_cmap('viridis'); zr = (-6, 8)
    for c in cands:
        plan = chk[c['id']]['plan']
        ev = make_ev(c['ec'], c['wc'], 45, c['dsg'])
        F, G, e = ev(np.asarray(c['x'], float)[None], extra=True)
        g = Geometry(ev, e['f'], e['d'], steps=np.arange(len(U)), node_limit=1)
        steps = [0, 20, 40]
        fig = plt.figure(figsize=(20, 11)); gs = fig.add_gridspec(2, 3, height_ratios=[1.6, 1])
        for a, k in enumerate(steps):
            ax = fig.add_subplot(gs[0, a])
            for nm in g.names:
                for p, q, r in g.caps[nm]:
                    zz = lambda X: X[0, k, 2] + plan.get(nm, 0) * T
                    zm = (zz(p) + zz(q)) / 2; colr = cm((zm - zr[0]) / (zr[1] - zr[0]))
                    if np.allclose(p[0, k], q[0, k]): ax.add_patch(plt.Circle(p[0, k, :2], r, color=colr, alpha=0.5))
                    else: ax.plot([p[0, k, 0], q[0, k, 0]], [p[0, k, 1], q[0, k, 1]], color=colr, lw=max(1.5, r * 5), alpha=0.6, solid_capstyle='round',
                                  ls='--' if nm == 'rod1' else '-')
                p0 = g.caps[nm][0][0][0, k]; q0 = g.caps[nm][0][1][0, k]
                if a == 2: ax.annotate(NAME.get(nm, nm), (p0[:2] + q0[:2]) / 2, fontsize=8)
            ax.set_aspect('equal'); ax.grid(alpha=.3); ax.set_title(f"top view u = {k / 40:.1f} (" + ('open' if k == 0 else 'folded' if k == 40 else 'middle') + ')')
            ax.set_xlabel('x along humerus (mm)'); ax.set_ylabel('y (mm)')
        sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(*zr)); fig.colorbar(sm, ax=ax, label='height z (mm)', shrink=0.7)
        # side view at three poses: x vs z
        for a, k in enumerate(steps):
            ax = fig.add_subplot(gs[1, a])
            for i, nm in enumerate(g.names):
                first = True
                for p, q, r in g.caps[nm]:
                    z0 = plan.get(nm, 0) * T
                    if not np.allclose(p[0, k], q[0, k]):
                        ax.plot([p[0, k, 0], q[0, k, 0]], [p[0, k, 2] + z0, q[0, k, 2] + z0], color=plt.cm.tab10(i), lw=2.5 if nm != 'rod1' else 4,
                                alpha=0.75, label=NAME.get(nm, nm) if first else None); first = False
            for a_, b_, X, ball in g.J:
                za = g.base(a_, X)[0, k] + plan.get(a_, 0) * T; zb = g.base(b_, X)[0, k] + plan.get(b_, 0) * T
                zz = [za, zb] + ([X[0, k, 2]] if ball else [])
                ax.plot([X[0, k, 0]] * 2, [min(zz), max(zz)], color='k', lw=1.5, marker='o' if ball else '_', ms=4)
            ax.set_xlabel('x (mm)'); ax.set_ylabel('height z (mm)'); ax.grid(alpha=.3); ax.set_title(f'side view u = {k / 40:.1f} (black: pins / ball studs)')
            if a == 0: ax.legend(fontsize=7, ncol=2)
        fig.suptitle(f"{c['id']} design {c['dsg']} {c['ec']}/{c['wc']} ({c['pick']}): levels {plan} x {T} mm; studs {chk[c['id']]['studs_mm']}", fontsize=11)
        plt.tight_layout(); plt.savefig(OUT / f"stack_{c['id']}.png", dpi=65); plt.close(fig); print(c['id'])


if __name__ == '__main__':
    main()
