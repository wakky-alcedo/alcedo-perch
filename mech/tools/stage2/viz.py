"""Figures for stage 2: folded wing vs the CG body (SDF zero contours), top/side/sections. Output goes to results/local (shows CG shape)."""
import sys, pathlib, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from body import Body
from mech import *


def contour(ax, b, axis, val, i, j, **kw):
    """zero contour of the body SDF on the plane coord[axis] = val, plotted in coords (i, j)"""
    k = int(round((val - b.lo[axis]) / b.pitch)); k = np.clip(k, 0, b.shape[axis] - 1)
    for G, c, ls in ((b.sdf, kw.get('c', '#5b7a99'), '-'), (b.sdf_t, '#9fb3c6', ':')):
        S = np.take(G, k, axis=axis)
        xs = b.lo[i] + np.arange(b.shape[i]) * b.pitch; ys = b.lo[j] + np.arange(b.shape[j]) * b.pitch
        A = S if i < j else S.T
        ax.contour(xs, ys, A.T, levels=[0], colors=[c], linestyles=ls, linewidths=1.3)


def silhouette(ax, b, i, j):
    S = b.sdf.min(axis=3 - i - j)
    xs = b.lo[i] + np.arange(b.shape[i]) * b.pitch; ys = b.lo[j] + np.arange(b.shape[j]) * b.pitch
    A = S if i < j else S.T
    ax.contourf(xs, ys, A.T, levels=[-100, 0], colors=['#dfe7ef'])


def fold_figure(d, b, out, title='', k=0, mech_pts=None):
    fw = folded_wing(d['Rf'][k:k + 1], d['E1'][k:k + 1], d['M1'][k:k + 1], d['fz'][k:k + 1] if 'fz' in d else None)
    feath = np.concatenate([fw['sec'], fw['pri']], 1)[0]; bones = np.array([[0, 0, 0], fw['E'][0], fw['W'][0], fw['T'][0]])
    fig, axs = plt.subplots(2, 4, figsize=(26, 13))
    for ax, (i, j, t) in zip(axs[0, :2], [(0, 1, 'top (x lateral, y cranial)'), (1, 2, 'side (y cranial, z dorsal)')]):
        silhouette(ax, b, i, j)
        ax.scatter(b.vis[:, i], b.vis[:, j], s=4, c='#2a7fd4', label='CG folded wing (visible)')
        ax.scatter(feath[:, i], feath[:, j], s=5, c='#e0702a', label='folded feathers (model)')
        ax.plot(bones[:, i], bones[:, j], 'r-o', lw=2.5, ms=4, label='bones')
        ax.plot(*b.tip[[i, j]], 'b*', ms=14, label='CG wing tip')
        if mech_pts is not None: ax.scatter(mech_pts[:, i], mech_pts[:, j], s=10, c='k', label='linkage (folded)')
        ax.plot([0], [0], 'k+', ms=14, mew=2); ax.set_aspect('equal'); ax.grid(alpha=.3); ax.set_title(t); ax.legend(fontsize=8, loc='lower left')
    ys = [10, 0, -10, -25, -40, -55]
    for ax, y in zip(list(axs[0, 2:]) + list(axs[1, :]), ys):
        contour(ax, b, 1, y, 0, 2)
        sel = np.abs(b.vis[:, 1] - y) < 2.5; ax.scatter(b.vis[sel, 0], b.vis[sel, 2], s=10, c='#2a7fd4')
        sel = np.abs(feath[:, 1] - y) < 2.5; ax.scatter(feath[sel, 0], feath[sel, 2], s=10, c='#e0702a')
        t = np.linspace(0, 1, 60)[:, None]
        B = np.concatenate([bones[0] + (bones[1] - bones[0]) * t, bones[1] + (bones[2] - bones[1]) * t, bones[2] + (bones[3] - bones[2]) * t])
        sel = np.abs(B[:, 1] - y) < 1.2; ax.scatter(B[sel, 0], B[sel, 2], s=25, c='r')
        if mech_pts is not None:
            sel = np.abs(mech_pts[:, 1] - y) < 2.5; ax.scatter(mech_pts[sel, 0], mech_pts[sel, 2], s=14, c='k')
        ax.axvline(-9, color='0.7', lw=0.8); ax.plot([0], [0], 'k+', ms=12)
        ax.set_aspect('equal'); ax.grid(alpha=.3); ax.set_title(f'section y = {y} (solid: body, dotted: torso only)'); ax.set_xlim(-45, 30); ax.set_ylim(-40, 28)
    axs[1, 3].axis('off')
    fig.suptitle(title, fontsize=13); plt.tight_layout(); plt.savefig(out, dpi=65); plt.close(fig)
