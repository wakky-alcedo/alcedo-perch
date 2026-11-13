"""Flat-shaded renders of a mesh from several directions (matplotlib). usage: shade.py name out.png [geom_index]"""
import sys, pathlib, numpy as np, trimesh, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
ROOT = pathlib.Path(__file__).resolve().parents[2]


def render(ax, V, F, view, up, title, color=(0.75, 0.8, 0.9)):
    view = np.asarray(view, float); view /= np.linalg.norm(view)
    r = np.cross(up, view); r /= np.linalg.norm(r); u = np.cross(view, r)
    P = np.c_[V @ r, V @ u]; depth = V @ view
    tri = V[F]; n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); n /= np.linalg.norm(n, axis=1)[:, None] + 1e-12
    L = -view + 0.5 * u; L /= np.linalg.norm(L)
    sh = 0.25 + 0.75 * np.abs(n @ L)
    o = np.argsort(depth[F].mean(1))[::-1]       # far first (view points from camera into scene)
    cols = np.clip(np.outer(sh[o], color), 0, 1)
    ax.add_collection(PolyCollection(P[F[o]], facecolors=cols, edgecolors='none'))
    ax.set_xlim(P[:, 0].min(), P[:, 0].max()); ax.set_ylim(P[:, 1].min(), P[:, 1].max()); ax.set_aspect('equal'); ax.set_title(title); ax.grid(alpha=.3)


if __name__ == '__main__':
    s = trimesh.load(ROOT / 'ref' / 'models' / f'{sys.argv[1]}.glb')
    gs = list(s.dump())
    if len(sys.argv) > 3: gs = [gs[int(sys.argv[3])]]
    m = trimesh.util.concatenate(gs)
    V, F = m.vertices, m.faces
    views = [((0, 0, -1), (0, 1, 0), 'side (from +z)'), ((0, 0, 1), (0, 1, 0), 'side (from -z)'), ((1, 0, 0), (0, 1, 0), 'front (from -x)'),
             ((0, -1, 0), (-1, 0, 0), 'top (from +y)'), ((-1, 0, 0), (0, 1, 0), 'back (from +x)'), ((0.5, -0.4, -0.75), (0, 1, 0), 'oblique')]
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    for ax, (v, up, t) in zip(axs.flat, views): render(ax, V, F, v, np.array(up, float), t)
    plt.tight_layout(); plt.savefig(sys.argv[2], dpi=70)
