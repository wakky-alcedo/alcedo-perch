"""Orthographic point-cloud views (3 axes) of a reference mesh, for orientation checks. usage: views.py name out.png"""
import sys, pathlib, numpy as np, trimesh, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROOT = pathlib.Path(__file__).resolve().parents[2]
s = trimesh.load(ROOT / 'ref' / 'models' / f'{sys.argv[1]}.glb')
geoms = list(s.dump())
fig, axs = plt.subplots(len(geoms), 3, figsize=(15, 5 * len(geoms)), squeeze=False)
for gi, g in enumerate(geoms):
    V = g.vertices
    for j, (a, b, c) in enumerate([(0, 1, 2), (0, 2, 1), (2, 1, 0)]):
        o = np.argsort(V[:, c]); ax = axs[gi, j]
        ax.scatter(V[o, a], V[o, b], c=V[o, c], s=0.3, cmap='viridis'); ax.set_aspect('equal')
        ax.set_title(f'geom{gi} {"xyz"[a]} vs {"xyz"[b]} (colour {"xyz"[c]})'); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(sys.argv[2], dpi=70)
