"""Belted Kingfisher scan: cross sections (perpendicular to the torso axis) and silhouettes, scaled to a 78.8 mm wing
(x 78.8/160). Shows where the folded wing lies on a real (taxidermy) bird. Origin = torso centroid (no shoulder estimate)."""
import sys, pathlib, numpy as np, trimesh, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from bk_body import bk_bird_mesh, ROOT
from parts import draw

S = 78.8 / 160.0
b = bk_bird_mesh(); b.vertices = b.vertices * S
PTS, _ = trimesh.sample.sample_surface_even(b, 250000, seed=0)
fig = plt.figure(figsize=(26, 12))
ax = fig.add_subplot(2, 5, 1); draw(ax, [b], [(0.8, 0.8, 0.85)], (-1, 0, 0), (0, 0, 1), 'BK from right (screen x = -y)')
ax = fig.add_subplot(2, 5, 2); draw(ax, [b], [(0.8, 0.8, 0.85)], (0, 0, -1), (0, 1, 0), 'BK from top')
ax = fig.add_subplot(2, 5, 3); draw(ax, [b], [(0.8, 0.8, 0.85)], (0, 1, 0), (0, 0, 1), 'BK from behind')
ys = [12, 6, 0, -6, -12, -20, -28]
for k, y in enumerate(ys):
    ax = fig.add_subplot(2, 5, 4 + k)
    s_ = PTS[np.abs(PTS[:, 1] - y) < 0.5]
    ax.scatter(s_[:, 0], s_[:, 2], s=1.5, c='k')
    ax.axvline(0, color='0.6', lw=0.8); ax.set_aspect('equal'); ax.grid(alpha=.3); ax.set_title(f'BK section y = {y} mm (scaled), x right, z dorsal')
    ax.set_xlim(-28, 28); ax.set_ylim(-30, 30)
plt.tight_layout(); plt.savefig(ROOT / 'results' / 'stage1' / 'bk_sections.png', dpi=65)
