"""Render the Belted Kingfisher scan (bird region only: y > ymin) from several directions."""
import sys, pathlib, numpy as np, trimesh, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from parts import draw
ROOT = pathlib.Path(__file__).resolve().parents[2]
ymin = float(sys.argv[2]) if len(sys.argv) > 2 else -160
s = trimesh.load(ROOT / 'ref' / 'models' / 'belted_kingfisher.glb'); gs = list(s.dump())
ms, cols = [], []
for gi, g in enumerate(gs):
    keep = g.vertices[g.faces].mean(1)[:, 1] > ymin
    m = g.submesh([np.where(keep)[0]], append=True); ms.append(m); cols.append([(0.95, 0.6, 0.3), (0.5, 0.7, 0.95)][gi])
fig, axs = plt.subplots(2, 3, figsize=(21, 14))
for ax, (v, up, t) in zip(axs.flat, [((0, 0, -1), (0, 1, 0), 'from +z'), ((0, 0, 1), (0, 1, 0), 'from -z'), ((1, 0, 0), (0, 1, 0), 'from -x'),
                                     ((-1, 0, 0), (0, 1, 0), 'from +x'), ((0, -1, 0), (0, 0, 1), 'top'), ((0.6, -0.3, 0.7), (0, 1, 0), 'oblique')]):
    draw(ax, ms, cols, v, up, t)
plt.tight_layout(); plt.savefig(sys.argv[1], dpi=60)
