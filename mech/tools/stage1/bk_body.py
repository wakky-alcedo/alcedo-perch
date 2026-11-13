"""Belted Kingfisher in its body frame (x right, y cranial, z dorsal, origin torso centroid). Bird-only surface samples."""
import sys, json, pathlib, numpy as np, trimesh
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from bk_load import bk_mesh, ROOT
from bk_frame import to_frame
FR = json.load(open(ROOT / 'results' / 'stage1' / 'bk_frame.json'))
R, O = np.array(FR['R']), np.array(FR['origin'])


def bk_bird_mesh():
    m = bk_mesh(); c = to_frame(m.vertices[m.faces].mean(1))
    keep = (c[:, 1] > -55) | ((c[:, 0] < -150) & (c[:, 1] > -170))
    b = m.submesh([np.where(keep)[0]], append=True)
    b.vertices = (b.vertices - O) @ R.T
    return b


if __name__ == '__main__':
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from parts import draw
    b = bk_bird_mesh()
    fig, axs = plt.subplots(1, 3, figsize=(26, 10))
    for ax, (v, up, t) in zip(axs, [((-1, 0, 0), (0, 0, 1), 'from right (+x): screen = (-y? check), z up'), ((0, -1, 0), (0, 0, 1), 'from front'), ((0, 0, -1), (0, 1, 0), 'from top')]):
        draw(ax, [b], [(0.8, 0.8, 0.85)], v, up, t); ax.minorticks_on(); ax.grid(which='minor', alpha=.15)
    plt.tight_layout(); plt.savefig(sys.argv[1], dpi=60)
