"""Belted Kingfisher: bird-aligned frame. u = forward (horizontal, tail->bill), v = up (model +y), w = bird's right.
Renders side/front/top in that frame for landmark picking."""
import sys, pathlib, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from bk_load import bk_mesh
from parts import draw
import trimesh
BILL = np.array([-85.5, -1.8, -11.7]); TAILX = np.array([93.3, -94.0, 82.8])
d = BILL - TAILX; d[1] = 0; FWD = d / np.linalg.norm(d); UP = np.array([0, 1.0, 0]); RIGHT = np.cross(FWD, UP)
R_BK = np.vstack([FWD, UP, RIGHT])      # p_frame = R_BK @ (p - BILL)


def to_frame(V):
    return (np.asarray(V) - BILL) @ R_BK.T


if __name__ == '__main__':
    m = bk_mesh(); c = m.vertices[m.faces].mean(1)
    keep = c[:, 1] > float(sys.argv[2]) if len(sys.argv) > 2 else c[:, 1] > -170
    mm = m.submesh([np.where(keep)[0]], append=True); mm.vertices = to_frame(mm.vertices)
    fig, axs = plt.subplots(1, 3, figsize=(24, 10))
    for ax, (v, up, t) in zip(axs, [((0, 0, 1), (0, 1, 0), 'side from bird right (u fwd -> left? see ticks), v up'),
                                    ((-1, 0, 0), (0, 1, 0), 'front (looking backward)'), ((0, -1, 0), (-1, 0, 0), 'top')]):
        draw(ax, [mm], [(0.8, 0.8, 0.85)], v, up, t)
        ax.minorticks_on(); ax.grid(which='minor', alpha=.15)
    plt.tight_layout(); plt.savefig(sys.argv[1], dpi=60)
