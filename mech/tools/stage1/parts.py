"""Label the connected components of the Common Kingfisher model and render them coloured.
Writes results/stage1/ck_parts.json (component index -> label, bounds)."""
import sys, json, pathlib, numpy as np, trimesh, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(pathlib.Path(__file__).parent))


def ck_components():
    s = trimesh.load(ROOT / 'ref' / 'models' / 'common_kingfisher.glb')
    g = list(s.dump())[0]
    parts = sorted(g.split(only_watertight=False), key=lambda p: -len(p.faces))
    return parts


def draw(ax, meshes, cols, view, up, title):
    view = np.asarray(view, float); view /= np.linalg.norm(view); up = np.asarray(up, float)
    r = np.cross(up, view); r /= np.linalg.norm(r); u = np.cross(view, r)
    allP, allC, allD = [], [], []
    for m, c in zip(meshes, cols):
        V, F = m.vertices, m.faces; tri = V[F]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); n /= np.linalg.norm(n, axis=1)[:, None] + 1e-12
        L = -view + 0.5 * u; L /= np.linalg.norm(L); sh = 0.3 + 0.7 * np.abs(n @ L)
        allP.append(np.stack([tri @ r, tri @ u], -1)); allC.append(np.clip(np.outer(sh, c[:3]), 0, 1)); allD.append((tri @ view).mean(1))
    P = np.concatenate(allP); C = np.concatenate(allC); Dd = np.concatenate(allD); o = np.argsort(Dd)[::-1]
    ax.add_collection(PolyCollection(P[o], facecolors=C[o], edgecolors='none'))
    ax.set_xlim(P[..., 0].min(), P[..., 0].max()); ax.set_ylim(P[..., 1].min(), P[..., 1].max()); ax.set_aspect('equal'); ax.set_title(title); ax.grid(alpha=.3)


if __name__ == '__main__':
    parts = ck_components()
    cmap = plt.get_cmap('tab20')
    info = []
    for i, p in enumerate(parts):
        info.append(dict(i=i, faces=len(p.faces), watertight=bool(p.is_watertight), bounds=np.round(p.bounds, 2).tolist(), centroid=np.round(p.centroid, 2).tolist()))
    cols = [cmap(i % 20) for i in range(len(parts))]
    fig, axs = plt.subplots(1, 3, figsize=(21, 8))
    for ax, (v, up, t) in zip(axs, [((0, 0, -1), (0, 1, 0), 'side from +z'), ((0, -1, 0), (-1, 0, 0), 'top'), ((1, 0, 0), (0, 1, 0), 'front')]):
        draw(ax, parts, cols, v, up, t)
    for i, p in enumerate(parts[:16]):
        c = p.centroid; axs[0].text(c[0], c[1], str(i), fontsize=9, color='k')
    plt.tight_layout(); plt.savefig(sys.argv[1], dpi=65)
    for d in info[:20]: print(d)
