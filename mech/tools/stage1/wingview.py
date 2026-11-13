"""Render the right folded-wing shell coloured by depth outside (+) / inside (-) the torso, in the body frame."""
import sys, pathlib, numpy as np, trimesh, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from parts import ck_components
from frame import to_body
P = ck_components()
W = P[0] if P[0].vertices[:, 2].mean() < 0 else P[1]
BODY = P[2]
sd = -trimesh.proximity.signed_distance(BODY, W.vertices)    # + outside torso
Vb = to_body(W.vertices); Bb = to_body(BODY.vertices)
fig, axs = plt.subplots(1, 3, figsize=(22, 8))
for ax, (i, j, t) in zip(axs, [(1, 2, 'side: y cranial vs z dorsal (from +x)'), (1, 0, 'top: y vs x lateral'), (0, 2, 'front: x vs z')]):
    ax.scatter(Bb[:, i], Bb[:, j], s=1, c='0.8')
    o = np.argsort(sd)
    sc = ax.scatter(Vb[o, i], Vb[o, j], s=5, c=np.clip(sd[o], -10, 6), cmap='coolwarm', vmin=-10, vmax=6)
    ax.set_aspect('equal'); ax.grid(alpha=.3); ax.set_title(t)
plt.colorbar(sc, ax=axs[-1], label='outside torso (+)')
plt.tight_layout(); plt.savefig(sys.argv[1], dpi=65)
