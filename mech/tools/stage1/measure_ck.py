"""Landmarks and proportions of the Common Kingfisher model (model units; bird faces -x, y up, z lateral, midline z=0)."""
import sys, json, pathlib, numpy as np, trimesh
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from parts import ck_components
P = ck_components()
WING_R, WING_L, BODY, HEAD, TAIL, BILL = P[1], P[0], P[2], P[3], P[4], P[10]
w = WING_R.vertices
# farthest pair in the wing shell
from scipy.spatial.distance import cdist
h = WING_R.convex_hull.vertices; Dm = cdist(h, h); i, j = np.unravel_index(Dm.argmax(), Dm.shape)
a, b = h[i], h[j]
if a[1] < b[1]: a, b = b, a                       # a = front/upper end (wrist), b = tip
print('wing shell farthest pair (wrist?, tip?)', np.round(a, 1), np.round(b, 1), 'length %.1f' % Dm.max())
# body-frame axis: PCA of the torso part
c = BODY.vertices.mean(0); U, S, Vt = np.linalg.svd(BODY.vertices - c, full_matrices=False)
print('torso centroid', np.round(c, 1), 'PCA axes', np.round(Vt, 3), 'sd', np.round(S / np.sqrt(len(c) * 0 + len(BODY.vertices)), 1))
# wing outside torso
sd = trimesh.proximity.signed_distance(BODY, w)   # + inside
print('wing verts outside torso: %.0f%%, max depth outside %.1f, max inside %.1f' % (100 * np.mean(sd < 0), -sd.min(), sd.max()))
# bill length: bill tip to the point where the bill leaves the head
bt = BILL.vertices[BILL.vertices[:, 0].argmin()]
sdb = trimesh.proximity.signed_distance(HEAD, BILL.vertices)
out = BILL.vertices[sdb < 0]
print('bill tip', np.round(bt, 1), 'exposed bill (tip to farthest bill vertex at head surface) %.1f' % np.linalg.norm(out - bt, axis=1).max())
# tail: tip and exposed length outside the torso
tt = TAIL.vertices[TAIL.vertices[:, 1].argmin()]
sdt = trimesh.proximity.signed_distance(BODY, TAIL.vertices)
print('tail tip', np.round(tt, 1), 'exposed tail length %.1f' % np.linalg.norm(TAIL.vertices[sdt < 0] - tt, axis=1).max())
print('head width %.1f, torso width %.1f, wing outer z %.1f' % (HEAD.extents[2], BODY.extents[2], w[:, 2].max()))
crown = HEAD.vertices[HEAD.vertices[:, 1].argmax()]
print('crown', np.round(crown, 1), 'bill tip-tail tip straight %.1f' % np.linalg.norm(bt - tt))
