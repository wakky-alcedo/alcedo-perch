"""Fit the midsagittal (symmetry) plane of the Belted Kingfisher bird (stump excluded) and build a body frame:
x = bird's right (plane normal), y = cranial (torso principal axis, in the plane), z = dorsal. Saves results/stage1/bk_frame.json."""
import sys, json, pathlib, numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from bk_load import bk_mesh, ROOT
from bk_frame import to_frame, R_BK, BILL
import trimesh

m = bk_mesh()
P, _ = trimesh.sample.sample_surface_even(m, 60000, seed=1)
F = to_frame(P)
bird = (F[:, 1] > -55) | ((F[:, 0] < -150) & (F[:, 1] > -170))   # drop stump, branch and base; keep tail + wing tips
torso = bird & (F[:, 1] > -60) & (F[:, 1] < 15) & (F[:, 0] < -25) & (F[:, 0] > -160)
Q = F[torso]; tree = cKDTree(Q)


def refl(x):
    a, b, off = x
    n = np.array([np.sin(a) * np.cos(b), np.sin(b), np.cos(a) * np.cos(b)])
    return n, off


def cost(x):
    n, off = refl(x); d = Q @ n - off; R = Q - 2 * d[:, None] * n
    dist, _ = tree.query(R); return np.mean(np.minimum(dist, 8.0))


best = None
for a0 in np.radians([-25, -12, 0, 12, 25]):
    for off0 in [Q[:, 2].mean() - 10, Q[:, 2].mean(), Q[:, 2].mean() + 10]:
        r = minimize(cost, [a0, 0.0, off0], method='Nelder-Mead', options=dict(xatol=1e-4, fatol=1e-5, maxiter=2000))
        if best is None or r.fun < best.fun: best = r
n, off = refl(best.x)
print('symmetry fit: mean mirror distance %.2f (cap 8), normal (frame) %s, offset %.1f' % (best.fun, np.round(n, 3), off))
# body frame in model coordinates
n_model = R_BK.T @ n
p0_model = BILL + R_BK.T @ (n * off)
Tm = (R_BK.T @ Q.T).T + BILL                           # torso points in model coords
Tin = Tm - ((Tm - p0_model) @ n_model)[:, None] * n_model      # projected into the plane
c = Tin.mean(0); U, S, Vt = np.linalg.svd(Tin - c, full_matrices=False)
yax = Vt[0] if Vt[0] @ (BILL - c) > 0 else -Vt[0]
xax = n_model if np.cross(n_model, yax) @ np.array([0, 1, 0]) > 0 else -n_model    # z = x × y must point up (dorsal)
zax = np.cross(xax, yax)
print('torso axis (model)', np.round(yax, 3), 'tilt from horizontal %.1f deg' % np.degrees(np.arcsin(yax[1])))
json.dump(dict(R=np.vstack([xax, yax, zax]).tolist(), origin=c.tolist(), mirror_err=best.fun), open(ROOT / 'results' / 'stage1' / 'bk_frame.json', 'w'), indent=1)
