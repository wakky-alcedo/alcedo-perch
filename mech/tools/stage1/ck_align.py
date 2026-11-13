"""Place the Common Kingfisher model in the v4 shoulder frame (mm; origin right shoulder, x lateral right, y cranial, z dorsal).

scale  : wing length. The folded-wing shell's longest chord (front end -> tip) = 78.8 mm (BTO, live adults).
         The CG has no carpal joint; the front end of the wing shell is taken as the carpal (estimate).
rotation: torso 1st principal axis = cranial, midline plane z_model = 0, bird's right = model -z.
shoulder (ESTIMATE, the model has no skeleton):
         x = SX mm lateral of the midline (v4 used 9 mm), y = carpal y - DY (v4 folded wrist is 4 mm ahead of the shoulder),
         z = dorsal surface of the torso at (x, y) minus DZ (plumage + tissue over the joint, guessed 7 mm).
"""
import sys, json, pathlib, numpy as np, trimesh
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from parts import ck_components
from frame import R_CK
ROOT = pathlib.Path(__file__).resolve().parents[2]
WING_LEN = 78.8
NAMES = {0: 'wingL', 1: 'wingR', 2: 'torso', 3: 'head', 4: 'tail', 10: 'bill', 8: 'face', 11: 'eye', 12: 'eye'}


def load_parts():
    P = ck_components()
    # bird's right wing is on model -z
    wr = 0 if P[0].vertices[:, 2].mean() < 0 else 1
    named = {'wing': P[wr], 'wing_other': P[1 - wr], 'torso': P[2], 'head': P[3], 'tail': P[4], 'bill': P[10], 'face': P[8]}
    return named, P


def chord(m):
    from scipy.spatial.distance import cdist
    h = m.vertices[m.convex_hull.vertices] if False else m.convex_hull.vertices
    D = cdist(h, h); i, j = np.unravel_index(D.argmax(), D.shape); return D.max(), h[i], h[j]


def transform(SX=9.0, DY=4.0, DZ=7.0, scale=None, kx=1.0, kz=1.0):
    """kx, kz: optional anisotropic slimming of the model about the torso centroid (lateral / dorso-ventral), applied before the shoulder rule"""
    named, P = load_parts()
    L, a, b = chord(named['wing'])
    s = WING_LEN / L if scale is None else scale
    tr0 = dict(R=R_CK.tolist(), scale=s, kx=kx, kz=kz, c=(named['torso'].vertices @ R_CK.T * s).mean(0).tolist(), shoulder_body=[0, 0, 0])
    Wb = apply(named['wing'].vertices, tr0)
    carpal_y = Wb[:, 1].max()
    T = named['torso']; Tb = apply(T.vertices, tr0)
    y_s = carpal_y - DY
    # dorsal surface of the torso at (x = SX, y = y_s): cast a ray downward from above
    tor = trimesh.Trimesh(Tb, T.faces, process=False)
    x_mid = 0.0                                          # midline in body frame is x = 0 (model z = 0)
    o = np.array([[x_mid + SX, y_s, 200.0]]); hit, _, _ = tor.ray.intersects_location(o, np.array([[0, 0, -1.0]]))
    z_top = hit[:, 2].max()
    S = np.array([x_mid + SX, y_s, z_top - DZ])
    return dict(scale=s, R=R_CK.tolist(), kx=kx, kz=kz, c=tr0['c'], shoulder_body=S.tolist(), carpal_y=float(carpal_y), wing_chord_model=float(L), z_top=float(z_top),
                params=dict(SX=SX, DY=DY, DZ=DZ, kx=kx, kz=kz))


def apply(V, tr):
    B = np.asarray(V) @ np.array(tr['R']).T * tr['scale']
    kx, kz = tr.get('kx', 1.0), tr.get('kz', 1.0)
    if kx != 1.0 or kz != 1.0:
        c = np.array(tr['c']); B = B.copy(); B[:, 0] = c[0] * 0 + (B[:, 0] - 0.0) * kx; B[:, 2] = c[2] + (B[:, 2] - c[2]) * kz   # midline x = 0
    return B - np.array(tr['shoulder_body'])


def aligned_parts(tr):
    named, P = load_parts()
    out = {}
    for k, m in named.items():
        out[k] = trimesh.Trimesh(apply(m.vertices, tr), m.faces, process=False)
    return out


if __name__ == '__main__':
    tr = transform()
    print(json.dumps(tr, indent=1))
    json.dump(tr, open(ROOT / 'results' / 'stage1' / 'ck_align.json', 'w'), indent=1)
    A = aligned_parts(tr)
    for k, m in A.items():
        print('%-11s bounds %s' % (k, np.round(m.bounds, 1).tolist()))
