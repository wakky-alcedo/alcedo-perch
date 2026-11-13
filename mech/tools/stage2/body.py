"""Body model for stage 2: signed distance field (SDF) of the Common Kingfisher CG body (torso + head + tail), slimmed to a given
feathered torso width, in the v4 shoulder frame (shoulder by the stage-1 rule). Plus the target folded-wing surface (visible part of
the CG wing shell) and the CG wing tip.

Cached in mech/results/local/ (derived from a CC BY-SA model -> not in git)."""
import sys, pathlib, numpy as np
from scipy import ndimage
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'stage1'))
ROOT = HERE.parents[1]
LOCAL = ROOT / 'results' / 'local'
CK_WIDTH = 57.7          # feathered torso width of the CG as scaled by wing length (stage 1)
PITCH = 0.5
LO, HI = np.array([-60.0, -110.0, -55.0]), np.array([45.0, 50.0, 40.0])


def build(width):
    import trimesh
    from ck_align import transform, aligned_parts
    k = width / CK_WIDTH
    tr = transform(kx=k, kz=k); A = aligned_parts(tr)
    shape = np.ceil((HI - LO) / PITCH).astype(int) + 1
    inside = np.zeros(shape, bool)
    for name in ('torso', 'head', 'tail'):
        m = A[name]; v = m.voxelized(PITCH).fill()
        pts = v.points                                     # centres of filled voxels
        idx = np.round((pts - LO) / PITCH).astype(int)
        ok = np.all((idx >= 0) & (idx < shape), 1); idx = idx[ok]
        inside[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    inside = ndimage.binary_closing(inside, iterations=1)
    d_out = ndimage.distance_transform_edt(~inside) * PITCH
    d_in = ndimage.distance_transform_edt(inside) * PITCH
    sdf = (d_out - d_in).astype(np.float32)               # + outside
    # torso-only SDF for the "folded wing rests on the torso" test
    ins_t = np.zeros(shape, bool); v = A['torso'].voxelized(PITCH).fill(); idx = np.round((v.points - LO) / PITCH).astype(int)
    ok = np.all((idx >= 0) & (idx < shape), 1); idx = idx[ok]; ins_t[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    sdf_t = (ndimage.distance_transform_edt(~ins_t) - ndimage.distance_transform_edt(ins_t)).astype(np.float32) * PITCH
    # target folded wing: visible part of the CG right wing shell (outside the torso by > 0.3 mm)
    W = A['wing']; c = W.triangles_center
    s = trimesh.proximity.signed_distance(A['torso'], c)
    vis = c[s < -0.3]
    tip = W.vertices[np.argmin(W.vertices[:, 1])]
    np.savez_compressed(LOCAL / f'body_w{int(width)}.npz', sdf=sdf, sdf_t=sdf_t, lo=LO, pitch=PITCH, vis=vis, tip=tip, width=width,
                        shoulder_body=np.array(tr['shoulder_body']))
    print(f'width {width}: grid {shape}, inside voxels {inside.sum()}, visible wing pts {len(vis)}, CG tip {np.round(tip, 1)}')


class Body:
    def __init__(self, width=45):
        d = np.load(LOCAL / f'body_w{int(width)}.npz')
        self.sdf, self.sdf_t, self.lo, self.pitch = d['sdf'], d['sdf_t'], d['lo'], float(d['pitch'])
        self.vis, self.tip, self.width = d['vis'], d['tip'], float(d['width'])
        self.shape = np.array(self.sdf.shape)

    def _interp(self, G, X):
        X = np.asarray(X, float); sh = X.shape[:-1]; X = X.reshape(-1, 3)
        f = (X - self.lo) / self.pitch
        out = ndimage.map_coordinates(G, f.T, order=1, mode='nearest')
        # outside the grid: distance to the box as a lower bound + the boundary value
        over = np.maximum(0, np.maximum(-f, f - (self.shape - 1))).max(1) * self.pitch
        return (out + over).reshape(sh)

    def sd(self, X):          # torso + head + tail, + outside (mm)
        return self._interp(self.sdf, X)

    def sd_torso(self, X):
        return self._interp(self.sdf_t, X)


if __name__ == '__main__':
    LOCAL.mkdir(parents=True, exist_ok=True)
    for w in (sys.argv[1:] or ['40', '45', '50']):
        build(float(w))
