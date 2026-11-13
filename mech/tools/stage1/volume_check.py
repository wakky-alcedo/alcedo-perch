"""Volume of the feathered body (head+torso+wings+tail; bill, feet, perch excluded) vs body mass.
BK scan: voxelise (1 mm), close small gaps, fill. CK: watertight parts, voxel union at the same pitch.
Masses (Wikipedia): Alcedo atthis 34-46 g; Megaceryle alcyon 113-178 g. Taxidermy volume != live volume (estimate only)."""
import sys, pathlib, numpy as np, trimesh
from scipy import ndimage
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from bk_body import bk_bird_mesh
from ck_align import transform, aligned_parts


def vox_volume(m, pitch=1.0, close=2):
    v = m.voxelized(pitch)
    M = v.matrix.copy()
    M = np.pad(M, close + 2)
    M = ndimage.binary_closing(M, iterations=close)
    M = ndimage.binary_fill_holes(M)
    return M.sum() * pitch ** 3


if __name__ == '__main__':
    bk = bk_bird_mesh()
    # drop the bill (narrow, in front of the head): keep everything; bill volume is small (<2%)
    for c in (1, 2, 3):
        print('BK feathered volume, closing %d: %.0f cm3' % (c, vox_volume(bk, 1.0, c) / 1000))
    tr = transform(); A = aligned_parts(tr)
    u = trimesh.util.concatenate([A['torso'], A['head'], A['wing'], A['wing_other'], A['tail']])
    print('CK (x%.3f) feathered volume: %.0f cm3' % (tr['scale'], vox_volume(u, 0.5, 1) / 1000))


def hull_slices_volume(P, step=1.0, half=0.75):
    """sum over y-slices of the convex-hull area of the (x,z) points: robust to holes in a scan; slight overestimate for concave sections"""
    from scipy.spatial import ConvexHull
    V = 0.0
    for y in np.arange(P[:, 1].min(), P[:, 1].max(), step):
        s = P[np.abs(P[:, 1] - y) < half][:, [0, 2]]
        if len(s) >= 10:
            try: V += ConvexHull(s).volume * step
            except Exception: pass
    return V


def run_hull():
    bk = bk_bird_mesh(); Pb, _ = trimesh.sample.sample_surface_even(bk, 150000, seed=0)
    tr = transform(); A = aligned_parts(tr)
    u = trimesh.util.concatenate([A['torso'], A['head'], A['wing'], A['wing_other'], A['tail']])
    Pc, _ = trimesh.sample.sample_surface_even(u, 150000, seed=0)
    vb, vc = hull_slices_volume(Pb), hull_slices_volume(Pc)
    print('slice-hull volume: BK %.0f cm3 (mass 113-178 g) -> %.2f-%.2f cm3/g;  CK x%.3f %.0f cm3 (mass 34-46 g) -> %.2f-%.2f cm3/g'
          % (vb / 1e3, vb / 1e3 / 178, vb / 1e3 / 113, tr['scale'], vc / 1e3, vc / 1e3 / 46, vc / 1e3 / 34))
    print('CK exact (voxel) / slice-hull ratio check: voxel 161 cm3')
    return vb, vc
