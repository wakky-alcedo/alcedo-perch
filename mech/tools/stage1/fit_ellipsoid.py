"""Axis-aligned ellipsoid fit to the Common Kingfisher outline (torso + folded wings = what the folded wing must rest on / be flush with),
in the v4 shoulder frame, for the as-is model and for slimmed variants. Prints a proposal to replace v4's torso ellipsoid."""
import sys, json, pathlib, numpy as np, trimesh
from scipy.optimize import least_squares
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from ck_align import transform, aligned_parts, ROOT


def fit(P, c0, r0):
    def res(x):
        c, r = x[:3], x[3:]
        return np.linalg.norm((P - c) / r, axis=1) - 1
    s = least_squares(res, np.r_[c0, r0], bounds=([-np.inf] * 3 + [1] * 3, [np.inf] * 3 + [200] * 3))
    c, r = s.x[:3], s.x[3:]
    rms = np.sqrt(np.mean((res(s.x) * r.mean()) ** 2))
    return c, r, rms


if __name__ == '__main__':
    out = {}
    for name, kw in [('as_is', {}), ('slim40', dict(kx=40 / 57.7, kz=40 / 57.7)), ('slim34', dict(kx=34 / 57.7, kz=34 / 57.7))]:
        tr = transform(**kw); A = aligned_parts(tr)
        for part, ms in [('torso', [A['torso']]), ('torso+wings', [A['torso'], A['wing'], A['wing_other']])]:
            m = trimesh.util.concatenate(ms)
            P, _ = trimesh.sample.sample_surface_even(m, 30000, seed=0)
            # keep the outer surface only (drop points inside the union: buried parts of the wing shells)
            if len(ms) > 1:
                inside = np.zeros(len(P), bool)
                for mm in ms: inside |= mm.contains(P) & (trimesh.proximity.signed_distance(mm, P) > 0.2)
                P = P[~inside]
            c, r, rms = fit(P, P.mean(0), (P.max(0) - P.min(0)) / 2)
            out[f'{name}/{part}'] = dict(c=np.round(c, 1).tolist(), r=np.round(r, 1).tolist(), rms=round(float(rms), 2))
            print(f'{name:7s} {part:12s} centre {np.round(c, 1)}  radii {np.round(r, 1)}  rms {rms:.2f} mm')
    print('v4 ellipsoid: centre [-9, -18, -6] radii [12, 32, 12]')
    json.dump(out, open(ROOT / 'results' / 'stage1' / 'ellipsoid_fits.json', 'w'), indent=1)
