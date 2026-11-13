"""Export decimated Common Kingfisher meshes in the v4 shoulder frame for the simulator: mech/ref/derived/ck_mesh.js
Variants: as-is (scaled by wing length) and slimmed to 40 mm width. The data is an adaptation of a CC BY-SA 4.0 model (see ref/README.md)."""
import sys, json, pathlib, numpy as np, trimesh
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from ck_align import transform, aligned_parts, ROOT
from fit_ellipsoid import fit

TARGET_FACES = {'torso': 900, 'head': 500, 'tail': 250, 'bill': 160, 'face': 0, 'wing': 450, 'wing_other': 450}


def dec(m, n):
    if n <= 0: return None
    if len(m.faces) <= n: return m
    return m.simplify_quadric_decimation(face_count=n)


out = dict(source='Common Kingfisher by Alejandro Guzman (Sketchfab), CC BY-SA 4.0. Decimated and transformed: mech/tools/stage1/export_mesh_js.py. This file is CC BY-SA 4.0.',
           frame='v4 shoulder frame, mm: x lateral (right), y cranial, z dorsal; origin = right shoulder (ESTIMATE)', variants={})
for name, kw, label in [('asis', {}, 'CG（翼長で縮尺、幅58 mm）'), ('slim40', dict(kx=40 / 57.7, kz=40 / 57.7), 'CG 細身（幅40 mmに縮めた）')]:
    tr = transform(**kw); A = aligned_parts(tr)
    parts = {}
    for k, n in TARGET_FACES.items():
        m = dec(A[k], n)
        if m is None: continue
        parts[k] = dict(V=np.round(m.vertices, 2).ravel().tolist(), F=m.faces.ravel().tolist())
    P, _ = trimesh.sample.sample_surface_even(A['torso'], 20000, seed=0)
    c, r, rms = fit(P, P.mean(0), (P.max(0) - P.min(0)) / 2)
    out['variants'][name] = dict(label=label, transform=dict(scale=tr['scale'], kx=tr['kx'], kz=tr['kz'], shoulder_rule=tr['params']),
                                 occluder=dict(c=np.round(c, 2).tolist(), r=np.round(r, 2).tolist()), parts=parts)
    print(name, {k: len(v['F']) // 3 for k, v in parts.items()}, 'occluder', np.round(c, 1), np.round(r, 1))
p = ROOT / 'ref' / 'derived' / 'ck_mesh.js'
p.write_text('// ' + out['source'] + '\nwindow.CK_MESH = ' + json.dumps(out, separators=(',', ':')) + ';\n', encoding='utf-8')
print(p, p.stat().st_size // 1024, 'KB')
