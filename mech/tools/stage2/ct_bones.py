"""Trace the folded-wing long bones in the DigiMorph Alcedo cristata CT (volume from ct_volume.py, local only).
Each long-bone shaft is a hollow ring in every coronal slice: detect rings, link them slice to slice into tracks, fit a 3D line to
each track. Output: results/local/ct_tracks.json (track points) and printed track list. Coordinates in mm:
z = cranial->caudal, y = dorsal->ventral (image rows), x = image columns."""
import json, pathlib, numpy as np
from scipy import ndimage as ndi

ROOT = pathlib.Path(__file__).resolve().parents[2]
PX, DZ = 0.0959, 0.234
d = np.load(ROOT / 'results' / 'local' / 'ct_volume.npz'); V = d['V']
Z0, Z1 = int(30 / DZ), int(68 / DZ)
rings = []
for k in range(Z0, Z1):
    a = ndi.median_filter(V[k].astype(float), 3)
    b = a > 140; lab, n = ndi.label(b)
    for i in range(1, n + 1):
        c = lab == i; ar = c.sum()
        if ar < 8 or ar > 500: continue
        f = ndi.binary_fill_holes(c); fa = f.sum()
        if fa < 1.2 * ar: continue
        y, x = np.array(np.where(f)).mean(1)
        rings.append((k * DZ, y * PX, x * PX, 2 * np.sqrt(fa / np.pi) * PX))
rings = np.array(rings)
# link into tracks
tracks = []; open_ = []
for z in np.unique(rings[:, 0]):
    cur = rings[rings[:, 0] == z]; used = set(); new_open = []
    for t in open_:
        last = t[-1]
        if z - last[0] > 3 * DZ + 1e-6: tracks.append(t); continue
        dd = np.hypot(cur[:, 1] - last[1], cur[:, 2] - last[2]); dd[list(used)] = 9
        j = int(np.argmin(dd)) if len(dd) else -1
        if j >= 0 and dd[j] < 0.7: t.append(cur[j]); used.add(j); new_open.append(t)
        elif z - last[0] <= 3 * DZ: new_open.append(t)
        else: tracks.append(t)
    for j in range(len(cur)):
        if j not in used: new_open.append([cur[j]])
    open_ = new_open
tracks += open_
tracks = [np.array(t) for t in tracks if t[-1][0] - t[0][0] >= 3.0]
out = []
for i, t in enumerate(sorted(tracks, key=lambda t: t[0][0])):
    P = t[:, :3]; c = P.mean(0); u = np.linalg.svd(P - c)[2][0]; u = u if u[0] > 0 else -u
    s = (P - c) @ u; a, b = c + s.min() * u, c + s.max() * u
    out.append(dict(id=i, n=len(t), z=[float(t[0][0]), float(t[-1][0])], a=a.tolist(), b=b.tolist(), dia=float(np.median(t[:, 3])),
                    pts=t[:, :3].tolist()))
    print('%2d n %3d  z %5.1f-%5.1f  y %5.1f->%5.1f  x %5.1f->%5.1f  dia %.2f  len %.1f' % (i, len(t), a[0], b[0], a[1], b[1], a[2], b[2],
                                                                                      np.median(t[:, 3]), np.linalg.norm(b - a)))
json.dump(out, open(ROOT / 'results' / 'local' / 'ct_tracks.json', 'w'))
