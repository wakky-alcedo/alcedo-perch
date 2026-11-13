"""My interpretation of the folded wing in the DigiMorph Alcedo cristata CT, drawn on three projections for checking.
Bone axes: humerus and ulna from the traced shafts (ct_bones.py); bone ends found by walking along the axis while bone is present;
hand (carpometacarpus + digit, not hollow) from bone pieces ventral to the ulna. Output: results/local/ct_interpretation.png and
results/stage2/ct_landmarks.json (numbers only)."""
import json, pathlib, numpy as np
from scipy import ndimage as ndi
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[2]
PX, DZ = 0.0959, 0.234
d = np.load(ROOT / 'results' / 'local' / 'ct_volume.npz'); V = d['V']
B = ndi.median_filter(V, size=(1, 3, 3)) > 140
T = {t['id']: t for t in json.load(open(ROOT / 'results' / 'local' / 'ct_tracks.json'))}
nz, ny, nx = V.shape
XM = 19.5                                        # body midline (x, mm) from the skull/spine


def bone_at(p, r=0.5):
    k = int(round(p[0] / DZ)); y, x = p[1] / PX, p[2] / PX; rr = int(np.ceil(r / PX))
    if not (0 <= k < nz): return False
    sub = B[k, max(0, int(y) - rr):int(y) + rr + 1, max(0, int(x) - rr):int(x) + rr + 1]
    return sub.any()


def axis(t):
    P = np.array(t['pts']); c = P.mean(0); u = np.linalg.svd(P - c)[2][0]
    return c, (u if u[0] > 0 else -u), P


def walk(c, u, s0, sgn, maxs=12):
    s = s0
    while abs(s - s0) < maxs and bone_at(c + (s + sgn * 0.25) * u): s += sgn * 0.25
    return c + s * u


def hand_points(ulna_c, ulna_u, side):
    pts = []
    for k in range(int(39 / DZ), int(60 / DZ)):
        lab, n = ndi.label(B[k])
        if n == 0: continue
        cm = ndi.center_of_mass(B[k], lab, range(1, n + 1)); ar = ndi.sum(B[k], lab, range(1, n + 1))
        z = k * DZ; s = (z - ulna_c[0]) / ulna_u[0]; q = ulna_c + s * ulna_u        # ulna axis at this slice
        for (y, x), a in zip(cm, ar):
            y, x = y * PX, x * PX
            if a < 3 or a > 120: continue
            if y - q[1] < 1.5 or y - q[1] > 9: continue                                 # ventral to the ulna, inside the wing stack
            if side * (x - q[2]) < -1.5 or abs(x - XM) < abs(q[2] - XM) - 4: continue   # not medial into the body
            if abs(x - XM) < 9: continue
            pts.append((z, y, x))
    return np.array(pts)


res = {}
for side, hid, uid in (('L', 3, 0), ('R', 5, 1)):
    sg = -1 if side == 'L' else 1
    ch, uh, Ph = axis(T[hid]); cu, uu, Pu = axis(T[uid])
    sh = (Ph - ch) @ uh; su = (Pu - cu) @ uu
    shoulder = walk(ch, uh, sh.min(), -1); elbow_h = walk(ch, uh, sh.max(), +1)
    wrist = walk(cu, uu, su.min(), -1); elbow_u = walk(cu, uu, su.max(), +1)
    elbow = (elbow_h + elbow_u) / 2
    H = hand_points(cu, uu, sg)
    # hand axis: line fit through the hand pieces, starting at the wrist
    if len(H) >= 5:
        ch2 = H.mean(0); u2 = np.linalg.svd(H - ch2)[2][0]; u2 = u2 if u2[0] > 0 else -u2
        tip = ch2 + ((H - ch2) @ u2).max() * u2
    else:
        tip = np.full(3, np.nan)
    ang = lambda a, o, b: float(np.degrees(np.arccos(np.clip(np.dot((a - o) / np.linalg.norm(a - o), (b - o) / np.linalg.norm(b - o)), -1, 1))))
    res[side] = dict(shoulder=shoulder.tolist(), elbow=elbow.tolist(), wrist=wrist.tolist(), hand_tip=tip.tolist(), hand_pts=H.tolist(),
                     hum_pts=Ph.tolist(), ulna_pts=Pu.tolist(), elbow_gap=float(np.linalg.norm(elbow_h - elbow_u)),
                     L_hum=float(np.linalg.norm(elbow_h - shoulder)), L_ulna=float(np.linalg.norm(elbow_u - wrist)), L_hand=float(np.linalg.norm(tip - wrist)),
                     E=ang(shoulder, elbow, wrist), M=ang(elbow, wrist, tip))
    r = res[side]
    print(side, 'shoulder', np.round(shoulder, 1), 'elbow', np.round(elbow, 1), '(gap %.1f)' % r['elbow_gap'], 'wrist', np.round(wrist, 1), 'tip', np.round(tip, 1))
    print('   lengths humerus %.1f ulna %.1f hand %.1f mm | elbow angle %.0f deg, wrist angle %.0f deg | hand pieces %d' % (
        r['L_hum'], r['L_ulna'], r['L_hand'], r['E'], r['M'], len(H)))

# ---- figure: 3 projections + 2 coronal slices, with the interpretation drawn on top
COL = dict(hum='#ff5050', ulna='#40c0ff', hand='#ffd040')
fig = plt.figure(figsize=(24, 17))
gs = fig.add_gridspec(2, 3)
views = [(fig.add_subplot(gs[0, :2]), 'top', 'dorsal view (dorsal 15 mm only, legs/belly hidden)'),
         (fig.add_subplot(gs[1, :2]), 'side', 'side view of the left half (x < 12 mm)'),
         (fig.add_subplot(gs[0, 2]), 'front', 'front view (slab z 36-64 mm)')]
for ax, kind, ttl in views:
    if kind == 'top':
        img = V[:, :int(15 / PX), :].max(1).T; ext = [0, nz * DZ, nx * PX, 0]; ij = (0, 2)
    elif kind == 'side':
        img = V[:, :, :int(12 / PX)].max(2).T; ext = [0, nz * DZ, ny * PX, 0]; ij = (0, 1)
    else:
        img = V[int(36 / DZ):int(64 / DZ)].max(0); ext = [0, nx * PX, ny * PX, 0]; ij = (2, 1)
    ax.imshow(img, cmap='gray', extent=ext, vmin=50, vmax=230)
    for side, r in res.items():
        if kind == 'side' and side == 'R': continue
        for key, c in (('hum_pts', COL['hum']), ('ulna_pts', COL['ulna']), ('hand_pts', COL['hand'])):
            P = np.array(r[key])
            if len(P): ax.scatter(P[:, ij[0]], P[:, ij[1]], s=4, c=c, alpha=.6)
        J = np.array([r['shoulder'], r['elbow'], r['wrist'], r['hand_tip']])
        ax.plot(J[:2, ij[0]], J[:2, ij[1]], '-', c=COL['hum'], lw=2.5); ax.plot(J[1:3, ij[0]], J[1:3, ij[1]], '-', c=COL['ulna'], lw=2.5)
        ax.plot(J[2:, ij[0]], J[2:, ij[1]], '-', c=COL['hand'], lw=2.5)
        for p, nm in zip(J, ('S', 'E', 'W', 'T')):
            ax.plot(p[ij[0]], p[ij[1]], 'o', mfc='none', mec='w', ms=11, mew=2)
            ax.text(p[ij[0]] + 0.6, p[ij[1]] - 0.6, nm + side, color='w', fontsize=12, weight='bold')
    if kind == 'front': ax.set_xlim(0, nx * PX); ax.set_ylim(22, 0)
    if kind == 'top': ax.set_xlim(25, 80)
    if kind == 'side': ax.set_xlim(25, 95); ax.set_ylim(36, 0)
    ax.set_title(ttl); ax.grid(alpha=.25, color='y')
for ax, zz in ((fig.add_subplot(gs[1, 2]), 50),):
    k = int(zz / DZ); ax.imshow(V[k], cmap='gray', extent=[0, nx * PX, ny * PX, 0], vmax=220); ax.set_ylim(25, 0)
    for side, r in res.items():
        for key, c in (('hum_pts', COL['hum']), ('ulna_pts', COL['ulna']), ('hand_pts', COL['hand'])):
            P = np.array(r[key]); P = P[np.abs(P[:, 0] - zz) < 0.15] if len(P) else P
            if len(P): ax.scatter(P[:, 2], P[:, 1], s=60, facecolors='none', edgecolors=c, linewidths=2)
    ax.set_title(f'coronal slice z = {zz} mm (circles: traced bones)')
fig.text(0.01, 0.005, 'red: humerus (S shoulder - E elbow), blue: ulna/forearm (E - W wrist), yellow: hand = carpometacarpus + digit (W - T tip). '
         'z: cranial->caudal, y: dorsal->ventral, x: image columns (L = bird side at small x). DigiMorph Alcedo cristata FMNH 319990.', fontsize=12)
fig.tight_layout(rect=(0, 0.02, 1, 1)); fig.savefig(ROOT / 'results' / 'local' / 'ct_interpretation.png', dpi=60)
json.dump({s: {k: v for k, v in r.items() if not k.endswith('_pts')} for s, r in res.items()},
          open(ROOT / 'results' / 'stage2' / 'ct_landmarks.json', 'w'), indent=1)
