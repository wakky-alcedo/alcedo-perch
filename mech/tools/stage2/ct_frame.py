"""From the DigiMorph Alcedo cristata CT (local volume) derive, in the v4 shoulder frame (x lateral, y cranial, z dorsal, mm, scaled
to the Common Kingfisher):
  - the shoulder rule used to place the CG body: SX (shoulder lateral of the midline), DY (shoulder caudal of the carpal edge of the
    folded wing surface), DZ (shoulder below the dorsal body surface);
  - the folded bone landmarks (elbow, wrist, hand direction) as a fit target.
Body frame from the CT: lateral = left->right shoulder, cranial = spine line (thoracic vertebrae + synsacrum, dorsal-most bone near the
midline), dorsal = completes the frame. Both sides are measured; the right side of the model gets the mean (left side mirrored).
Output: results/stage2/ct_target.json"""
import json, pathlib, numpy as np
from scipy import ndimage as ndi

ROOT = pathlib.Path(__file__).resolve().parents[2]
PX, DZ_ = 0.0959, 0.234
LU_AT = 32.0                                      # Common Kingfisher ulna (model)
d = np.load(ROOT / 'results' / 'local' / 'ct_volume.npz'); V = d['V']
LM = json.load(open(ROOT / 'results' / 'stage2' / 'ct_landmarks.json'))
nz, ny, nx = V.shape
B = ndi.median_filter(V, size=(1, 3, 3)) > 140
P = lambda v: np.array(v, float)                 # landmarks are (z, y, x) in mm
SL, SR = P(LM['L']['shoulder']), P(LM['R']['shoulder'])

# --- spine line: in each slice z 44..75 mm, the dorsal-most bone pixel within +-2.5 mm of the shoulder midpoint x
xm0 = (SL[2] + SR[2]) / 2
sp = []
for k in range(int(44 / DZ_), int(75 / DZ_)):
    xs = slice(int((xm0 - 2.5) / PX), int((xm0 + 2.5) / PX))
    col = B[k, :, xs]; ys, xx = np.where(col)
    if len(ys) == 0: continue
    # vertebral body/neural spine: take the median of the bone in the dorsal 6 mm of what is there
    top = ys.min(); sel = ys < top + int(4 / PX)
    sp.append((k * DZ_, ys[sel].mean() * PX, (xx[sel].mean() + xs.start) * PX))
sp = np.array(sp); c = sp.mean(0); u_sp = np.linalg.svd(sp - c)[2][0]; u_sp = u_sp if u_sp[0] < 0 else -u_sp
# cranial axis = principal axis of the solid torso (muscle + organs), the same definition as the CG torso axis in stage 1
pts = []
for k in range(int(40 / DZ_), int(85 / DZ_), 2):
    a = ndi.median_filter(V[k].astype(float), 5); m = ndi.binary_opening(a > 80, np.ones((9, 9))); lab, n = ndi.label(m)
    if n == 0: continue
    m = lab == (np.argmax(ndi.sum(m, lab, range(1, n + 1))) + 1); y, x = np.where(m)
    pts.append(np.c_[np.full(len(y[::7]), k * DZ_), y[::7] * PX, x[::7] * PX])
Pt = np.concatenate(pts); c = Pt.mean(0); u = np.linalg.eigh(np.cov((Pt - c).T))[1][:, -1]; u = u if u[0] < 0 else -u
print('spine-line vs torso-axis angle %.1f deg' % np.degrees(np.arccos(abs(u @ u_sp))))
lat = SR - SL; lat /= np.linalg.norm(lat)        # towards the CT 'R' side
cran = u - (u @ lat) * lat; cran /= np.linalg.norm(cran)
dors = np.cross(lat, cran)
if dors[1] > 0: dors = -dors                      # dorsal = decreasing image row
print('spine pts', len(sp), 'cranial dir (z,y,x)', np.round(cran, 3), 'dorsal', np.round(dors, 3), 'lateral', np.round(lat, 3))
print('pitch of spine vs scan z axis %.1f deg, roll (shoulder height diff) %.1f deg' % (np.degrees(np.arccos(abs(u[0]))), np.degrees(np.arcsin(lat[1]))))

scale = LU_AT / np.mean([LM[s]['L_ulna'] for s in 'LR'])
mid = (SL + SR) / 2
out = {'scale': float(scale), 'sides': {}}
# body tissue mask (closed so wet feathers/skin count as body) for surfaces
tissue = None


def body_mask(k):
    a = ndi.median_filter(V[k].astype(float), 3)
    m = ndi.binary_closing(a > 35, np.ones((7, 7)))
    return ndi.binary_fill_holes(m)


def frame_coords(p, side_sign):
    q = p - mid
    return np.array([side_sign * (q @ lat), q @ cran, q @ dors])   # x lateral (towards this side), y cranial, z dorsal


for side, sgn in (('L', -1), ('R', 1)):
    r = LM[side]; S = P(r['shoulder'])
    loc = lambda p: (frame_coords(P(p), sgn) - frame_coords(S, sgn)) * scale
    # SX: lateral offset of the shoulder from the midplane
    SX = abs((S - mid) @ lat) * scale + 0.0
    SX = abs(((S - c) - ((S - c) @ cran) * cran) @ lat) * scale                    # from the torso axis
    # DZ: depth of the shoulder below the dorsal body surface, measured along -dorsal from the shoulder
    k0 = int(round(S[0] / DZ_)); depth = np.nan
    for t in np.arange(0, 20, 0.05):
        p = S + t * dors; k = int(round(p[0] / DZ_)); yy, xx = int(round(p[1] / PX)), int(round(p[2] / PX))
        if k != k0: m = body_mask(k); k0 = k
        else: m = body_mask(k) if t == 0 else m
        if not (0 <= yy < ny and 0 <= xx < nx) or not m[yy, xx]: depth = t; break
    # DY: carpal edge of the folded wing surface = the most cranial tissue point within 4 mm of the wrist (lateral/dorsal)
    W = P(r['wrist']); best = W.copy()
    for k in range(max(0, int((W[0] - 8) / DZ_)), int(W[0] / DZ_) + 1):
        m = body_mask(k); yy, xx = np.where(m)
        pts = np.c_[np.full(len(yy), k * DZ_), yy * PX, xx * PX]
        dd = np.linalg.norm(pts[:, 1:] - W[1:], axis=1); sel = dd < 3.0
        if sel.any():
            cand = pts[sel]; cy = cand @ cran
            j = np.argmax(cy)
            if cand[j] @ cran > best @ cran: best = cand[j]
    DY = ((best - S) @ cran) * scale
    E, Wr, T = loc(r['elbow']), loc(r['wrist']), loc(r['hand_tip'])
    hand = (T - Wr) / np.linalg.norm(T - Wr)
    out['sides'][side] = dict(SX=float(SX), DY=float(DY), DZ=float(depth * scale), elbow=E.tolist(), wrist=Wr.tolist(), hand_dir=hand.tolist(),
                              E_deg=r['E'], M_deg=r['M'], hum_len=r['L_hum'] * scale, ulna_len=r['L_ulna'] * scale)
    print(side, 'SX %.1f DY %.1f DZ %.1f mm (scaled x%.2f) | elbow %s wrist %s hand dir %s | E %.0f M %.0f' % (
        SX, DY, depth * scale, scale, np.round(E, 1), np.round(Wr, 1), np.round(hand, 2), r['E'], r['M']))
S_ = out['sides']
mean = lambda k: np.mean([S_[s][k] for s in 'LR'], 0)
out['rule'] = dict(SX=float(mean('SX')), DY=float(mean('DY')), DZ=float(mean('DZ')))
hd = mean('hand_dir'); hd = hd / np.linalg.norm(hd)
out['target'] = dict(elbow=mean('elbow').tolist(), wrist=mean('wrist').tolist(), hand_dir=hd.tolist(), E=float(mean('E_deg')), M=float(mean('M_deg')),
                     spread_elbow=float(np.linalg.norm(np.array(S_['L']['elbow']) - S_['R']['elbow']) / 2),
                     spread_wrist=float(np.linalg.norm(np.array(S_['L']['wrist']) - S_['R']['wrist']) / 2))
print('rule', {k: round(v, 1) for k, v in out['rule'].items()}, 'target', {k: np.round(v, 2) for k, v in out['target'].items()})
json.dump(out, open(ROOT / 'results' / 'stage2' / 'ct_target.json', 'w'), indent=1)
