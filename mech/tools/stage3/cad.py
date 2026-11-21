"""Stage 3 CAD of one candidate (build123d / OpenCascade): printable parts, hardware, an interference check of the real
solids over all 41 poses, part drawings with hole coordinates, a bill of materials and a 3D viewer.

usage: cad.py ID [CANDIDATES_JSON]   (environment as the candidate's stage-2 pipeline, e.g. MIDLINE=1 SWIVEL=20 DIHEDRAL=8
       STAGE_TAG=_dh8m ELBOW_MERGE=1 STACK_BALL=1 ...)  -> results/stage3/<ID>/
Hardware at 1:1 (HW below; change there): 681 bearings (1 x 3 x 1 mm) on the elbow-side pins, 1 mm steel pins, a 3 mm ball
(drilled bead / steel ball) on a 1 mm pin for every ball joint, printed sockets on the rods, 3 mm bamboo skewer for rod 1,
the wrist a resin-on-steel pin (no bearing, user 2026-11-14). The shoulder: a 2 mm shaft along the shoulder axis into a hub
under the humerus plate (motor drive not designed yet).
All dimensions are nominal (no print clearance); the drawings list them."""
import os, sys, json, pathlib, numpy as np
import build123d as bd
import trimesh
from model import Model, T, TP, R_SKEW, R_SOCK, R_SOCK_H, R_BONE, R_ROD, R_BOSS, R_PIN, GAP, U, unit

ROOT = pathlib.Path(__file__).resolve().parents[2]
CID = sys.argv[1]
OUTD = ROOT / 'results' / 'stage3' / CID
HW = dict(pin_d=1.0, brg_od=3.0, brg_id=1.0, brg_w=1.0, ball_d=2 * R_SOCK_H, sock_r=R_SOCK, sock_h=2 * R_SOCK_H, neck_cone=22.0,
          skewer_d=2 * R_SKEW, sh_d=2.0)
# which part houses the bearing of a pin joint (the pin is fixed in the other part); the wrist has no bearing (resin hole)
HOUSE = {'E': 'uln', 'G1': 'bc1', 'B2': 'rod2', 'U': 'rod2', 'W': None}
FIXED_IN = {'E': 'hum', 'G1': 'hum', 'B2': 'bc1', 'U': 'uln', 'W': 'hand'}
BALL_ON = {'B1': 'bc1', 'B4': 'hum', 'Q': 'hand', 'A': 'body'}       # the part carrying the ball's pin
COL = {'hum': '#c0504d', 'uln': '#4f81bd', 'hand': '#9bbb59', 'bc1': '#f79646', 'rod2': '#8064a2', 'rod1': '#a5865b', 'rod4': '#4bacc6',
       'hw': '#7f7f7f', 'ball': '#d9d9d9', 'body': '#bbbbbb'}
JA = {'hum': '上腕骨', 'uln': '尺骨', 'hand': '手', 'bc1': '肘レバー（旧ベルクランク1）', 'rod2': 'ロッド2', 'rod1': 'ロッド1（竹串）', 'rod4': 'ロッド4'}

m = Model(CID, sys.argv[2] if len(sys.argv) > 2 else None)
OUTD.mkdir(parents=True, exist_ok=True)
J = {nm: (a, b, X, kind) for nm, a, b, X, kind in m.joints()}
MERGED = float(np.linalg.norm(m.ep['B1'][0] - m.ep['B2'][0])) < 1e-6
Z = lambda nm: m.level_z(nm)


def loc_from(R, t):
    """build123d Location with rotation R (columns = local axes in parent) and origin t"""
    return bd.Location(bd.Plane(origin=tuple(t), x_dir=tuple(R[:, 0]), z_dir=tuple(R[:, 2])))


def cyl(p, axis, r, z0, z1):
    """cylinder of radius r along axis through p, from p + z0 axis to p + z1 axis"""
    a = unit(np.asarray(axis, float)); x = unit(np.cross(a, [1, 0, 0] if abs(a[0]) < .9 else [0, 1, 0]))
    pl = bd.Plane(origin=tuple(np.asarray(p) + z0 * a), x_dir=tuple(x), z_dir=tuple(a))
    return pl * bd.Cylinder(r, z1 - z0, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))


def plate(caps2d, z, th=TP):
    """union of capsules / discs (x, y in the part frame) extruded th around height z"""
    with bd.BuildSketch(bd.Plane.XY.offset(z - th / 2)) as sk:
        for p, q, r in caps2d:
            p, q = np.asarray(p[:2], float), np.asarray(q[:2], float); L = float(np.linalg.norm(q - p))
            if L < 1e-6:
                with bd.Locations((float(p[0]), float(p[1]))): bd.Circle(r)
            else:
                c = (p + q) / 2; ang = float(np.degrees(np.arctan2(*(q - p)[::-1])))
                with bd.Locations(bd.Location((float(c[0]), float(c[1])), ang)): bd.SlotCenterToCenter(L, 2 * r)
    return bd.extrude(sk.sketch, th)


def local_caps(nm, extra=()):
    out = [(m.to_local(nm, p[0]), m.to_local(nm, q[0]), r) for p, q, r in m.caps[nm]]
    return out + list(extra)


# ---------------------------------------------------------------- printable parts (local frames)
parts, notes = {}, {}
pinpos = lambda jn, nm: m.to_local(nm, J[jn][2][0])

# humerus: bone + brackets from the model; plate at its level; shoulder hub below the shoulder end along the shoulder axis
from stack import SH_R, SH_L, BODY_C
a_h = m.Rh[0].T @ (m.axis * np.sign(m.axis @ BODY_C))     # shoulder axis in the humerus frame, pointing into the body
X0 = a_h * Z('hum') / a_h[2]                                # where the axis crosses the humerus plate's mid-plane
hum = plate([(p, q, r) for p, q, r in local_caps('hum')] + [((0, 0, 0), X0, R_BONE)], Z('hum'))
hum = hum + cyl(X0, a_h, SH_R, 0, SH_L) - cyl(X0, a_h, HW['sh_d'] / 2, -TP, SH_L + 0.1)
parts['hum'] = hum
notes['hum'] = f"肩の軸穴 ⌀{HW['sh_d']}：板の中央面で {np.round(X0, 2).tolist()}、向き {np.round(a_h, 3).tolist()}（体の中へ）、ボス ⌀{2 * SH_R} × {SH_L} mm"

# ulna: bone E-W, lever E-U; bearing boss at E (the model's bone radius is the bearing radius: boss R_BOSS added)
parts['uln'] = plate(local_caps('uln', [(pinpos('E', 'uln'), pinpos('E', 'uln'), R_BOSS)]), Z('uln'))

# hand: plate in its own plane (normal n_hand), through W, at its level along the normal
n = m.n_hand
x_h = unit(np.cross([0, 1, 0], n)) if abs(n[1]) < .9 else unit(np.cross([1, 0, 0], n)); y_h = np.cross(n, x_h)
Rp = np.stack([x_h, y_h, n], 1)                             # hand-plane frame in hand coordinates
to_p = lambda X: Rp.T @ X
hand_caps = [(to_p(p) * [1, 1, 0], to_p(q) * [1, 1, 0], r) for p, q, r in local_caps('hand')]
parts['hand'] = loc_from(Rp, np.zeros(3)) * plate(hand_caps, Z('hand'))

# lever (bellcrank 1) and rod 2
parts['bc1'] = plate(local_caps('bc1'), Z('bc1'))
parts['rod2'] = plate(local_caps('rod2'), Z('rod2'))

# rod 4: bar between the ball centres (its level offset along the local z), sockets at the balls
L4 = float(np.linalg.norm(m.wp['Q'][0] - m.wp['B4'][0])); z4 = Z('rod4')
rod4 = plate([((0, 0, 0), (L4, 0, 0), R_ROD)], z4)
if abs(z4) > 1e-6:                                        # cranks back to the sockets
    rod4 = rod4 + cyl((0, 0, 0), (0, 0, 1), R_ROD, min(0, z4), max(0, z4)) + cyl((L4, 0, 0), (0, 0, 1), R_ROD, min(0, z4), max(0, z4))
parts['rod4'] = rod4

# rod 1: skewer between the sockets
L1 = float(np.linalg.norm(m.ep['B1'][0] - m.ep['A'][0]))
parts['rod1'] = cyl((0, 0, 0), (1, 0, 0), HW['skewer_d'] / 2, HW['sock_r'] - 0.5, L1 - HW['sock_r'] + 0.5)


# ---------------------------------------------------------------- joints: holes, pins, bearings, balls, sockets
def axis_of(jn, nm):
    """pin / stud axis in the frame of part nm"""
    if jn == 'W': return m.Rm[0].T @ m.w_world[0] if nm == 'hand' else m.frame(nm, 0)[0].T @ m.w_world[0]
    if nm == 'hand': return n
    R0 = m.frame(nm, 0)[0]; return R0.T @ m.Rh[0][:, 2]


def center_at(nm, jn):
    """part nm's plate centre at joint jn (nm's frame): the joint point moved along nm's plate normal to the plate's mid-plane"""
    p = pinpos(jn, nm); nr = n if nm == 'hand' else np.array([0, 0, 1.0])
    return p + nr * (Z(nm) - float(np.dot(p, nr)))


hw = []          # (name, owner part, solid in the owner's frame, colour key)
holes = {nm: [] for nm in parts}
for jn, (a, b, X, kind) in J.items():
    if kind == 'pin':
        fx = FIXED_IN[jn]; ot = b if fx == a else a; hs = HOUSE[jn]
        p = pinpos(jn, fx); ax = unit(axis_of(jn, fx))
        s_f = float(np.dot(center_at(fx, jn) - p, ax))
        s_o = float(np.dot(m.to_local(fx, m.to_world(ot, center_at(ot, jn), 0)) - p, ax))
        lo, hi = min(s_f, s_o) - TP / 2, max(s_f, s_o) + TP / 2
        hw.append((f'pin_{jn}', fx, cyl(p, ax, HW['pin_d'] / 2, lo, hi), 'hw'))
        holes[fx].append((jn, p, ax, HW['pin_d'], '圧入ピン'))
        if hs:
            axh = unit(axis_of(jn, hs)); c = center_at(hs, jn)
            ring = cyl(c, axh, HW['brg_od'] / 2, -HW['brg_w'] / 2, HW['brg_w'] / 2) - cyl(c, axh, HW['brg_id'] / 2, -1, 1)
            hw.append((f'brg_{jn}', hs, ring, 'ball'))
            holes[hs].append((jn, pinpos(jn, hs), axh, HW['brg_od'], '681ベアリング'))
        else:
            holes[ot].append((jn, pinpos(jn, ot), unit(axis_of(jn, ot)), HW['pin_d'] + 0.05, '樹脂穴（ピンが回る）'))
    else:
        car = BALL_ON[jn]; rod = b if car == a else a
        hw.append((f'sock_{jn}', rod, None, 'sock'))
        if car == 'body':
            hw.append((f'ball_{jn}', 'body', bd.Pos(*X[0]) * bd.Sphere(HW['ball_d'] / 2), 'ball')); continue
        p = pinpos(jn, car); ax = unit(axis_of(jn, car))
        s_pl = float(np.dot(center_at(car, jn) - p, ax)); sg = np.sign(s_pl) or 1.0
        lo, hi = sorted((-sg * HW['ball_d'] / 4, s_pl + sg * TP / 2))
        hw.append((f'pin_{jn}', car, cyl(p, ax, HW['pin_d'] / 2, lo, hi), 'hw'))
        hw.append((f'ball_{jn}', car, bd.Pos(*p) * bd.Sphere(HW['ball_d'] / 2) - cyl(p, ax, HW['pin_d'] / 2, -2, 2), 'ball'))
        holes[car].append((jn, p, ax, HW['pin_d'], f'玉ピン（板の中心から玉の中心まで {abs(s_pl):.2f} mm）'))


def fib_sphere(n):
    i = np.arange(n) + 0.5; ph = np.arccos(1 - 2 * i / n); th = np.pi * (1 + 5 ** 0.5) * i
    return np.c_[np.cos(th) * np.sin(ph), np.sin(th) * np.sin(ph), np.cos(ph)]


def minimax_axis(V, it=400):
    """unit vector with the least maximum angle to the unit vectors V (Badoiu-Clarkson on the sphere)"""
    c = unit(V.mean(0))
    for t in range(1, it + 1):
        f = V[np.argmin(V @ c)]; c = unit(c + (f - c) / (t + 1))
    return c


sock_tilt = {}
# sockets on the rods: a cylinder around the ball, its axis along the stud (pose 0), minus the ball and an entry cone for the stud
for i, (nm_, own, sol, ck) in enumerate(hw):
    if ck != 'sock': continue
    jn = nm_[5:]; a, b, X, kind = J[jn]; rod = own; pc = m.to_local(rod, X[0]); car = BALL_ON[jn]
    # socket axis = the minimax centre of the stud directions seen from the rod over the stroke, so the socket tilts both ways
    # about it by the least angle (C1: along the pose-0 stud it tilted 0..22 deg one way and its rim dipped into the lever plate)
    Rr = [m.frame(rod, k)[0] for k in range(m.N)]
    if car == 'body':
        # the body stud direction is free: the one (pointing from the ball into the body mount) with the least socket tilt,
        # kept >= 45 deg off the rod axis at every pose so the stud clears the skewer
        best = None
        for d in fib_sphere(4000):
            V = np.array([R.T @ d for R in Rr])
            if np.min(np.degrees(np.arccos(np.clip(-V[:, 0], -1, 1)))) < 45: continue      # rod x points from A away from the stud side
            c = minimax_axis(V); t = float(np.degrees(np.arccos(np.clip(V @ c, -1, 1))).max())
            if best is None or t < best[0]: best = (t, d, c)
        tilt, d_body, axl = best; side = 1.0
        notes['body'] = f"体側の玉 A {np.round(X[0], 2).tolist()}：玉ピンの向き（玉から体の取付へ、ワールド）{np.round(d_body, 3).tolist()}"
    else:
        axw = lambda k: m.Rh[k][:, 2] if car != 'hand' else m.Rm[k] @ n
        V = np.array([Rr[k].T @ axw(k) for k in range(m.N)])
        side = np.sign(np.dot(m.to_local(rod, m.to_world(car, center_at(car, jn), 0)) - pc, V.mean(0))) or 1.0   # stud from the plate side
        V = side * V; axl = minimax_axis(V); tilt = float(np.degrees(np.arccos(np.clip(V @ axl, -1, 1))).max()); side = 1.0
    cone_r = HW['pin_d'] / 2 + HW['sock_h'] / 2 * np.tan(np.radians(HW['neck_cone'])) + 0.3
    a_ = unit(side * axl); xx = unit(np.cross(a_, [1, 0, 0] if abs(a_[0]) < .9 else [0, 1, 0]))
    pl = bd.Plane(origin=tuple(pc), x_dir=tuple(xx), z_dir=tuple(a_))
    # stud-side chamfer (cone at the stroke's max tilt + 20 deg), so the tilting rim stays out of the stud's plate
    # and the stud-side face 0.4 mm lower than the ball top (the ball is held by the lip where the entry cone meets the ball,
    # r ~1.4 at h ~0.5; C1: at 13 deg the face's inner edge came within 0.13 mm of the lever plate)
    r_top = cone_r + 0.2; h_t = HW['sock_h'] / 2 - 0.4; h_c = h_t - (HW['sock_r'] - r_top) * np.tan(np.radians(min(tilt + 20, 40)))
    sk = (pl * bd.Pos(0, 0, -HW['sock_h'] / 2) * bd.Cylinder(HW['sock_r'], h_c + HW['sock_h'] / 2, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))
          + pl * bd.Pos(0, 0, h_c) * bd.Cone(HW['sock_r'], r_top, h_t - h_c, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))
          - bd.Pos(*pc) * bd.Sphere(HW['ball_d'] / 2))
    sock_tilt[jn] = round(tilt, 1)
    cone = pl * bd.Cone(HW['pin_d'] / 2 + 0.2, cone_r, HW['sock_h'] / 2 + 0.2, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))
    hw[i] = (nm_, rod, sk - cone, 'sock')
for nm_, own, sol, ck in hw:
    if ck == 'sock' and own in parts: parts[own] = parts[own] + sol
# holes into the printed parts
for nm, lst in holes.items():
    if nm not in parts: continue
    for jn, p, ax, dia, what in lst:
        parts[nm] = parts[nm] - cyl(p, ax, dia / 2, -8, 8)

# ---------------------------------------------------------------- export parts, meshes
pd = OUTD / 'parts'; pd.mkdir(exist_ok=True)
meshes = {}
for nm, s in parts.items():
    bd.export_step(s, str(pd / f'{CID}_{nm}.step')); bd.export_stl(s, str(pd / f'{CID}_{nm}.stl'), tolerance=0.01, angular_tolerance=0.2)
    meshes[nm] = trimesh.load(pd / f'{CID}_{nm}.stl')
hwm = []
for nm_, own, sol, ck in hw:
    if sol is None or ck == 'sock': continue
    f = OUTD / f'_{nm_}.stl'; bd.export_stl(sol, str(f), tolerance=0.01, angular_tolerance=0.2); hwm.append((nm_, own, trimesh.load(f), ck)); f.unlink()


# ---------------------------------------------------------------- interference over the motion
def placed(k):
    out = [(nm, nm, me.copy().apply_transform(m.M(nm, k))) for nm, me in meshes.items()]
    out += [(nm_, own, me.copy().apply_transform(m.M(own, k)), ) for nm_, own, me, ck in hwm]
    return out


def allowed(a, oa, b, ob):
    """pairs that touch by design: a joint's hardware with its two parts and with each other"""
    jn_of = lambda s: s.split('_', 1)[1] if '_' in s else None
    ja, jb = jn_of(a), jn_of(b)
    for jn, (p1, p2, X, kind) in J.items():
        grp = {p1, p2, f'pin_{jn}', f'brg_{jn}', f'ball_{jn}'}
        if jn in ('B1', 'B2') and MERGED: grp |= {'rod1', 'rod2', 'pin_B1', 'pin_B2', 'brg_B2', 'ball_B1'}   # one joint B
        if a in grp and b in grp and (ja or jb): return True     # a joint's hardware with its parts / each other (not part vs part)
    return False


MIRROR = np.diag([-1.0, 1, 1, 1]); MIRROR[0, 3] = 2 * -12.7     # left wing = mirror image about the body midline x = -12.7


def interference(steps=range(0, 41, 1), near=0.3):
    """real solids over the motion: overlaps (mm3) and gaps below `near` mm (the layer gap GAP) between parts that are not joined
    by design; and the right wing against the mirrored left wing (every pair)"""
    import manifold3d as mf
    MF = lambda me: mf.Manifold(mf.Mesh(vert_properties=np.asarray(me.vertices, np.float32), tri_verts=np.asarray(me.faces, np.uint32)))
    over, close, lr = {}, {}, {}
    put = lambda D, key, v, k, better: D.__setitem__(key, (round(float(v), 4), round(float(U[k]), 3))) if key not in D or better(v, D[key][0]) else None
    for k in steps:
        P = placed(k)
        Mf = [(a, oa, MF(me), me.bounds) for a, oa, me in P]
        for i in range(len(Mf)):
            for j in range(i + 1, len(Mf)):
                a, oa, A, ba = Mf[i]; b, ob, B, bb = Mf[j]
                if oa == ob or allowed(a, oa, b, ob): continue
                if np.any(ba[1] < bb[0] - near) or np.any(bb[1] < ba[0] - near): continue
                v = (A ^ B).volume()
                if v > 1e-3: put(over, f'{a}-{b}', v, k, lambda x, y: x > y)
                elif not ({oa, ob} == {'uln', 'hand'}):
                    g = A.min_gap(B, near)
                    if g < near - 1e-3: put(close, f'{a}-{b}', g, k, lambda x, y: x < y)
        Lm = [(a, me.copy().apply_transform(MIRROR)) for a, oa, me in P]
        for a, oa, A, ba in Mf:
            for b, me in Lm:
                bb = me.bounds
                if np.any(ba[1] < bb[0] - 1.0) or np.any(bb[1] < ba[0] - 1.0): continue
                B = MF(me); v = (A ^ B).volume()
                g = -v if v > 1e-3 else A.min_gap(B, 1.0)
                if g < 1.0 - 1e-3: put(lr, f'R:{a}-L:{b}', g, k, lambda x, y: x < y)
    return dict(overlap_mm3=over, gap_below_mm=close, left_right_gap_mm=lr)


# ---------------------------------------------------------------- feather bending jigs (stage 2f choice, feather_bend_<ID>_chosen.json)
def jigs():
    from kin import PRIM, SEC
    fb = None
    for p_ in sorted((ROOT / 'results' / 'stage2').glob(f'*/feather_bend_{CID}_chosen.json')) + sorted((ROOT / 'results' / 'stage2').glob('*/feather_bend_chosen.json')):
        d_ = json.load(open(p_, encoding='utf-8'))
        if CID in d_: fb = d_[CID]
    if fb is None: return {}
    out = {}
    for grp, tab in (('pri', PRIM), ('sec', SEC)):
        a = np.radians(fb[grp]['alpha_deg']); R = fb[grp]['R_mm'] or 1e9; L = max(t[1] for t in tab) + 3
        r = np.linspace(0, L, 120); z = r * np.tan(a) - r ** 2 / (2 * R)
        base = z.min() - 3.0
        pts = [(0.0, base)] + [(float(ri), float(zi)) for ri, zi in zip(r, z)] + [(float(L), base)]
        face = bd.Plane.XZ * bd.make_face(bd.Polyline(*pts, close=True))
        jig = bd.extrude(face, 8)
        f = OUTD / 'parts' / f'{CID}_jig_{grp}.stl'; bd.export_stl(jig, str(f)); bd.export_step(jig, str(f.with_suffix('.step')))
        out[grp] = dict(alpha_deg=fb[grp]['alpha_deg'], R_mm=fb[grp]['R_mm'], length_mm=round(L, 1), camber=fb[grp]['camber'])
    return out


# ---------------------------------------------------------------- drawings (plan of every plate part, holes with coordinates)
def drawings():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'sans-serif']
    names = [nm for nm in ('hum', 'uln', 'hand', 'bc1', 'rod2', 'rod4') if nm in meshes]
    fig, axs = plt.subplots(len(names), 1, figsize=(10, 3.3 * len(names)), dpi=110)
    rows = []
    for ax_, nm in zip(axs, names):
        me = meshes[nm]
        if nm == 'hand': me = me.copy().apply_transform(np.linalg.inv(np.block([[Rp, np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]])))
        sec = me.section(plane_origin=me.centroid * [0, 0, 1], plane_normal=[0, 0, 1])
        if sec is not None:
            for pth in sec.discrete: ax_.plot(pth[:, 0], pth[:, 1], color=COL[nm], lw=1.2)
        else:
            v = me.vertices; ax_.scatter(v[:, 0], v[:, 1], s=.1, color=COL[nm])
        for jn, p, axv, dia, what in holes.get(nm, []):
            pp = Rp.T @ p if nm == 'hand' else p
            ax_.add_patch(plt.Circle(pp[:2], dia / 2, fill=False, color='k', lw=.8))
            ax_.annotate(f'{jn} ({pp[0]:.2f}, {pp[1]:.2f}) ⌀{dia:g}', pp[:2], xytext=(4, 6), textcoords='offset points', fontsize=7)
            tilt = float(np.degrees(np.arccos(abs((Rp.T @ axv if nm == 'hand' else axv)[2]))))
            rows.append((JA.get(nm, nm), jn, round(float(pp[0]), 3), round(float(pp[1]), 3), dia, round(tilt, 1), what))
        ax_.set_aspect('equal'); ax_.grid(alpha=.3)
        th = f'{TP:.1f}' if nm != 'rod4' else f'{TP:.1f}'
        ax_.set_title(f'{JA.get(nm, nm)}  板厚 {th} mm、層 {m.plan.get(nm, 0):+d}（{Z(nm):+.1f} mm）', fontsize=9, loc='left')
    fig.tight_layout(); fig.savefig(OUTD / f'{CID}_drawings.png'); plt.close(fig)
    return rows


# ---------------------------------------------------------------- 3D viewer (three.js), both wings, slider over the motion
def viewer():
    def pack(me):
        return dict(v=np.round(np.asarray(me.vertices), 3).ravel().tolist(), f=np.asarray(me.faces).ravel().tolist())
    items = [dict(name=nm, owner=nm, col=COL[nm], mesh=pack(me)) for nm, me in meshes.items()]
    items += [dict(name=nm_, owner=own, col=COL['ball' if ck == 'ball' else 'hw'], mesh=pack(me)) for nm_, own, me, ck in hwm if own != 'body']
    owners = sorted({it['owner'] for it in items})
    mats = {o: [np.round(m.M(o, k), 5).T.ravel().tolist() for k in range(m.N)] for o in owners}
    body_pts = dict(A=np.round(m.ep['A'][0], 3).tolist(), S=[0, 0, 0], axis=np.round(m.axis * np.sign(m.axis @ BODY_C), 4).tolist())
    data = json.dumps(dict(items=items, mats=mats, N=m.N, mid=-12.7, body=body_pts))
    html = (pathlib.Path(__file__).with_name('viewer_template.html')).read_text(encoding='utf-8')
    (OUTD / f'{CID}_viewer.html').write_text(html.replace('/*DATA*/null', data).replace('__TITLE__', f'{CID} CAD'), encoding='utf-8')

if __name__ == '__main__':
    inter = interference()
    print('interference:', json.dumps(inter, ensure_ascii=False))
    # summary
    info = dict(id=CID, plan=m.plan, studs=m.studs, F=np.round(m.F, 3).tolist(), feasible=bool(np.all(m.G <= 0)),
                consistency_mm={k: round(v, 5) for k, v in m.consistency().items()}, interference=inter,
                rods=dict(rod1_ball_to_ball=round(L1, 3), rod4_ball_to_ball=round(L4, 3),
                          rod2_pin_to_pin=round(float(np.linalg.norm(m.ep['U'][0] - m.ep['B2'][0])), 3)),
                holes={nm: [dict(joint=jn, at=np.round(p, 3).tolist(), axis=np.round(ax, 4).tolist(), dia=dia, what=w) for jn, p, ax, dia, w in lst]
                       for nm, lst in holes.items()}, notes=notes, hw=HW, socket_tilt_deg=sock_tilt)
    info['jigs'] = jigs(); rows = drawings(); viewer()
    info['hole_table'] = rows
    json.dump(info, open(OUTD / 'cad.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(json.dumps(info['rods']), info['consistency_mm'])
