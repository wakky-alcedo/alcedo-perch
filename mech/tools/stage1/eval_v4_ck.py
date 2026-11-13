"""Evaluate v4 against the Common Kingfisher mesh (aligned by ck_align.py) and draw comparison figures.
Signed distances: + outside the torso mesh, - inside (mm)."""
import sys, json, pathlib, numpy as np, trimesh, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from scipy.spatial import cKDTree
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from ck_align import transform, aligned_parts, ROOT
import v4geom

BODY_C, BODY_R = np.array([-9.0, -18.0, -6.0]), np.array([12.0, 32.0, 12.0])
LH, LU = v4geom.LH, v4geom.LU


def sd_mesh(m, X):
    return -trimesh.proximity.signed_distance(m, np.asarray(X))       # + outside


def ell_sd(X):
    return (np.linalg.norm((np.asarray(X) - BODY_C) / BODY_R, axis=-1) - 1) * 12.0


def wing_region_dist(X, q):
    """port of the wing-region part of outsideDist3 (simulator): distance outside the wing (propatagium triangle / bands), humerus frame"""
    Rh = np.array(q['Rh']); L = np.asarray(X) @ Rh          # tr3(Rh, X) = Rh^T X
    We, Te = Rh.T @ np.array(q['W']), Rh.T @ np.array(q['T'])
    out = []
    for l in L:
        if abs(l[2]) > 4:
            out.append(abs(l[2]) - 4); continue
        S, E, W, T = np.zeros(2), np.array([LH, 0.0]), We[:2], Te[:2]
        tri = [S, E, W]; sg = np.sign((E[0] - S[0]) * (W[1] - S[1]) - (E[1] - S[1]) * (W[0] - S[0])) or 1
        dmax = 0.0
        for i in range(3):
            a, b = tri[i], tri[(i + 1) % 3]; ex, ey = b - a; Ln = np.hypot(ex, ey) or 1e-9
            dmax = max(dmax, -((l[0] - a[0]) * (-ey) + (l[1] - a[1]) * ex) * sg / Ln)
        best = max(0.0, dmax)
        for o, d, Lb, ymin in ((S, E, LH, -12), (E, W, LU, -38), (W, T, v4geom.LMAN, -45)):
            ang = np.arctan2(d[1] - o[1], d[0] - o[0]); c, s = np.cos(-ang), np.sin(-ang); dx, dy = l[0] - o[0], l[1] - o[1]
            loc = (c * dx - s * dy, s * dx + c * dy)
            best = min(best, max(0.0, loc[1] - 1, ymin - loc[1], -2 - loc[0], loc[0] - Lb - 2))
        out.append(best)
    return np.array(out)


def evaluate(tr, d, draw=None, A=None):
    A = A or aligned_parts(tr); tor = A['torso']
    tgt = d['target']; steps = d['steps']; qf = steps[-1]
    res = {}
    # folded pose: bones and feathers vs torso
    B = v4geom.bones(qf, 8); Fp = v4geom.feather_points(qf, tgt)
    sB, sF = sd_mesh(tor, B), sd_mesh(tor, Fp)
    res['fold_bone_sd_min'] = float(sB.min()); res['fold_bone_sd_max'] = float(sB.max())
    res['fold_feather_sd_min'] = float(sF.min()); res['fold_feather_sd_max'] = float(sF.max())
    res['fold_feather_inside_frac(<-1mm)'] = float(np.mean(sF < -1.0))
    res['fold_feather_float_frac(>5mm)'] = float(np.mean(sF > 5.0))
    # wing tip vs the model's wing-shell tip
    tips = np.array([P[2] for k, P in v4geom.feathers(qf, tgt) if k == 'p'])
    v4tip = tips[np.argmin(tips[:, 1])]
    Wv = A['wing'].vertices; mtip = Wv[np.argmin(Wv[:, 1])]
    res['v4_tip'] = v4tip.round(1).tolist(); res['model_tip'] = mtip.round(1).tolist(); res['tip_dist'] = float(np.linalg.norm(v4tip - mtip))
    res['v4_wrist'] = np.round(qf['W'], 1).tolist(); res['model_wing_front_y'] = float(Wv[:, 1].max())
    # distance of the model's visible wing shell to v4's folded feathers
    vis = Wv[sd_mesh(tor, Wv) > 0.3]
    dvis, _ = cKDTree(Fp).query(vis)
    res['visible_shell_covered_frac(<4mm)'] = float(np.mean(dvis < 4.0))
    res['visible_shell_to_v4_median'] = float(np.median(dvis))
    # whole path: bones sweeping into the torso; mechanism protrusion with the mesh torso
    pen, prot_e, prot_m, worst = [], [], [], None
    for k, q in enumerate(steps):
        if k % 4: continue
        b = v4geom.bones(q, 6)[2:]                      # skip the proximal humerus (inside the body by design)
        pen.append(sd_mesh(tor, b).min())
        Xs = np.array(q['samples'])
        wr = wing_region_dist(Xs, q)
        sdm, sde = sd_mesh(tor, Xs), ell_sd(Xs)
        om = np.where(sdm <= -0.5, 0, np.minimum(np.maximum(0, sdm + 0.5), wr))     # simulator rule with the mesh torso
        oe = np.where(sde <= -0.5, 0, np.minimum(np.maximum(0, sde + 0.5), wr))     # simulator rule with the ellipsoid
        prot_e.append(oe.max()); prot_m.append(om.max())
        if worst is None or om.max() > worst[0]: worst = (float(om.max()), q['u'], Xs[om.argmax()].round(1).tolist())
    res['path_bone_sd_min'] = float(np.min(pen))
    res['mech_prot_ellipsoid_max'] = float(np.max(prot_e)); res['mech_prot_mesh_max'] = float(np.max(prot_m)); res['mech_prot_mesh_worst(prot,u,X)'] = worst
    if draw: figures(A, d, tr, draw)
    return res


def figures(A, d, tr, out):
    tgt = d['target']; qf = d['steps'][-1]
    tor, wing = A['torso'], A['wing']
    fig, axs = plt.subplots(2, 4, figsize=(26, 13))
    th = np.linspace(0, 2 * np.pi, 200)

    def sil(ax, m, i, j, **kw):
        ax.add_collection(PolyCollection(m.vertices[m.faces][:, :, [i, j]], **kw))
    Wv = wing.vertices; vis = sd_mesh(tor, Wv) > 0.3
    for ax, (i, j, t) in zip(axs[0, :2], [(0, 1, 'top view (x lateral, y cranial)'), (1, 2, 'side view (y cranial, z dorsal)')]):
        sil(ax, tor, i, j, facecolors='#c9d6e3', edgecolors='none')
        for nm, c in (('head', '#d9cfc0'), ('tail', '#d9d9d9')):
            sil(ax, A[nm], i, j, facecolors=c, edgecolors='none', alpha=0.6)
        ax.scatter(Wv[vis, i], Wv[vis, j], s=3, c='#2a7fd4', label='model wing shell (visible part)')
        ax.plot(BODY_C[i] + BODY_R[i] * np.cos(th), BODY_C[j] + BODY_R[j] * np.sin(th), 'k--', lw=1.5, label='v4 ellipsoid')
        for kind, P in v4geom.feathers(qf, tgt):
            ax.fill(P[:, i], P[:, j], fc='none', ec='#e0702a' if kind == 'p' else '#e0a02a', lw=0.8)
        Bn = np.array([[0, 0, 0], qf['E'], qf['W'], qf['T']]); ax.plot(Bn[:, i], Bn[:, j], 'r-o', lw=2.5, ms=4, label='v4 folded bones')
        ax.plot([0], [0], 'k+', ms=14, mew=2)
        ax.autoscale(); ax.set_aspect('equal'); ax.grid(alpha=.3); ax.set_title(t); ax.legend(loc='lower left', fontsize=8)
    ys = [10, 0, -10, -20, -30, -45]
    for ax, y in zip(list(axs[0, 2:]) + list(axs[1, :]), ys):
        for m, c, lab in ((tor, '#5b7a99', 'model torso'), (wing, '#2a7fd4', 'model wing shell'), (A['wing_other'], '#9ab', None)):
            sec = m.section(plane_origin=[0, y, 0], plane_normal=[0, 1, 0])
            if sec is None: continue
            for ent in sec.discrete:
                ax.plot(ent[:, 0], ent[:, 2], color=c, lw=1.5, label=lab); lab = None
        t2 = 1 - ((y - BODY_C[1]) / BODY_R[1]) ** 2
        if t2 > 0:
            ax.plot(BODY_C[0] + BODY_R[0] * np.sqrt(t2) * np.cos(th), BODY_C[2] + BODY_R[2] * np.sqrt(t2) * np.sin(th), 'k--', lw=1.5, label='v4 ellipsoid')
        Fp = v4geom.feather_points(qf, tgt, per=12); sel = np.abs(Fp[:, 1] - y) < 2.5
        ax.scatter(Fp[sel, 0], Fp[sel, 2], s=8, c='#e0702a', label='v4 folded feathers (±2.5 mm)')
        B = v4geom.bones(qf, 30); sel = np.abs(B[:, 1] - y) < 1.5
        ax.scatter(B[sel, 0], B[sel, 2], s=25, c='r', label='v4 folded bones')
        ax.axvline(-9, color='0.6', lw=0.8); ax.plot([0], [0], 'k+', ms=12)
        ax.set_aspect('equal'); ax.grid(alpha=.3); ax.set_title(f'section y = {y} mm (x lateral, z dorsal; grey line = midline)'); ax.legend(fontsize=7, loc='lower left')
        ax.set_xlim(-45, 30); ax.set_ylim(-42, 25)
    axs[1, 3].axis('off')
    s = tr['params']
    fig.suptitle(f"Common Kingfisher CG (x{tr['scale']:.3f}, wing length 78.8 mm) vs v4. Shoulder = origin (+), ESTIMATE: {s['SX']} mm lateral, carpal-{s['DY']} mm, {s['DZ']} mm under the back", fontsize=13)
    plt.tight_layout(); plt.savefig(out, dpi=70)


if __name__ == '__main__':
    d = v4geom.load()
    tr = transform()
    out = ROOT / 'results' / 'stage1'
    r = evaluate(tr, d, draw=out / 'ck_vs_v4.png')
    print(json.dumps(r, indent=1))
    json.dump(r, open(out / 'ck_vs_v4_metrics.json', 'w'), indent=1)
