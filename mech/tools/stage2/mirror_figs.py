"""Figures of both wings at the pose where they come closest (from mirror_check.json): view from the front (x-z) and from above (x-y),
the left wing as the mirror image of the right one about the body midline. Parts only (no reference body shape) -> git-safe.
usage: mirror_figs.py ID [ID ...]   -> results/stage2/<STAGE>/mirror_<ID>.png  (MIDLINE=1 for the stage-2f candidates)"""
import sys, json, numpy as np
IDS, sys.argv = sys.argv[1:], sys.argv[:1]          # analyze.py reads sys.argv on import
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from mirror_check import build, mir, MID, OUT, U, seg2

COL = dict(hum='#c0392b', uln='#c0392b', hand='#c0392b', bc1='#2471a3', rod1='#7d3c98', rod2='#1e8449', rod4='#d68910', bc2='#2471a3', rod3='#1e8449')
NAME = dict(hum='上腕骨', uln='尺骨', hand='手', bc1='ベルクランク', rod1='ロッド1（竹串）', rod2='ロッド2', rod4='ロッド4')
plt.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'sans-serif']


def capsule(ax, p, q, r, col, alpha, ls='-'):
    d = q - p; n = np.hypot(*d)
    a0 = np.arctan2(d[1], d[0]) if n > 1e-9 else 0.0
    t1 = np.linspace(a0 + np.pi / 2, a0 + 3 * np.pi / 2, 16); t2 = t1 + np.pi
    pts = np.r_[p + r * np.c_[np.cos(t1), np.sin(t1)], q + r * np.c_[np.cos(t2), np.sin(t2)]]
    ax.fill(pts[:, 0], pts[:, 1], color=col, alpha=alpha, lw=0)
    ax.plot(*np.r_[pts, pts[:1]].T, color=col, lw=0.6, ls=ls)


def main(ids):
    cands = {c['id']: c for c in json.load(open(OUT / 'candidates.json', encoding='utf-8'))}
    chk = json.load(open(OUT / 'check_2e.json', encoding='utf-8'))
    mc = json.load(open(OUT / 'mirror_check.json', encoding='utf-8'))
    for i in ids:
        c = cands[i]; caps, f = build(c, chk[i]['plan'])
        s = int(np.argmin(np.abs(U - mc[i]['u'])))
        # where do the two wings overlap at this pose
        hits = []
        for na, pa, qa, ra in caps:
            for nb, pb, qb, rb in caps:
                d, sc, tc = seg2(pa[s:s + 1], qa[s:s + 1], mir(pb[s:s + 1]), mir(qb[s:s + 1]))
                if d[0] - ra - rb < 0:
                    x1 = pa[s] + (qa[s] - pa[s]) * sc[0]; x2 = mir(pb[s:s + 1])[0] + (mir(qb[s:s + 1])[0] - mir(pb[s:s + 1])[0]) * tc[0]
                    hits.append((x1 + x2) / 2)
        fig = plt.figure(figsize=(12, 9.5)); gs = fig.add_gridspec(2, 2, height_ratios=(1, 1.25))
        axs = [fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
        views = (((0, 1), '上から見た図（横 x・前後 y）全体', None), ((0, 1), '上から見た図 中心線付近の拡大', 22),
                 ((0, 2), '前から見た図（横 x・上下 z）中心線付近の拡大', 22))
        for ax, ((a, b), ttl, zoom) in zip(axs, views):
            for side, tf, al, ls in (('右翼', lambda X: X, 0.45, '-'), ('左翼（鏡写し）', mir, 0.22, '--')):
                for nm, p, q, r in caps:
                    P, Q = tf(p[s:s + 1])[0], tf(q[s:s + 1])[0]
                    capsule(ax, P[[a, b]], Q[[a, b]], r, COL.get(nm, 'k'), al, ls)
            ax.axvline(MID, color='k', lw=1, ls=':'); ax.text(MID, 1.0, ' 体の中心線', transform=ax.get_xaxis_transform(), va='top', fontsize=9)
            for h in hits: ax.plot(h[a], h[b], 'x', color='red', ms=9, mew=2)
            ax.set_aspect('equal', adjustable='box' if zoom else 'datalim'); ax.set_title(ttl, fontsize=11); ax.grid(alpha=0.25)
            if zoom: ax.set_xlim(MID - zoom, MID + zoom)
            ax.set_xlabel('x mm（肩関節が 0）'); ax.set_ylabel(('z' if b == 2 else 'y') + ' mm')
        names = sorted({nm for nm, *_ in caps}, key=list(COL).index)
        hd = [plt.Rectangle((0, 0), 1, 1, color=COL[n], alpha=0.5) for n in names]
        fig.legend(hd, [NAME.get(n, n) for n in names], loc='lower center', ncol=len(names), fontsize=9, frameon=False)
        st = '重なりあり（赤×）' if hits else '重なりなし'
        fig.suptitle(f"{i}（案{c['dsg']}）左右の翼が最も近づく姿勢 u = {U[s]:.2f}：最小すき間 {mc[i]['min_clear']:.1f} mm、{st}。濃い実線＝右翼、薄い破線＝左翼", fontsize=11)
        fig.tight_layout(rect=(0, 0.04, 1, 0.96))
        out = OUT / f'mirror_{i}.png'; fig.savefig(out, dpi=110); plt.close(fig); print(out)


if __name__ == '__main__':
    main(IDS)
