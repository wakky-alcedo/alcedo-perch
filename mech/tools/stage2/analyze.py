"""Merge the NSGA-III runs, report seed-to-seed variation, pick candidates, verify them (Monte Carlo, body width 40/50),
and write tables + figures.  usage: analyze.py [width] [tilt] [2d]
  tilt: design B, runs_tilt/ -> results/stage2/tilt/;  tilt 2d: design B with link interference + per-joint play, runs_2d/ -> results/stage2/2d/
Outputs: results/stage2/summary.json, candidates.json, seeds.csv, pareto.png, seeds.png (no CG content -> git);
         results/local/cand_*.png (CG body drawn -> not in git)."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
import sys, json, glob, pathlib, collections, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from problem import Evaluator, U, SIG_DIM, CLEAR
from mech import *

ROOT = pathlib.Path(__file__).resolve().parents[2]
TILT = 'tilt' in sys.argv; S2D = '2d' in sys.argv; S2E = '2e' in sys.argv; sys.argv = [v for v in sys.argv if v not in ('tilt', '2d', '2e')]
TILT = TILT or S2E
from problem import STAGE
RUNS = ROOT / 'results' / 'stage2' / (f'runs_{STAGE}' if S2E else 'runs_2d' if S2D else 'runs_tilt' if TILT else 'runs')
OUT = ROOT / 'results' / 'stage2' / (STAGE if S2E else '2d' if S2D else 'tilt' if TILT else ''); OUT.mkdir(exist_ok=True); LOCAL = ROOT / 'results' / 'local'
PREFIX = ('F' if STAGE == '2f' else 'E') if S2E else 'D' if S2D else 'B' if TILT else 'c'
WIDTH = int(sys.argv[1]) if len(sys.argv) > 1 else 45
OBJ = ['joint error (deg)', 'min transmission (deg)', 'protrusion (mm)', 'sensitivity 2σ (deg)', 'fold fit', 'max part (mm)']
CFG_JA = {('body', 'ulna'): '肘BC:体（スティーブンソン型・空間）／手首BC:尺骨', ('hum', 'ulna'): '肘BC:上腕骨（ワット型）／手首BC:尺骨',
          ('ulna', 'ulna'): '肘BC:尺骨／手首BC:尺骨', ('body', 'hum'): '肘BC:体／手首BC:上腕骨', ('hum', 'hum'): '肘BC:上腕骨／手首BC:上腕骨',
          ('ulna', 'hum'): '肘BC:尺骨／手首BC:上腕骨', ('hum', 'direct'): '肘BC:上腕骨／手首:直結ロッド', ('ulna', 'direct'): '肘BC:尺骨／手首:直結ロッド',
          ('direct', 'hum'): '肘:直結ロッド／手首BC:上腕骨', ('direct', 'direct'): '肘・手首とも直結ロッド（ベルクランクなし）'}
# goal levels for picking a balanced compromise (Chebyshev): value at which each term reaches 1
GOAL = dict(err=(0, 2), tmin=(45, 15), prot=(0, 1), sens=(0, 3), fit=(8, 4), part=(10, 6))


def cheb(F):
    z = np.c_[F[:, 0] / 2, (45 + F[:, 1]) / 15, F[:, 2] / 1, F[:, 3] / 3, (F[:, 4] - 8) / 4, (F[:, 5] - 10) / 6]
    return np.max(z, 1), z


def load():
    runs = []
    for f in sorted(glob.glob(str(RUNS / f'*_w{WIDTH}_s*.npz'))):
        name = pathlib.Path(f).stem; dsg = ''
        if S2E: dsg, name = name.split('-', 1)
        ec, wc, w, s = name.split('_')
        d = np.load(f)
        runs.append(dict(ec=ec, wc=wc, dsg=dsg, seed=int(s[1:]), X=d['X'], F=d['F'], G=d['G'], log=d['log'], time=float(d['time'])))
    return runs


def make_ev(ec, wc, width, dsg=''):
    """evaluator of a configuration (stage 2e: design A planar / B tilted, with the real-height layer plan)"""
    if dsg: return Evaluator(ec, wc, width, tilt='A' if dsg == 'A' else True, stack=True)
    return Evaluator(ec, wc, width, tilt=TILT)


cname = lambda c: (c[2] + ':' if len(c) > 2 and c[2] else '') + f'{c[0]}/{c[1]}'
clabel = lambda c: (('案A（平面）' if c[2] == 'A' else '案B（手首傾き）') + ' ' if len(c) > 2 and c[2] else '') + CFG_JA[(c[0], c[1])]


def monte_carlo(ev, x, n=400, amp=0.05, play=1.0, seed=0):
    """nonlinear MC: every manufactured dimension uniform +-amp, every rod length additionally +-play*ev.play (joint clearances of its
    two ends and of the main hinges it crosses). Max joint-angle deviation per sample."""
    F0, G0, ex = ev(x[None], extra=True); d, f = ex['d'], ex['f']
    flat = flatten(d, ev.ec, ev.tilt); D_ = flat.shape[1]
    rng = np.random.default_rng(seed)
    pert = rng.uniform(-amp, amp, (n, D_))
    names = [k for k, m in ev.dims for _ in range(m)]
    for i, k in enumerate(names):
        if k in RODS: pert[:, i] += rng.uniform(-play * ev.play[k], play * ev.play[k], n)
    rep = lambda a: np.repeat(a, n, 0)
    dd = {k: (rep(v) if isinstance(v, np.ndarray) and v.shape[:1] == (1,) else v) for k, v in d.items()}
    dd = unflatten(np.repeat(flat, n, 0) + pert, ev.ec, dd, ev.tilt)
    ref = dict(pe=rep(f['pe']), psi=rep(f['psi']), b3=rep(f['b3']))
    if f.get('b1') is not None: ref['b1'] = rep(f['b1'])
    with np.errstate(all='ignore'):
        g = forward(dd, ref, U)
    dev = np.maximum(np.abs(g['E'] - f['E']).max(1), np.abs(g['M'] - f['M']).max(1))
    fail = (np.min([m.min(1) for m in g['margins']], 0) <= 0) | ~np.isfinite(dev)
    dev = np.where(fail, np.nan, dev)
    return dict(p50=float(np.nanpercentile(dev, 50)), p95=float(np.nanpercentile(dev, 95)), max=float(np.nanmax(dev)), fail=float(fail.mean()))


def describe(ev, x):
    F, G, e = ev(x[None], extra=True)
    d = e['d']
    r = dict(F=F[0].round(3).tolist(), G=G[0].round(3).tolist(), feasible=bool(np.all(G <= 0)),
             E=[float(e['f']['E'][0, 0]), float(e['f']['E'][0, -1])], M=[float(e['f']['M'][0, 0]), float(e['f']['M'][0, -1])],
             trans=e['trans'][0].round(1).tolist(), margins_mm=e['margins'][0].round(2).tolist(), swivel=float(e['swivel'][0]),
             lever_min=float(e['lev'][0]), part_max=float(e['part'][0]), rods=e['rods'][:, 0].round(2).tolist(),
             rodforce_N_per_Nmm=float(e['rodforce'][0]), gain_max=float(e['gain'][0]), worst_lin_005=float(e['worst005'][0]), sweep=float(e['sweep'][0]),
             fit_parts=e['fitx']['parts'][0].round(2).tolist(), tip_off=float(e['fitx']['tipd'][0]), theta=float(np.degrees(d['th'][0])),
             axis=d['axis'][0].round(4).tolist(), E1=float(d['E1'][0]), M1=float(d['M1'][0]), fz=float(d['fz'][0]),
             res=[float(e['res_e'][0]), float(e['res_w'][0])], ct_pos=float(e['fitx']['ct_pos'][0]), ct_hand=float(e['fitx']['ct_hand'][0]), tilt=bool(d['tilt']))
    if d['tilt']: r.update(tw=float(np.degrees(d['tw'][0])), phi=float(np.degrees(d['phi'][0])), dl=float(d['dl'][0]))
    if 'layers' in e: r.update(layers=int(e['layers'][0]), layers_wing=int(e['layers_wing'][0]),
                               clash={f'{a}-{b}': round(float(v[0]), 2) for (a, b), v in e['layx']['pairs'].items() if v[0] < 1.0})
    if e.get('stack_plans') is not None: r.update(stack_unplaced=int(G[0, -1]), stack_plan=e['stack_plans'][0])
    return r, e


def main():
    runs = load()
    print('runs', len(runs))
    cfgs = sorted({(r['ec'], r['wc'], r['dsg']) for r in runs}, key=lambda c: (c[2], list(CFG_JA).index(c[:2])))
    seeds_rows = []; allX, allF, allC = [], [], []
    for c in cfgs:
        for r in [r for r in runs if (r['ec'], r['wc'], r['dsg']) == c]:
            feas = np.all(r['G'] <= 0, 1)
            row = dict(ec=c[0], wc=c[1], dsg=c[2], seed=r['seed'], feasible=int(feas.sum()), time_s=round(r['time']))
            if feas.any():
                F = r['F'][feas]; s, _ = cheb(F)
                row.update({f'best_{k}': float(v) for k, v in zip(['err', 'negtmin', 'prot', 'sens', 'fit', 'part'], F.min(0))}, best_cheb=float(s.min()))
                allX.append(r['X'][feas]); allF.append(F); allC += [c] * feas.sum()
            seeds_rows.append(row)
    X = [x for a in allX for x in a]; F = np.concatenate(allF); C = np.array([cname(c) for c in allC])
    nd = NonDominatedSorting().do(F, only_non_dominated_front=True)
    print('feasible', len(F), 'global non-dominated', len(nd))
    # seed table
    import csv
    keys = ['dsg', 'ec', 'wc', 'seed', 'feasible', 'best_err', 'best_negtmin', 'best_prot', 'best_sens', 'best_fit', 'best_part', 'best_cheb', 'time_s']
    with open(OUT / 'seeds.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, keys); w.writeheader(); [w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k in keys}) for r in seeds_rows]
    # per config summary over seeds
    summ = {}
    for c in cfgs:
        rs = [r for r in seeds_rows if (r['ec'], r['wc'], r['dsg']) == c]
        ch = np.array([r.get('best_cheb', np.nan) for r in rs])
        summ[cname(c)] = dict(label=clabel(c), seeds=len(rs), seeds_with_feasible=int(np.sum([r['feasible'] > 0 for r in rs])),
                                 best_cheb_median=float(np.nanmedian(ch)) if np.isfinite(ch).any() else None,
                                 best_cheb_min=float(np.nanmin(ch)) if np.isfinite(ch).any() else None, best_cheb_max=float(np.nanmax(ch)) if np.isfinite(ch).any() else None,
                                 n_in_global_front=int(np.sum(C[nd] == cname(c))))
    print(json.dumps(summ, ensure_ascii=False, indent=1))
    # candidates: best compromise per configuration + global extremes among reasonable designs
    s, z = cheb(F)
    cand = []
    for c in cfgs:
        m = np.where(C == cname(c))[0]
        if len(m): cand.append(('best-' + cname(c), m[np.argmin(s[m])]))
    ok = np.where(s <= max(1.6, 1.5 * s.min()))[0]                       # "reasonable" relative to the best compromise
    groups = [('', ok)]
    if S2E:                                                               # stage 2e: extremes per design (A vs B compared like for like)
        dz = np.array([cc.split(':')[0] for cc in C])
        groups = [(g + ':', ok[dz[ok] == g]) for g in ('A', 'B')]
    for pre, okg in groups:
        for k, nm in [(4, 'best-fit'), (3, 'best-sens'), (1, 'best-tmin'), (0, 'best-err')]:
            if len(okg): cand.append((pre + nm, okg[np.argmin(F[okg, k])]))
        if S2E:                                                           # robust and well folded: sens <= 2.3, err <= 3.5, prot <= 1.5, tmin >= 25 -> best fit
            dz = np.array([cc.split(':')[0] for cc in C])
            k = np.where((dz == pre[:-1]) & (F[:, 3] <= 2.3) & (F[:, 0] <= 3.5) & (F[:, 2] <= 1.5) & (F[:, 1] <= -25))[0]
            if len(k): cand.insert(0, (pre + 'robust-fit', k[np.argmin(F[k, 4])]))
    scr = OUT / 'screen_mc.json'
    if S2E and STAGE == '2f' and scr.exists():                            # stage 2f: picked on the nonlinear MC (screen_mc.py) instead of the
        rows = json.load(open(scr, encoding='utf-8'))                     # linear sensitivity: err <= 4, MC +-0.20 mm 95% <= 8.5 deg -> best fit
        for g in ('A', 'B'):
            k = [r for r in rows if r['C'].startswith(g + ':') and r['F'][0] <= 4.0 and r['mc020']['p95'] <= 8.5]
            if k:
                b_ = min(k, key=lambda r: r['F'][4]); assert np.allclose(F[nd[b_['i']]], b_['F'], atol=1e-3)
                cand.insert(0, (g + ':robust-mc', nd[b_['i']]))
    gate = OUT / 'gate.json'
    if S2E and STAGE == '2f' and gate.exists():                           # stage 2f gate (gate_2f.py): only designs that passed every post-check,
        rows = [r for r in json.load(open(gate, encoding='utf-8')) if r['pass']]   # picked by the user's priorities (2026-11-15):
        cand = []                                                         # repeatable, folded / open shape, opens easily
        for grp in sorted({r['C'] for r in rows}):
            k = [r for r in rows if r['C'] == grp and r['F'][0] <= 4.0]
            rules = [('repeat', lambda r: r['fit_bent'] <= 15, lambda r: r['mc020']),
                     ('shape', lambda r: r['mc020'] <= 9.0, lambda r: r['fit_bent']),
                     ('open', lambda r: r['mc020'] <= 9.5 and r['fit_bent'] <= 15, lambda r: r['F'][1])]
            for nm, keep, key in rules:
                kk = [r for r in k if keep(r)]
                if kk:
                    b_ = min(kk, key=key); assert np.allclose(F[nd[b_['i']]], b_['F'], atol=1e-3)
                    cand.append((f'{grp}:{nm}', nd[b_['i']]))
    seen, uniq = set(), []
    kept = OUT / 'kept.json'
    if S2E and STAGE == '2f' and gate.exists() and kept.exists():      # user 2026-11-17: earlier candidates keep their ids; new picks only for
        kj = json.load(open(kept, encoding='utf-8'))                     # the configs in "new", numbered with their own prefix
        for k in kj['kept']:
            m = [i for i in nd if C[i] == k['C'] and np.allclose(F[i], k['F'], atol=6e-3)]
            assert len(m) == 1, (k['id'], m); seen.add(m[0]); uniq.append((k['id'], k['pick'], m[0]))
        cnt = collections.Counter()
        for nm, i in cand:
            pre = next((v for cf, v in kj['new'].items() if (':' + cf) in nm), None)
            if pre is None or i in seen: continue
            cnt[pre] += 1; seen.add(i); uniq.append((f'{pre}{cnt[pre]}', nm, i))
    else:
        for nm, i in cand:
            if i in seen: continue
            seen.add(i); uniq.append((f"{PREFIX}{len(uniq) + 1}", nm, i))
    out = []
    for j, (cid, nm, i) in enumerate(uniq):
        dsg, ecwc = C[i].split(':') if ':' in C[i] else ('', C[i]); ec, wc = ecwc.split('/')
        ev = make_ev(ec, wc, WIDTH, dsg)
        r, e = describe(ev, X[i])
        r.update(id=cid, pick=nm, ec=ec, wc=wc, dsg=dsg, label=clabel((ec, wc, dsg)), x=X[i].tolist(), cheb=float(s[i]))
        r['mc005'] = monte_carlo(ev, X[i], amp=0.05); r['mc010'] = monte_carlo(ev, X[i], amp=0.10); r['mc020'] = monte_carlo(ev, X[i], amp=0.20)   # 3D-printer tolerance
        r['mc_play'] = monte_carlo(ev, X[i], amp=0.0)                                  # joint play only (bearings / balls / resin pins)
        for w in (40, 50):
            ew = make_ev(ec, wc, w, dsg); Fw, Gw, xw = ew(X[i][None], extra=True)
            r[f'w{w}'] = dict(fit=float(Fw[0, 4]), prot=float(Fw[0, 2]), sweep=float(xw['sweep'][0]), tip_off=float(xw['fitx']['tipd'][0]))
        out.append(r)
        print(r['id'], nm, ec, wc, 'F', np.round(r['F'], 2), 'MC0.05 p95 %.2f max %.2f fail %.3f' % (r['mc005']['p95'], r['mc005']['max'], r['mc005']['fail']),
              'swivel %.0f' % r['swivel'], 'w40 fit %.1f w50 fit %.1f' % (r['w40']['fit'], r['w50']['fit']), flush=True)
    json.dump(dict(width=WIDTH, configs=summ, n_feasible=int(len(F)), n_front=int(len(nd))), open(OUT / 'summary.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    json.dump(out, open(OUT / 'candidates.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    np.savez_compressed(OUT / 'front.npz', X=np.array([X[i] for i in nd], dtype=object), F=F[nd], C=C[nd])
    figures(F, C, nd, out, seeds_rows, cfgs)


def figures(F, C, nd, cands, seeds_rows, cfgs):
    cols = dict(zip([cname(c) for c in cfgs] if S2E else ['/'.join(c) for c in CFG_JA], ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e', '#8c564b', '#17becf', '#bcbd22', '#e377c2', '#7f7f7f']))
    Fp = F.copy(); Fp[:, 1] *= -1
    pairs = [(0, 3), (2, 4), (1, 3), (4, 3), (5, 2), (0, 1)]
    fig, axs = plt.subplots(2, 3, figsize=(20, 12))
    for ax, (a, b) in zip(axs.flat, pairs):
        for c in cols:
            m = C == c
            if not m.any(): continue
            ax.scatter(Fp[m, a], Fp[m, b], s=3, c=cols[c], alpha=0.15)
            mm = np.intersect1d(np.where(m)[0], nd)
            ax.scatter(Fp[mm, a], Fp[mm, b], s=12, c=cols[c], label=c, edgecolors='k', linewidths=0.3)
        for r in cands:
            v = np.array(r['F']); v[1] *= -1
            ax.scatter(v[a], v[b], s=120, facecolors='none', edgecolors='k', linewidths=1.5); ax.annotate(r['id'], (v[a], v[b]), fontsize=10, xytext=(4, 4), textcoords='offset points')
        ax.set_xlabel(OBJ[a]); ax.set_ylabel(OBJ[b]); ax.grid(alpha=.3)
        lim = np.nanpercentile(Fp[nd], [1, 99], axis=0)
        ax.set_xlim(lim[0, a] - 0.1 * abs(lim[1, a] - lim[0, a]) - 0.1, lim[1, a] * 1.05 + 0.1); ax.set_ylim(lim[0, b] - 0.1 * abs(lim[1, b] - lim[0, b]) - 0.1, lim[1, b] * 1.05 + 0.1)
    axs[0, 0].legend(fontsize=8, markerscale=2)
    fig.suptitle(f'Stage 2 Pareto fronts (feasible designs, bold = globally non-dominated, circles = candidates), body width {WIDTH} mm', fontsize=13)
    plt.tight_layout(); plt.savefig(OUT / 'pareto.png', dpi=70); plt.close(fig)
    # seeds
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    for i, c in enumerate(cfgs):
        v = [r['best_cheb'] for r in seeds_rows if (r['ec'], r['wc'], r['dsg']) == c and 'best_cheb' in r]
        ax.scatter(np.full(len(v), i) + np.random.default_rng(0).uniform(-0.15, 0.15, len(v)), v, c=cols[cname(c)], s=30)
        nf = sum(1 for r in seeds_rows if (r['ec'], r['wc'], r['dsg']) == c and r['feasible'] == 0)
        ax.annotate(f'no feasible: {nf}', (i, 0), fontsize=8, ha='center')
    ax.set_xticks(range(len(cfgs))); ax.set_xticklabels([cname(c) for c in cfgs]); ax.set_ylabel('best compromise score per seed (lower = better)')
    ax.set_title('seed-to-seed variation (each dot = one NSGA-III run)'); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(OUT / 'seeds.png', dpi=70); plt.close(fig)


if __name__ == '__main__':
    main()
