"""Stage 2f gate: every member of the Monte Carlo screen (screen_mc.json) goes through all post-checks, so the candidates can be
chosen among designs that are known to pass, by the user's priorities (2026-11-15: opens easily, folded / open shape, smooth and
repeatable motion; local protrusions are acceptable):
- layer plan, all 41 poses, no search limit (stack.full_check);
- crossing of parts in plan, true skewer diameter (crossing_check.check);
- left / right wing at the body midline (mirror_check);
- smoothness: max gain (deg per deg of motor) and its largest step between neighbouring poses;
- fold fit with bowed feathers (feather_bend, coarse grid).
usage: MIDLINE=1 gate_2f.py [screen file]  -> results/stage2/<STAGE>/gate.json"""
import sys, json, numpy as np
SCREEN = sys.argv[1] if len(sys.argv) > 1 else 'screen_mc.json'
sys.argv = sys.argv[:1] + ['45', '2e']
from analyze import make_ev, OUT
from problem import U, GAP
from stack import full_check
from crossing_check import check as crossing
from mirror_check import build, mir, seg2
import feather_bend as fb

fb.A_GRID = np.radians([0, 10, 20, 30]); fb.R_GRID = np.array([35, 50, 70, 100, np.inf])


def mirror(c, plan):
    caps, f = build(c, plan); best = np.inf
    for na, pa, qa, ra in caps:
        for nb, pb, qb, rb in caps:
            d, _, _ = seg2(pa, qa, mir(pb), mir(qb)); best = min(best, float((d - ra - rb).min()))
    return best - GAP


def main():
    rows = json.load(open(OUT / SCREEN, encoding='utf-8'))
    z = np.load(OUT / 'front.npz', allow_pickle=True)
    evs = {}; out = []
    for r in rows:
        dsg, cfg = r['C'].split(':'); ec, wc = cfg.split('/')
        ev = evs.setdefault(r['C'], make_ev(ec, wc, 45, dsg)); x = np.asarray(z['X'][r['i']], float)
        c = dict(id=f"m{r['i']}", ec=ec, wc=wc, dsg=dsg, x=x.tolist())
        res = dict(i=r['i'], C=r['C'], F=r['F'], mc005=r['mc005']['p95'], mc020=r['mc020']['p95'], mc020max=r['mc020']['max'])
        v, plan, studs = full_check(ev, x); res.update(unplaced=v, plan=plan)
        if v == 0:
            cr = crossing(c, plan); res['cross_min'] = min([t['clear'] for t in cr.values()], default=1.0)
            res['cross_pair'] = min(cr, key=lambda k: cr[k]['clear']) if cr else None
            res['mirror_min'] = round(mirror(c, plan), 2)
        F, G, e = ev(x[None], extra=True); d, f = e['d'], e['f']
        gE, gM = ev.gain(d, f); g = np.maximum(gE, gM)[0]
        res['gain_max'] = round(float(g.max()), 2)
        res['gain_step'] = round(float(max(np.abs(np.diff(gE[0])).max(), np.abs(np.diff(gM[0])).max())), 3)
        f0 = fb.fit_of(ev, d, (0, np.inf, 0, np.inf))[0]
        bs = min(((fb.fit_of(ev, d, (a, R, 0, np.inf))[0], a, R) for a in fb.A_GRID for R in fb.R_GRID))
        bp = min(((fb.fit_of(ev, d, (bs[1], bs[2], a, R))[0], a, R) for a in fb.A_GRID for R in fb.R_GRID))
        res.update(fit_flat=round(f0, 2), fit_bent=round(bp[0], 2), bend=[round(float(np.degrees(bs[1])), 1), float(bs[2]),
                                                                         round(float(np.degrees(bp[1])), 1), float(bp[2])])
        res['pass'] = bool(v == 0 and res['cross_min'] >= 0 and res['mirror_min'] >= 0)
        out.append(res)
        print(res['i'], res['C'], 'pass' if res['pass'] else 'FAIL', {k: res.get(k) for k in ('unplaced', 'cross_min', 'mirror_min', 'mc020',
              'gain_max', 'gain_step', 'fit_flat', 'fit_bent')}, flush=True)
        json.dump(out, open(OUT / ('gate.json' if SCREEN == 'screen_mc.json' else 'gate_' + SCREEN), 'w', encoding='utf-8'), indent=1, default=float)


if __name__ == '__main__':
    main()
