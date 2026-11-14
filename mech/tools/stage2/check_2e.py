"""Stage 2e post-checks of the candidates: exact layer plan over all 41 poses (stack.full_check), stud / pin lengths, and the
elastic stretch of the cranked wrist rod 4.
Cranked rod: its body sits e = |level| * LAYER_T off the line through its ball centres; with an axial force F the body bends
under M = F e, and the ball-to-ball distance changes by delta = F e^2 L / (E I) (rigid cranks, pinned ends). Material and
section are ESTIMATES: PLA E = 3000 N/mm2, section 2.0 mm wide x 0.9 mm in the bending (layer) direction, or stiffer sections.
F = the largest rod-4 force with the spring at twice its minimum preload (loads.py). The angle error is delta x d(wrist)/dL4.
usage: check_2e.py   (after analyze.py 45 2e and loads.py 2e)  -> results/stage2/2e/check_2e.json"""
import json, pathlib, numpy as np
from analyze import make_ev
from stack import full_check, T
ROOT = pathlib.Path(__file__).resolve().parents[2]
from problem import STAGE
OUT = ROOT / 'results' / 'stage2' / STAGE
E_PLA = 3000.0
SECTIONS = {'plate 2.0 x 0.9': (2.0, 0.9), 'plate 2.0 x 2.1 (two layers)': (2.0, 2.1), 'tube/truss equiv. 3 x 3': (3.0, 3.0)}


def main():
    cands = json.load(open(OUT / 'candidates.json', encoding='utf-8'))
    loads = json.load(open(OUT / 'loads.json', encoding='utf-8'))
    res = {}
    for c in cands:
        ev = make_ev(c['ec'], c['wc'], 45, c.get('dsg', 'B'))
        v, plan, studs = full_check(ev, np.asarray(c['x'], float))
        r = dict(unplaced=v, plan=plan, studs_mm=studs)
        L = loads.get(c['id'], {})
        F4 = L.get('force_max_with_spring_N', {}).get('L4')
        if F4 is not None and v == 0:
            F, G, e = ev(np.asarray(c['x'], float)[None], extra=True)
            L4 = float(np.asarray(e['d']['L4']).ravel()[0])
            ecc = abs(plan.get('rod4', 0)) * T
            k = L['dangle_dL_deg_per_mm']['L4']['M']
            r['rod4'] = dict(L_mm=round(L4, 1), crank_offset_mm=round(ecc, 2), F_max_N=round(F4, 3), dM_dL_deg_per_mm=round(k, 2),
                             stretch={nm: dict(delta_mm=round(F4 * ecc ** 2 * L4 / (E_PLA * b * h ** 3 / 12), 4),
                                               angle_deg=round(k * F4 * ecc ** 2 * L4 / (E_PLA * b * h ** 3 / 12), 3))
                                      for nm, (b, h) in SECTIONS.items()})
        res[c['id']] = r
        print(c['id'], c.get('dsg'), c['ec'], c['wc'], json.dumps(r, ensure_ascii=False), flush=True)
    json.dump(res, open(OUT / 'check_2e.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
