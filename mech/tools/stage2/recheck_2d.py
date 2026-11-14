"""Stage 2e: the stage-2d candidates D1-D10 under the stage-2e layer rules (thickness option b, cranked rod 4, skewer rod 1),
exact over all 41 poses.  usage: recheck_2d.py  -> results/stage2/2e/recheck_2d.json"""
import json, pathlib, numpy as np
from analyze import make_ev
from stack import full_check
R2 = pathlib.Path(__file__).resolve().parents[2] / 'results' / 'stage2'
out = {}
for c in json.load(open(R2 / '2d' / 'candidates.json', encoding='utf-8')):
    if c['ec'] == 'ulna' or c['wc'] == 'ulna':
        out[c['id']] = dict(ec=c['ec'], wc=c['wc'], skipped='bellcrank on the ulna: not in the stage-2e configurations'); continue
    ev = make_ev(c['ec'], c['wc'], 45, 'B')
    v, plan, studs = full_check(ev, np.asarray(c['x'], float))
    out[c['id']] = dict(ec=c['ec'], wc=c['wc'], unplaced=v, plan=plan, studs_mm=studs)
    print(c['id'], out[c['id']], flush=True)
json.dump(out, open(R2 / '2e' / 'recheck_2d.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
