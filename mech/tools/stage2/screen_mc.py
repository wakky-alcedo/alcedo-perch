"""Stage 2f: the linear sensitivity (objective 4) did not rank the candidates by their nonlinear spread at +-0.20 mm (F1: sens 2.13,
MC 95% 10.2 deg; E1: 2.21, 6.7 deg). Screen the merged front with the Monte Carlo directly: every member that meets loose limits gets
MC at +-0.05 and +-0.20 mm (200 samples each, joint play included).
usage: MIDLINE=1 screen_mc.py  -> results/stage2/<STAGE>/screen_mc.json"""
import sys, json, numpy as np
sys.argv = sys.argv[:1] + ['45', '2e']
from analyze import make_ev, monte_carlo, OUT
z = np.load(OUT / 'front.npz', allow_pickle=True); F, C = z['F'], z['C']
m = (F[:, 3] <= 3.0) & (F[:, 0] <= 4.5) & (F[:, 2] <= 2.0) & (-F[:, 1] >= 25) & (F[:, 4] <= 18)
# no bellcrank (direct/direct): its elbow rod always stands 10+ mm off the body; the user accepts local protrusions (2026-11-15)
zero = np.array(['direct/direct' in str(c) for c in C])
m |= zero & (F[:, 3] <= 3.0) & (F[:, 0] <= 4.5) & (F[:, 2] <= 15.0) & (-F[:, 1] >= 25) & (F[:, 4] <= 19.5)
evs = {}; rows = []
for i in np.nonzero(m)[0]:
    dsg, cfg = str(C[i]).split(':'); ec, wc = cfg.split('/')
    ev = evs.setdefault(str(C[i]), make_ev(ec, wc, 45, dsg)); x = np.asarray(z['X'][i], float)
    a = monte_carlo(ev, x, n=200, amp=0.05); b = monte_carlo(ev, x, n=200, amp=0.20)
    rows.append(dict(i=int(i), C=str(C[i]), F=F[i].round(3).tolist(), mc005=a, mc020=b))
    print(i, C[i], F[i].round(2), round(a['p95'], 2), round(b['p95'], 2), flush=True)
json.dump(rows, open(OUT / 'screen_mc.json', 'w', encoding='utf-8'), indent=1)
