import sys, json, os, numpy as np
sys.argv = ['v3opt.py', 'wrist']
src = open('v3opt.py').read().split("if __name__ == '__main__':")[0]
src = src.replace("pen += 1.5 * max(0, 4.5 - np.hypot(wx, wy)) + 1.5 * max(0, 4.5 - np.hypot(hx, hy)) + 0.05 * max(0, swing - 100)",
                  "pen += 1.5 * max(0, 3.0 - np.hypot(wx, wy)) + 1.5 * max(0, 4.5 - np.hypot(hx, hy)) + 0.05 * max(0, swing - 140)")
exec(src)
from scipy.optimize import differential_evolution
x1 = np.array([19.193, -11.9998, 27.805, -3.755, 13.18, 30.75, -2.0, 0.8536, 0.2])
rng = np.random.default_rng(5)
lo = np.array([b[0] for b in B]); hi = np.array([b[1] for b in B])
pop = [x1] + [np.clip(x1 + rng.normal(0, 1, 9) * (hi - lo) * 0.05, lo, hi) for _ in range(120)] + [lo + rng.random(9) * (hi - lo) for _ in range(130)]
pop = np.array(pop); pop[:, 8] = np.clip(pop[:, 8], 0, 1)
r = differential_evolution(evaluate, B, init=pop, maxiter=350, tol=1e-12, polish=False, mutation=(0.5, 1.0), recombination=0.8, seed=11)
d = evaluate(r.x, detail=True)
print('REFINED', round(r.fun, 3), json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()}), flush=True)
json.dump(dict(cfg=cfg, which='wrist', x=r.x.tolist(), d=d), open('v3_wrist_refined.json', 'w'))
