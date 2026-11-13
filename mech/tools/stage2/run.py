"""One NSGA-III run. usage: run.py ec wc seed [gens] [pop] [width]  -> results/stage2/runs/<ec>_<wc>_s<seed>.npz"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
import sys, time, pathlib, numpy as np
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize
from pymoo.core.callback import Callback
from problem import LinkProblem

ROOT = pathlib.Path(__file__).resolve().parents[2]


class Log(Callback):
    def __init__(self):
        super().__init__(); self.rows = []

    def notify(self, algo):
        F, G = algo.pop.get('F'), algo.pop.get('G'); feas = np.all(G <= 0, 1)
        cv = np.maximum(G, 0).sum(1)
        best = F[feas].min(0) if feas.any() else np.full(F.shape[1], np.nan)
        self.rows.append([algo.n_gen, feas.sum(), cv.min()] + list(best))


def main(ec, wc, seed, gens=300, pop=300, width=45):
    prob = LinkProblem(ec, wc, width)
    ref = get_reference_directions('energy', 6, 120, seed=1)
    # initial population: half uniform random, half with the pose near the best pose-only solution (the linkage part random)
    rng = np.random.default_rng(seed); xl, xu = prob.xl, prob.xu
    X0 = xl + (xu - xl) * rng.random((pop, len(xl)))
    import json
    cands = [json.load(open(f)) for f in (ROOT / 'results' / 'stage2' / 'pose_only').glob(f'w{int(width)}_s*.json') if 'nosweep' not in f.name]
    pj = min(cands, key=lambda r: r['cost'])['x']           # best pose-only solution over its seeds
    half = pop // 2
    X0[:half, :6] = np.clip(np.array(pj) + rng.normal(0, 1, (half, 6)) * np.array([0.08, 0.08, 0.08, 1.5, 1.5, 0.5]), xl[:6], xu[:6])
    algo = NSGA3(ref_dirs=ref, pop_size=pop, sampling=X0, eliminate_duplicates=True)
    log = Log(); t = time.time()
    res = minimize(prob, algo, ('n_gen', gens), seed=seed, callback=log, verbose=False)
    pop_ = res.pop; X, F, G = pop_.get('X'), pop_.get('F'), pop_.get('G')
    out = ROOT / 'results' / 'stage2' / 'runs'; out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f'{ec}_{wc}_w{int(width)}_s{seed}.npz', X=X, F=F, G=G, log=np.array(log.rows), time=time.time() - t,
                        vars=np.array([v[0] for v in prob.ev.vars]))
    feas = np.all(G <= 0, 1)
    print(f'{ec}/{wc} seed {seed}: {time.time() - t:.0f}s, feasible {feas.sum()}/{len(X)}',
          ('best ' + ' '.join(f'{v:.2f}' for v in F[feas].min(0))) if feas.any() else f'min CV {np.maximum(G, 0).sum(1).min():.2f}', flush=True)


if __name__ == '__main__':
    a = sys.argv[1:]
    main(a[0], a[1], int(a[2]), *(int(v) for v in a[3:6]))
