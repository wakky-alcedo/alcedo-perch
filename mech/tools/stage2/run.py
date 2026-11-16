"""One NSGA-III run. usage: run.py ec wc seed [gens] [pop] [width] [tilt]  -> results/stage2/runs[_tilt]/<ec>_<wc>_w<width>_s<seed>.npz
tilt: design B (tilted wrist hinge, spatial rod 4, primaries with root twist / shifted folded angle).
Since stage 2d (link interference + per-joint play) the design-B runs go to runs_2d/; configurations that were already run in
runs_tilt/ get a quarter of the initial population from their old final populations (warm start)."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
import sys, time, pathlib, numpy as np
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize
from pymoo.core.callback import Callback
from problem import LinkProblem, MIDLINE

ROOT = pathlib.Path(__file__).resolve().parents[2]


class Log(Callback):
    def __init__(self):
        super().__init__(); self.rows = []; self.progress = None; self.t0 = time.time()

    def notify(self, algo):
        F, G = algo.pop.get('F'), algo.pop.get('G'); feas = np.all(G <= 0, 1)
        cv = np.maximum(G, 0).sum(1)
        best = F[feas].min(0) if feas.any() else np.full(F.shape[1], np.nan)
        self.rows.append([algo.n_gen, feas.sum(), cv.min()] + list(best))
        if self.progress and algo.n_gen % 10 == 0:          # 2026-11-16: the batch log only shows finished runs
            self.progress.write_text(f'gen {algo.n_gen} feasible {feas.sum()} minCV {cv.min():.3f} t {time.time() - self.t0:.0f}s')


def main(ec, wc, seed, gens=300, pop=300, width=45, tilt=False, design=None):
    if design: tilt = 'A' if design == 'A' else True        # stage 2e: design A (planar) or B (tilted wrist), real-height layer plan
    prob = LinkProblem(ec, wc, width, tilt, stack=bool(design))
    ref = get_reference_directions('energy', 6, 120, seed=1)
    # initial population: half uniform random, half with the pose near the best pose-only solution (the linkage part random)
    rng = np.random.default_rng(seed); xl, xu = prob.xl, prob.xu
    X0 = xl + (xu - xl) * rng.random((pop, len(xl)))
    import json
    pat = f'tiltB_w{int(width)}_s*.json' if tilt else f'w{int(width)}_s*.json'
    cands = [json.load(open(f)) for f in (ROOT / 'results' / 'stage2' / 'pose_only').glob(pat) if 'nosweep' not in f.name and 'planar' not in f.name]
    pj = min(cands, key=lambda r: r['cost'])['x']           # best pose-only solution over its seeds
    half = pop // 2
    X0[:half, :6] = np.clip(np.array(pj[:6]) + rng.normal(0, 1, (half, 6)) * np.array([0.08, 0.08, 0.08, 1.5, 1.5, 0.5]), xl[:6], xu[:6])
    if tilt == 'A':                             # only dl from the pose-only optimum (tw = phi = 0)
        j = len(xl) - 3
        X0[:half, j] = np.clip(pj[8] + rng.normal(0, 4.0, half), xl[j], xu[j])
    elif tilt:                                  # tw, phi, dl from the pose-only optimum too (they are the last 5 variables but two)
        j = len(xl) - 5
        X0[:half, j:j + 3] = np.clip(np.array(pj[6:9]) + rng.normal(0, 1, (half, 3)) * np.array([4.0, 3.0, 4.0]), xl[j:j + 3], xu[j:j + 3])
    old = ROOT / 'results' / 'stage2' / ('runs_2d' if design else 'runs_tilt') / f'{ec}_{wc}_w{int(width)}_s{seed}.npz'
    if design and MIDLINE: old = ROOT / 'results' / 'stage2' / 'runs_2e' / f'{design}-{ec}_{wc}_w{int(width)}_s{seed}.npz'   # stage 2f: warm start from 2e (same variables)
    if design and MIDLINE and os.environ.get('WARM2F') == '1':     # stage 2f v1: restart from the v0 final populations (midline without layer offsets)
        old = ROOT / 'results' / 'local' / os.environ.get('WARM_DIR', 'runs_2f_v0') / f'{design}-{ec}_{wc}_w{int(width)}_s{seed}.npz'
    if tilt and old.exists():
        Xo = np.load(old)['X']
        if tilt == 'A' and Xo.shape[1] != len(xl): Xo = np.delete(Xo, [Xo.shape[1] - 5, Xo.shape[1] - 4], 1)       # drop tw, phi
        m = min(len(Xo), pop // (2 if os.environ.get('WARM2F') == '1' else 4)); X0[half:half + m] = np.clip(Xo[rng.choice(len(Xo), m, replace=False)], xl, xu)
    algo = NSGA3(ref_dirs=ref, pop_size=pop, sampling=X0, eliminate_duplicates=True)
    log = Log(); t = time.time()
    out = ROOT / 'results' / 'stage2' / (('runs_2f' if MIDLINE else 'runs_2e') if design else 'runs_2d' if tilt else 'runs')
    if os.environ.get('RUN_OUT'): out = ROOT / 'results' / 'stage2' / os.environ['RUN_OUT']      # parallel runs must not overwrite
    out.mkdir(parents=True, exist_ok=True)
    log.progress = out / f'{design + "-" if design else ""}{ec}_{wc}_w{int(width)}_s{seed}.progress'
    res = minimize(prob, algo, ('n_gen', gens), seed=seed, callback=log, verbose=False)
    pop_ = res.pop; X, F, G = pop_.get('X'), pop_.get('F'), pop_.get('G')
    np.savez_compressed(out / f'{design + "-" if design else ""}{ec}_{wc}_w{int(width)}_s{seed}.npz', X=X, F=F, G=G, log=np.array(log.rows), time=time.time() - t,
                        vars=np.array([v[0] for v in prob.ev.vars]))
    feas = np.all(G <= 0, 1)
    print(f'{design or ""}{ec}/{wc}{" tilt" if tilt else ""} seed {seed}: {time.time() - t:.0f}s, feasible {feas.sum()}/{len(X)}',
          ('best ' + ' '.join(f'{v:.2f}' for v in F[feas].min(0))) if feas.any() else f'min CV {np.maximum(G, 0).sum(1).min():.2f}', flush=True)


if __name__ == '__main__':
    a = sys.argv[1:]
    tilt = 'tilt' in a; a = [v for v in a if v != 'tilt']
    design = next((v[1] for v in a if v in ('dA', 'dB')), None); a = [v for v in a if v not in ('dA', 'dB')]
    main(a[0], a[1], int(a[2]), *(int(v) for v in a[3:6]), tilt=tilt, design=design)
