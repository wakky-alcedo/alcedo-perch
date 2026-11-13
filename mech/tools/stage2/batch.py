"""Run all configurations x seeds in parallel (one process per run). usage: batch.py gens pop seeds workers [width] [tilt]
tilt: design B, without the body-mounted elbow bellcrank (it protruded 7-17 mm in every planar run)."""
import sys, subprocess, itertools, pathlib, time
from concurrent.futures import ThreadPoolExecutor
HERE = pathlib.Path(__file__).resolve().parent
PY = sys.executable
TILT = 'tilt' in sys.argv; argv = [v for v in sys.argv if v != 'tilt']
gens, pop, nseeds, workers = (int(v) for v in argv[1:5]); width = argv[5] if len(argv) > 5 else '45'
CFG = [('hum', 'ulna'), ('body', 'ulna'), ('ulna', 'ulna'), ('hum', 'hum'), ('body', 'hum'), ('ulna', 'hum')]
if TILT: CFG = [c for c in CFG if c[0] != 'body']
jobs = [(ec, wc, s) for s in range(nseeds) for ec, wc in CFG]
# slow configs first so the pool stays busy
jobs.sort(key=lambda j: 0 if j[0] == 'ulna' else 1)
t0 = time.time()


def run(j):
    ec, wc, s = j
    r = subprocess.run([PY, str(HERE / 'run.py'), ec, wc, str(s), str(gens), str(pop), width] + (['tilt'] if TILT else []), capture_output=True, text=True, cwd=HERE)
    line = (r.stdout.strip().splitlines() or ['(no output) ' + r.stderr[-300:]])[-1]
    print(f'[{time.time() - t0:5.0f}s] {line}', flush=True)


with ThreadPoolExecutor(workers) as ex:
    list(ex.map(run, jobs))
print('done', time.time() - t0)
