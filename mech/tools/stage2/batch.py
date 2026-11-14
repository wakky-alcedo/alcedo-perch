"""Run all configurations x seeds in parallel (one process per run). usage: batch.py gens pop seeds workers [width] [tilt]
tilt: design B, without the body-mounted elbow bellcrank (it protruded 7-17 mm in every planar run)."""
import sys, subprocess, itertools, pathlib, time
from concurrent.futures import ThreadPoolExecutor
HERE = pathlib.Path(__file__).resolve().parent
PY = sys.executable
TILT = 'tilt' in sys.argv; argv = [v for v in sys.argv if v != 'tilt']
gens, pop, nseeds, workers = (int(v) for v in argv[1:5]); width = argv[5] if len(argv) > 5 else '45'
CFG = [('hum', 'ulna'), ('body', 'ulna'), ('ulna', 'ulna'), ('hum', 'hum'), ('body', 'hum'), ('ulna', 'hum')]
# stage 2d: the two bellcrank configurations that survived +-0.2 mm, and the ones with one or no bellcrank (direct rods)
if TILT: CFG = [('hum', 'hum'), ('ulna', 'hum'), ('hum', 'direct'), ('ulna', 'direct'), ('direct', 'hum'), ('direct', 'direct')]
S2E = '2e' in argv; S2F = '2f' in argv; argv = [v for v in argv if v not in ('2e', '2f')]
jobs = [(ec, wc, s) for s in range(nseeds) for ec, wc in CFG]
# stage 2e: design A (planar) and B (tilted wrist) with the real-height layer plan; bellcranks only on the humerus
if S2E: jobs = [(ec, wc, s, dsg) for s in range(nseeds) for dsg in ('dA', 'dB') for ec, wc in [('hum', 'hum'), ('hum', 'direct'), ('direct', 'hum'), ('direct', 'direct')]]
# slow configs first so the pool stays busy
# stage 2f (run with MIDLINE=1 in the environment): midline constraint, only the configuration that admitted a layer plan in 2e
if S2F: jobs = [('hum', 'direct', s, dsg) for s in range(nseeds) for dsg in ('dA', 'dB')]
jobs.sort(key=lambda j: 0 if j[0] == 'ulna' else 1)
t0 = time.time()


def run(j):
    ec, wc, s = j[:3]
    r = subprocess.run([PY, str(HERE / 'run.py'), ec, wc, str(s), str(gens), str(pop), width] + (['tilt'] if TILT else []) + list(j[3:]), capture_output=True, text=True, cwd=HERE)
    line = (r.stdout.strip().splitlines() or ['(no output) ' + r.stderr[-300:]])[-1]
    print(f'[{time.time() - t0:5.0f}s] {line}', flush=True)


with ThreadPoolExecutor(workers) as ex:
    list(ex.map(run, jobs))
print('done', time.time() - t0)
