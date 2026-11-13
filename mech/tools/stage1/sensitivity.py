"""v4 vs the Common Kingfisher mesh under different assumptions: shoulder placement (estimate) and body slimming."""
import sys, json, pathlib, itertools, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from ck_align import transform, ROOT
from eval_v4_ck import evaluate
import v4geom
d = v4geom.load()
rows = []
cases = [dict(), dict(SX=5), dict(SX=13), dict(DY=0), dict(DY=8), dict(DZ=4), dict(DZ=10),
         dict(kx=40 / 57.7, kz=40 / 57.7), dict(kx=34 / 57.7, kz=34 / 57.7), dict(kx=40 / 57.7, kz=40 / 57.7, DZ=5)]
keys = ['fold_bone_sd_min', 'fold_feather_sd_min', 'fold_feather_sd_max', 'fold_feather_inside_frac(<-1mm)', 'tip_dist',
        'visible_shell_covered_frac(<4mm)', 'visible_shell_to_v4_median', 'mech_prot_mesh_max']
for c in cases:
    tr = transform(**c); r = evaluate(tr, d)
    row = dict(case=c, **{k: round(r[k], 2) for k in keys}); rows.append(row); print(json.dumps(row), flush=True)
json.dump(rows, open(ROOT / 'results' / 'stage1' / 'sensitivity.json', 'w'), indent=1)
