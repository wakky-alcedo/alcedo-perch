"""Print the structure of the reference meshes (geometry names, vertex counts, bounds)."""
import sys, pathlib, numpy as np, trimesh
ROOT = pathlib.Path(__file__).resolve().parents[2]          # mech/
for f in ['common_kingfisher', 'belted_kingfisher']:
    s = trimesh.load(ROOT / 'ref' / 'models' / f'{f}.glb')
    print('==', f, type(s).__name__)
    print(' meta', {k: str(v)[:400] for k, v in s.metadata.items()})
    for name, g in s.geometry.items():
        print(' geom', name, len(g.vertices), len(g.faces), 'watertight', g.is_watertight, 'ext', np.round(g.extents, 3))
    for n in s.graph.nodes_geometry:
        T, gname = s.graph[n]
        print(' node', n, '->', gname, 'scale', np.round(np.linalg.norm(T[:3, :3], axis=0), 4))
    m = s.dump(concatenate=True)
    print(' bounds', np.round(m.bounds, 3), 'extents', np.round(m.extents, 3))
