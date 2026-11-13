"""Add the stage-2 candidates (results/stage2/candidates.json) to the simulator's version list (part1.html, between the
__STAGE2 markers), rebuild kingfisher_wing_fold.html, and check that the simulator reproduces the Python kinematics."""
import json, pathlib, subprocess, sys, numpy as np
from problem import Evaluator, U
from mech import *

ROOT = pathlib.Path(__file__).resolve().parents[2]
TOOLS = ROOT / 'tools'
r4 = lambda v: round(float(v), 4)


def entry(c, mesh):
    ev = Evaluator(c['ec'], c['wc'], 45)
    F, G, e = ev(np.array([c['x']]), extra=True); d = e['d']
    W = dict(G2x=r4(d['G2'][0, 0]), G2y=r4(d['G2'][0, 1]), h3x=r4(d['h3'][0, 0]), h3y=r4(d['h3'][0, 1]), rb3=r4(d['rb3'][0]), Lc3=r4(d['Lc3'][0]),
             b4x=r4(d['b4'][0, 0]), b4y=r4(d['b4'][0, 1]), L4=r4(d['L4'][0]), qx=r4(d['q'][0, 0]), qy=r4(d['q'][0, 1]))
    if c['ec'] == 'body':
        topo, em = 'two3', 'body'
        dims = dict(G1x=r4(d['G1w'][0, 0]), G1y=r4(d['G1w'][0, 1]), G1z=r4(d['G1w'][0, 2]), h1x=r4(d['h1'][0, 0]), h1y=r4(d['h1'][0, 1]), h1z=r4(d['h1'][0, 2]),
                    rb1=r4(d['rb1'][0]), Lc1=r4(d['Lc1'][0]), b2x=r4(d['b2'][0, 0]), b2y=r4(d['b2'][0, 1]), b2z=r4(d['b2'][0, 2]), L2=r4(d['L2'][0]),
                    wx=r4(d['w'][0, 0]), wy=r4(d['w'][0, 1]), **W)
    elif c['ec'] == 'hum':
        topo, em = 'twoB', 'hum'
        dims = dict(Ax=r4(d['A'][0, 0]), Ay=r4(d['A'][0, 1]), Az=r4(d['A'][0, 2]), G1x=r4(d['g'][0, 0]), G1y=r4(d['g'][0, 1]), rb1=r4(d['rb1'][0]), Lc1=r4(d['Lc1'][0]),
                    b2x=r4(d['b2'][0, 0]), b2y=r4(d['b2'][0, 1]), L2=r4(d['L2'][0]), wx=r4(d['w'][0, 0]), wy=r4(d['w'][0, 1]), **W)
    else:
        topo, em = 'twoU', 'ulna'
        dims = dict(Ax=r4(d['A'][0, 0]), Ay=r4(d['A'][0, 1]), Az=r4(d['A'][0, 2]), G1x=r4(d['g'][0, 0]), G1y=r4(d['g'][0, 1]), rb1=r4(d['rb1'][0]), Lc1=r4(d['Lc1'][0]),
                    b2x=r4(d['b2'][0, 0]), b2y=r4(d['b2'][0, 1]), L2=r4(d['L2'][0]), kx=r4(d['k'][0, 0]), ky=r4(d['k'][0, 1]), **W)
    F0 = c['F']
    why = (f"段階2の候補（{c['label']}、選び方: {c['pick']}）。胴幅45 mmのCGに合わせて、畳み姿勢とリンク寸法を同時に多目的最適化した局所解の1つ。"
           f"肘{c['E'][0]:.0f}→{c['E'][1]:.0f}°、手首{c['M'][0]:.0f}→{c['M'][1]:.0f}°、関節角の誤差{F0[0]:.1f}°、最小伝達角{-F0[1]:.0f}°、"
           f"はみ出し（CGメッシュ基準）{F0[2]:.1f} mm、寸法感度2σ {F0[3]:.1f}°（MC ±0.05 mmで95%が{c['mc005']['p95']:.1f}°以内）、"
           f"畳み姿勢の当てはまり{F0[4]:.1f}、最大部品{F0[5]:.0f} mm、ボールジョイント首振り{c['swivel']:.0f}°、羽根の浮かせ{c['fz']:.1f} mm、モーター{c['theta']:.0f}°。")
    occ = mesh['variants']['w45']['occluder']
    return dict(id=c['id'], status='candidate', date='2026-11-13', title=f"段階2候補 {c['id']}：{c['label']}", topo=topo, wristMount=c['wc'], elbowMount=em,
                why=why, shoulder=dict(axis=[r4(v) for v in d['axis'][0]], Ro=[[r4(v) for v in row] for row in Ro]), body=dict(c=occ['c'], r=occ['r']),
                bodyMesh='w45', featherZ=r4(d['fz'][0]),
                target=dict(AH0=0, AH1=round(float(np.degrees(d['th'][0])), 3), E0=150.0, E1=r4(d['E1'][0]), M0=160.0, M1=r4(d['M1'][0])),
                tilt3d=dict(tS=0, tE=0, tW=0), dims=dims), e


def main():
    cands = json.load(open(ROOT / 'results' / 'stage2' / 'candidates.json', encoding='utf-8'))
    import re
    mtxt = (ROOT / 'ref' / 'derived' / 'ck_mesh.js').read_text(encoding='utf-8')
    mesh = json.loads(mtxt[mtxt.index('=') + 1:].strip().rstrip(';'))
    entries, ex = [], {}
    for c in cands:
        en, e = entry(c, mesh); entries.append(en); ex[c['id']] = e
    p1 = (TOOLS / 'part1.html').read_text(encoding='utf-8')
    block = '// __STAGE2_START__\n' + ''.join('  ' + json.dumps(en, ensure_ascii=False) + ',\n' for en in entries) + '  // __STAGE2_END__\n'
    if '// __STAGE2_START__' in p1:
        a = p1.index('// __STAGE2_START__'); b = p1.index('// __STAGE2_END__\n') + len('// __STAGE2_END__\n')
        p1 = p1[:a] + block.lstrip() + p1[b:]
    else:
        p1 = p1.replace('const VERSIONS = [\n', 'const VERSIONS = [\n  ' + block, 1)
    (TOOLS / 'part1.html').write_text(p1, encoding='utf-8')
    html = p1 + (TOOLS / 'part2.html').read_text(encoding='utf-8')
    (ROOT / 'kingfisher_wing_fold.html').write_text(html, encoding='utf-8')
    # verification against the simulator
    for c in cands:
        out = ROOT / 'results' / 'local' / f"sim_{c['id']}.json"
        subprocess.run(['node', str(TOOLS / 'stage1' / 'dump_path.js'), c['id'], str(out)], cwd=TOOLS, check=True, capture_output=True)
        s = json.load(open(out)); Es = np.array([q['Edeg'] for q in s['steps']]); Ms = np.array([q['Mdeg'] for q in s['steps']])
        u = np.array([q['u'] for q in s['steps']])
        f = ex[c['id']]['f']
        Ep, Mp = np.interp(u, U, f['E'][0]), np.interp(u, U, f['M'][0])
        print(c['id'], c['ec'], c['wc'], 'sim vs python: max |dE| %.3f |dM| %.3f, rodTypes %s, sim prot (ellipsoid) %.2f' % (
            np.abs(Es - Ep).max(), np.abs(Ms - Mp).max(), ''.join(s['rodTypes']), max(max(q['prot']) for q in s['steps'])))


if __name__ == '__main__':
    main()
