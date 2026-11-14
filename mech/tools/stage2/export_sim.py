"""Add the stage-2 candidates (results/stage2/candidates.json) to the simulator's version list (part1.html, between the
__STAGE2 markers), rebuild kingfisher_wing_fold.html, and check that the simulator reproduces the Python kinematics.
usage: export_sim.py [tilt] [2d]   (tilt: design B candidates from results/stage2/tilt/, ids B1.., between the __STAGE2B markers;
                                    tilt 2d: stage-2d candidates from results/stage2/2d/, ids D1.., between the __STAGE2D markers)"""
import json, pathlib, subprocess, sys, numpy as np
from problem import Evaluator, U
from mech import *

ROOT = pathlib.Path(__file__).resolve().parents[2]
TOOLS = ROOT / 'tools'
r4 = lambda v: round(float(v), 4)
TILT = 'tilt' in sys.argv; S2D = '2d' in sys.argv; S2E = '2e' in sys.argv; TILT = TILT or S2E
from problem import STAGE
MARK = ('STAGE2F' if STAGE == '2f' else 'STAGE2E') if S2E else 'STAGE2D' if S2D else 'STAGE2B' if TILT else 'STAGE2'
CAND = ROOT / 'results' / 'stage2' / (STAGE if S2E else '2d' if S2D else 'tilt' if TILT else '') / 'candidates.json'


def make_ev(c):
    if c.get('dsg'): return Evaluator(c['ec'], c['wc'], 45, tilt='A' if c['dsg'] == 'A' else True, stack=True)
    return Evaluator(c['ec'], c['wc'], 45, tilt=TILT)



def entry(c, mesh):
    ev = make_ev(c)
    F, G, e = ev(np.array([c['x']]), extra=True); d = e['d']
    if c['wc'] == 'direct':
        W = dict(b4x=r4(d['b4'][0, 0]), b4y=r4(d['b4'][0, 1]), L4=r4(d['L4'][0]), qx=r4(d['q'][0, 0]), qy=r4(d['q'][0, 1]))
    else:
        W = dict(G2x=r4(d['G2'][0, 0]), G2y=r4(d['G2'][0, 1]), h3x=r4(d['h3'][0, 0]), h3y=r4(d['h3'][0, 1]), rb3=r4(d['rb3'][0]), Lc3=r4(d['Lc3'][0]),
                 b4x=r4(d['b4'][0, 0]), b4y=r4(d['b4'][0, 1]), L4=r4(d['L4'][0]), qx=r4(d['q'][0, 0]), qy=r4(d['q'][0, 1]))
    if TILT: W.update(b4z=r4(d['b4'][0, 2]), qz=r4(d['q'][0, 2]))
    if c['ec'] == 'direct':
        topo, em = 'twoD', 'direct'
        dims = dict(Ax=r4(d['A'][0, 0]), Ay=r4(d['A'][0, 1]), Az=r4(d['A'][0, 2]), wdx=r4(d['wd'][0, 0]), wdy=r4(d['wd'][0, 1]), wdz=r4(d['wd'][0, 2]),
                    Ld=r4(d['Ld'][0]), **W)
    elif c['ec'] == 'body':
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
    if c['wc'] == 'direct': topo = 'twoD'
    F0 = c['F']
    why0 = f"段階2の候補（{c['label']}、選び方: {c['pick']}）。胴幅45 mmのCGに合わせて、畳み姿勢とリンク寸法を同時に多目的最適化した局所解の1つ。"
    why = (f"肘{c['E'][0]:.0f}→{c['E'][1]:.0f}°、手首{c['M'][0]:.0f}→{c['M'][1]:.0f}°、関節角の誤差{F0[0]:.1f}°、最小伝達角{-F0[1]:.0f}°、"
           f"はみ出し（CGメッシュ基準）{F0[2]:.1f} mm、寸法感度2σ {F0[3]:.1f}°（MC ±0.05 mmで95%が{c['mc005']['p95']:.1f}°以内）、"
           f"畳み姿勢の当てはまり{F0[4]:.1f}、最大部品{F0[5]:.0f} mm、ボールジョイント首振り{c['swivel']:.0f}°、羽根の浮かせ{c['fz']:.1f} mm、モーター{c['theta']:.0f}°。")
    occ = mesh['variants']['w45']['occluder']
    if S2E:
        st = c.get('stack', {})
        lay = (f"立体の層チェック：{'合格' if st.get('unplaced', 1) == 0 else '不合格'}（ロッド1は竹串で直線、ロッド4は曲げて層を移る）、"
               f"ガタだけ（蝶番を含む）95%で{c['mc_play']['p95']:.1f}°、±0.2 mmで95%が{c['mc020']['p95']:.1f}°。")
        why = (f"段階{STAGE}（{c['label']}）。手首軸の傾き{np.degrees(d['tw'][0]):.0f}°。" + lay) + why
    elif S2D:
        lay = f"リンクの重なり{c['layers']}層（翼の部分{c['layers_wing']}層）、ガタだけのずれ 95%で{c['mc_play']['p95']:.1f}°、±0.2 mmで95%が{c['mc020']['p95']:.1f}°。"
        why = (f"段階2d（干渉と関節ごとのガタを入れた再計算、{c['label']}）。手首軸の傾き{np.degrees(d['tw'][0]):.0f}°。" + lay) + why
    elif TILT:
        why = (f"案B（手首の軸を尺骨まわりに{np.degrees(d['tw'][0]):.0f}°傾け、畳んだ手を腕の面から外す。ロッド4は両端ロッドエンド）。"
               f"初列風切の付け根のねじれ{np.degrees(d['phi'][0]):.0f}°、畳んだ時の初列の角度ずらし{d['dl'][0]:.0f}°。CTの肘・手首位置からのずれ{c['ct_pos']:.1f} mm、"
               f"手の向きのずれ{c['ct_hand']:.0f}°（選び方: {c['pick']}）。") + why
    else:
        why = why0 + why
    v = dict(id=c['id'], status='candidate', date='2026-11-14' if TILT else '2026-11-13',
             title=(f"段階{STAGE}候補 {c['id']}：{c['label']}" if S2E else f"段階2d候補 {c['id']}：{c['label']}" if S2D else f"案B候補 {c['id']}：{c['label']}（手首軸 傾き）" if TILT else f"段階2候補 {c['id']}：{c['label']}"), topo=topo, wristMount=c['wc'], elbowMount=em,
             why=why, shoulder=dict(axis=[r4(v) for v in d['axis'][0]], Ro=[[r4(v) for v in row] for row in Ro]), body=dict(c=occ['c'], r=occ['r']),
             bodyMesh='w45', featherZ=r4(d['fz'][0]),
             target=dict(AH0=0, AH1=round(float(np.degrees(d['th'][0])), 3), E0=150.0, E1=r4(d['E1'][0]), M0=160.0, M1=r4(d['M1'][0])),
             tilt3d=dict(tS=0, tE=0, tW=0), dims=dims)
    if TILT: v.update(wristTilt=r4(np.degrees(d['tw'][0])), featherTwist=r4(np.degrees(d['phi'][0])), primFold=r4(d['dl'][0]))
    return v, e


def write(p1, entries):
    S, E_ = f'// __{MARK}_START__', f'// __{MARK}_END__\n'
    block = S + '\n' + ''.join('  ' + json.dumps(en, ensure_ascii=False) + ',\n' for en in entries) + '  ' + E_
    if S in p1:
        a = p1.index(S); b = p1.index(E_) + len(E_)
        p1 = p1[:a] + block.lstrip() + p1[b:]
    else:
        p1 = p1.replace('const VERSIONS = [\n', 'const VERSIONS = [\n  ' + block, 1)
    (TOOLS / 'part1.html').write_text(p1, encoding='utf-8')
    (ROOT / 'kingfisher_wing_fold.html').write_text(p1 + (TOOLS / 'part2.html').read_text(encoding='utf-8'), encoding='utf-8')


def compare(c, e):
    out = ROOT / 'results' / 'local' / f"sim_{c['id']}.json"
    subprocess.run(['node', str(TOOLS / 'stage1' / 'dump_path.js'), c['id'], str(out)], cwd=TOOLS, check=True, capture_output=True)
    s = json.load(open(out)); Es = np.array([q['Edeg'] for q in s['steps']]); Ms = np.array([q['Mdeg'] for q in s['steps']])
    u = np.array([q['u'] for q in s['steps']]); f = e['f']
    dE, dM = np.abs(Es - np.interp(u, U, f['E'][0])).max(), np.abs(Ms - np.interp(u, U, f['M'][0])).max()
    return s, dE, dM


def main():
    cands = json.load(open(CAND, encoding='utf-8'))
    mtxt = (ROOT / 'ref' / 'derived' / 'ck_mesh.js').read_text(encoding='utf-8')
    mesh = json.loads(mtxt[mtxt.index('=') + 1:].strip().rstrip(';'))
    entries, ex = [], {}
    for c in cands:
        en, e = entry(c, mesh); entries.append(en); ex[c['id']] = e
    p1 = (TOOLS / 'part1.html').read_text(encoding='utf-8')
    write(p1, entries)
    # verification against the simulator. The simulator picks the open-pose assembly branch closest to the target angles; when two
    # branches are both close it can pick another one than the Python model -> then fix the branch explicitly (ver.pick).
    for c, en in zip(cands, entries):
        s, dE, dM = compare(c, ex[c['id']])
        if max(dE, dM) > 1.0:
            best = None
            for m in range(16):
                en['pick'] = [(m >> i) & 1 for i in range(4)]; write(p1, entries)
                r = compare(c, ex[c['id']])
                if best is None or max(r[1], r[2]) < best[0]: best = (max(r[1], r[2]), en['pick'], r)
            en['pick'] = best[1]; write(p1, entries); s, dE, dM = best[2]
            print(c['id'], 'open-pose branch fixed to', best[1])
        print(c['id'], c['ec'], c['wc'], 'sim vs python: max |dE| %.3f |dM| %.3f, rodTypes %s, sim prot (ellipsoid) %.2f' % (
            dE, dM, ''.join(s['rodTypes']), max(max(q['prot']) for q in s['steps'])))


if __name__ == '__main__':
    main()
