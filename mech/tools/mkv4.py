"""build the v4 VERSIONS entry from the optimiser output and splice it into part1.html as test_v4.html"""
import json, sys, os
P = json.load(open('v4_pose.json'))
EF = os.environ.get('EF', 'v4_elbow_all.json'); EL = json.load(open(EF)); el = EL[int(sys.argv[1])]['d'] if len(sys.argv) > 1 else min(EL, key=lambda o: o['score'])['d']
WR = json.load(open('v4_wrist_all.json')); wr = WR[int(sys.argv[2])]['d'] if len(sys.argv) > 2 else min(WR, key=lambda o: o['score'])['d']
r = lambda v: round(float(v), 4)
W = dict(G2x=r(wr['G'][0]), G2y=r(wr['G'][1]), h3x=r(wr['h'][0]), h3y=r(wr['h'][1]), rb3=r(wr['rb1']), Lc3=r(wr['Lc1']),
         b4x=r(wr['b2'][0]), b4y=r(wr['b2'][1]), L4=r(wr['L2']), qx=r(wr['w'][0]), qy=r(wr['w'][1]))
if 'A' in el:
    dims = dict(Ax=r(el['A'][0]), Ay=r(el['A'][1]), Az=r(el['A'][2]), G1x=r(el['G'][0]), G1y=r(el['G'][1]), rb1=r(el['rb1']), Lc1=r(el['Lc1']),
                b2x=r(el['b2'][0]), b2y=r(el['b2'][1]), L2=r(el['L2']), wx=r(el['w'][0]), wy=r(el['w'][1]), **W)
else:
    dims = dict(G1x=r(el['G'][0]), G1y=r(el['G'][1]), G1z=r(el['G'][2]), h1x=r(el['h'][0]), h1y=r(el['h'][1]), h1z=r(el['h'][2]),
                rb1=r(el['rb1']), Lc1=r(el['Lc1']), b2x=r(el['b2'][0]), b2y=r(el['b2'][1]), b2z=r(el['b2'][2]), L2=r(el['L2']), wx=r(el['w'][0]), wy=r(el['w'][1]), **W)
entry = dict(id='v4', status='current', date='2026-11-12', title='畳んだ姿が体の輪郭に沿う3D版', topo='twoB' if 'A' in el else 'two3', wristMount='ulna', elbowMount='hum' if 'A' in el else 'body',
             why='__WHY__', shoulder=dict(axis=[r(v) for v in P['axis']], Ro=[[r(v) for v in row] for row in P['Ro']]),
             body=dict(c=[-9, -18, -6], r=[12, 32, 12]),
             target=dict(AH0=0, AH1=round(P['theta'], 2), E0=P['E0'], E1=round(P['E1'], 2), M0=P['M0'], M1=round(P['M1'], 2)),
             tilt3d=dict(tS=0, tE=0, tW=0), dims=dims)
json.dump(entry, open('v4_entry.json', 'w'))
print(json.dumps(entry)[:600])
print('elbow', {k: el[k] for k in ('t1', 't2', 'resmax', 'swing', 'hidden')}, 'br', el['br'])
print('wrist', {k: wr[k] for k in ('t1', 't2', 'resmax', 'swing', 'hidden')}, 'br', wr['br'])

p1 = open('part1.html').read()
js = 'const VERSIONS = [\n  ' + json.dumps(entry, ensure_ascii=False) + ','
p1 = p1.replace('const VERSIONS = [', js, 1).replace("{id:'v3', status:'current'", "{id:'v3', status:'superseded'")
open('test_v4.html', 'w').write(p1 + open('part2.html').read())
