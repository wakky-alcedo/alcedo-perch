"""Stage 3 report: the V8-3 layout re-optimised with the CAD-driven rules (one elbow joint B, ball sockets clear of plates,
shoulder hub) -> pick C1, its CAD (parts, drawings, hole table, interference over the motion, viewer) and the open items.
-> results/local/stage3_<ID>_report.html (embeds the CG-derived candidate figure, CC BY-SA 4.0: stays out of git).
usage: report_stage3.py [ID]   (after results/stage2/run_2f_dh8m.sh and cad.py ID with the same environment)"""
import sys, json, base64, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
R2, R3, RL = ROOT / 'results' / 'stage2', ROOT / 'results' / 'stage3', ROOT / 'results' / 'local'
CID = sys.argv[1] if len(sys.argv) > 1 else 'C1'
J = lambda p: json.load(open(p, encoding='utf-8'))
img = lambda p: 'data:image/png;base64,' + base64.b64encode(p.read_bytes()).decode()
fig = lambda p, cap: f"<figure><img src='{img(p)}' loading='lazy'><figcaption>{cap}</figcaption></figure>" if p.exists() else ''

N, O = R2 / '2f_dh8m', R2 / '2f_dh8'
c = next(x for x in J(N / 'candidates.json') if x['id'] == CID)
v = next(x for x in J(O / 'candidates.json') if x['id'] == 'V8-3')
fbn, fbo = J(N / 'feather_bend_chosen.json')[CID], J(O / 'feather_bend_V8-3_chosen.json')['V8-3']
ld = {CID: J(N / 'loads.json')[CID], 'V8-3': J(O / 'loads.json')['V8-3']}
mi = {CID: J(N / 'mirror_check.json')[CID]['min_clear'], 'V8-3': J(O / 'mirror_check.json')['V8-3']['min_clear']}
cad = J(R3 / CID / 'cad.json')
levels = lambda x: max(x['stack_plan'].values()) - min(x['stack_plan'].values()) + 1

rows = [('関節角の誤差（°、小さいほど良い）', lambda x: x['F'][0], 2),
        ('最小伝達角（°、大きいほど良い）', lambda x: -x['F'][1], 1),
        ('体からのはみ出し（mm）', lambda x: x['F'][2], 2),
        ('±0.05 mm の誤差で最悪のずれ（°）', lambda x: x['F'][3], 2),
        ('畳み当てはまり・平らな羽根', lambda x: x['F'][4], 2),
        ('畳み当てはまり・反らせた羽根（反り約7%）', lambda x: (fbn if x is c else fbo)['fit_bent'], 2),
        ('最大の部品の長さ（mm）', lambda x: x['F'][5], 1),
        ('±0.20 mm の誤差：95%点（°）', lambda x: x['mc020']['p95'], 1),
        ('±0.20 mm の誤差：最大（°）', lambda x: x['mc020']['max'], 1),
        ('±0.20 mm の誤差：外れ（%）', lambda x: x['mc020']['fail'] * 100, 2),
        ('ボールジョイントの首振り・段階2の指標（°）', lambda x: x['swivel'], 1),
        ('最短のレバー（mm）', lambda x: x['lever_min'], 2),
        ('モーターの回転角（°）', lambda x: x['theta'], 1),
        ('左右の翼のすき間（mm）', lambda x: mi[x['id']], 2),
        ('最大のロッド力・ばね込み（N）', lambda x: max(ld[x['id']]['force_max_with_spring_N'].values()), 2),
        ('層の数', levels, 0),
        ('竹串（ロッド1）玉〜玉（mm）', lambda x: x['rods'][0], 1)]
tbl = '<tr><th></th><th>V8-3（新しい規則の前）</th><th>' + CID + '（新しい規則で再計算）</th></tr>' + ''.join(
    f"<tr><td style='text-align:left'>{n}</td><td>{f(v):.{d}f}</td><td>{f(c):.{d}f}</td></tr>" for n, f, d in rows)

ht = ''.join(f"<tr><td style='text-align:left'>{r[0]}</td><td>{r[1]}</td><td>{r[2]:.2f}</td><td>{r[3]:.2f}</td><td>⌀{r[4]:g}</td><td>{r[5]:g}</td><td style='text-align:left'>{r[6]}</td></tr>"
             for r in cad['hole_table'])
st = cad['socket_tilt_deg']; itf = cad['interference']; jg = cad['jigs']

bom = [('3Dプリント部品（片翼）', '上腕骨・尺骨・手・肘レバー・ロッド2・ロッド4、竹串の両端のソケット2個', '6＋2', 'parts/*.stl, *.step（板厚0.9 mm）'),
       ('681 ベアリング（1×3×1 mm）', 'E（尺骨）、G1（肘レバー）、B と U（ロッド2）', '4', ''),
       ('⌀1 mm 鋼ピン', 'E、W、G1、B（玉とロッド2で共用）、U、B4、Q', '7', '長さは未決（下の「未決」）'),
       ('⌀3 mm の玉（⌀1 の穴あき）', 'B（肘レバーの下）、B4（上腕骨）、Q（手）、A（体）', '4', '首振りは下の表'),
       ('竹串 ⌀3 mm', 'ロッド1', f"1（{cad['rods']['rod1_ball_to_ball']:.1f} mm、玉〜玉）", ''),
       ('⌀2 mm 軸', '肩（上腕骨のボスに通す）', '1', ''),
       ('羽根の曲げ治具', f"初列：根元 {jg['pri']['alpha_deg']:g}°・半径 {jg['pri']['R_mm']:g} mm／次列：{jg['sec']['alpha_deg']:g}°・{jg['sec']['R_mm']:g} mm", '2', 'parts/*_jig_pri, *_jig_sec')]
bomh = ''.join(f"<tr><td style='text-align:left'>{a}</td><td style='text-align:left'>{b}</td><td>{n}</td><td style='text-align:left'>{r}</td></tr>" for a, b, n, r in bom)

html = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>段階3 CAD {CID}</title>
<style>
:root{{--bg:#fff;--ink:#1d2430;--mute:#5d6878;--line:#d9dee6;--hi:#fff4d6}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#14181f;--ink:#e6eaf0;--mute:#9aa5b5;--line:#2c3440;--hi:#3a3220}} img{{background:#fff}}}}
:root[data-theme="dark"]{{--bg:#14181f;--ink:#e6eaf0;--mute:#9aa5b5;--line:#2c3440;--hi:#3a3220}}
body{{background:var(--bg);color:var(--ink);font:15px/1.65 system-ui,'Hiragino Sans','Yu Gothic UI',sans-serif;margin:0 auto;padding:16px;max-width:1200px}}
h1{{font-size:1.5em}} h2{{margin-top:2em;border-bottom:1px solid var(--line)}} .mute{{color:var(--mute)}}
.tw{{overflow-x:auto}} table{{border-collapse:collapse;font-size:13px}} td,th{{border:1px solid var(--line);padding:3px 7px;text-align:right}}
th{{background:var(--bg);vertical-align:bottom}} img{{max-width:100%;height:auto}} figure{{margin:1.2em 0}} figcaption{{color:var(--mute);font-size:13px}} li{{margin:.3em 0}}
</style></head><body>
<h1>段階3：V8-3 を部品の形にした結果（{CID}）</h1>
<p class="mute">2026-11-21。数値はすべて計算値で、実測ではありません。最適化は各シードの局所解で、大域最適の保証はありません。</p>

<h2>結論（先に）</h2>
<ul>
<li><b>V8-3 をそのまま部品にすると成り立ちませんでした。</b>CAD にして3つの問題が見つかりました。
<ol>
<li>肘レバーの2つの関節（竹串の玉 B1 とロッド2のピン B2）が 0.75 mm しか離れておらず、別々には作れない。→ 1つの関節 B にまとめる（玉は板の下、ピンは板の上）。</li>
<li>ロッド側の玉ソケット（⌀5 × 高さ3 mm）が、近くの板に当たる。→ 層の規則に「ソケットの高さ＋板の半分＋すき間」を追加。</li>
<li>肘レバーの回転中心が、肩の軸のボスと重なっていた（0.45 mm）。→ 肩のボス（⌀4 × 4 mm）を当たり判定に追加。</li>
</ol>
この3つを入れると V8-3 は層が組めなくなる（3部品が置けない）ので、同じ構成で8シードを再計算しました（V8系の最終集団から開始、200世代、集団300）。</li>
<li><b>解が出たのは8シード中2つで、全確認を通った候補を1つ選びました（{CID}）。</b>V8-3 と比べて、ほぼ同等〜少し悪い程度です（下の表）。
良くなったのは左右のすき間（{mi['V8-3']:.1f} → {mi[CID]:.1f} mm）と層の数（{levels(v)} → {levels(c)}）。
悪くなったのは寸法誤差への強さ（±0.20 mm の95%点 {v['mc020']['p95']:.1f} → {c['mc020']['p95']:.1f}°）と、反らせた羽根の当てはまり（{fbo['fit_bent']:.1f} → {fbn['fit_bent']:.1f}）です。</li>
<li><b>CAD での当たり確認（41姿勢、実物の形、玉・ソケット・ベアリング・ピン込み）は、ほぼ問題なしです。</b>
重なりは尺骨と手の {itf['overlap_mm3'].get('uln-hand', [0])[0]:.4f} mm³ だけでした。手首のピンでつながる2部品で、STL の三角形化の誤差の大きさです。
0.3 mm より近づく組と、左右の翼が1 mm 以内に近づく組はありません。</li>
<li><b>ボールジョイントの首振りは、ソケットの向きを最適にすると B で {st['B1']:.0f}°、ほかは {max(st['B4'], st['Q'], st['A']):.0f}° 以下です。</b>
段階2の指標（{c['swivel']:.1f}°、上限20°）は、玉のスタッドまわりの回転（ボールリンクでは自由）も含むので大きめに出ます。
買ったボールジョイントで確かめるべき数字は「ロッドとスタッドの角度の振れ幅」で、B が ±{st['B1']:.0f}° です。</li>
</ul>

<h2>決めていただきたいこと</h2>
<ul>
<li>{CID} で試作に進んでよいか。V8-3 との差は小さいですが、寸法誤差への強さは2°ほど落ちています。</li>
<li>竹串の体側の玉 A のスタッドの向き（体のフレームへの取り付け方）。計算では、首振りが最小になる向きは {cad['notes'].get('body', '').split('ワールド）')[-1]}（ワールド、玉から体の取付へ）です。</li>
</ul>

<h2>V8-3 との比較</h2>
<div class="tw"><table>{tbl}</table></div>
<p class="mute">「新しい規則」：肘の関節を1つにまとめる、玉ソケットと板のすき間、肩のボス。V8-3 の数値は新しい規則の前のものです（新しい規則では層が組めません）。
反らせた羽根は両方とも反り約7%（初列：根元15°・半径85 mm、次列：15°・70 mm）。</p>

<h2>CAD で直したこと（再計算の後）</h2>
<ul>
<li>玉ソケットの向き：開いた姿勢のスタッドの向きに合わせていたので、畳むとソケットが片側に最大22°傾き、縁が肘レバーの板に 0.49 mm³ 食い込んでいました。
全行程で傾きが最小になる向き（ミニマックス）に変え、B の傾きは ±{st['B1']:.1f}° になりました。</li>
<li>ソケットのスタッド側：面取り（傾き＋20°の円すい）を付け、上面を 0.4 mm 下げました。玉は入口の円すいと玉が接するふち（半径約1.4 mm、高さ約0.5 mm）で保持されるので、抜けにくさは変わりません。
直す前は、畳んだときに上面の内側のふちが板に 0.13 mm まで近づいていました。</li>
<li>体側の玉 A：スタッドの向きは自由に決められるので、ロッドから45°以上離れる向きの中で首振りが最小のもの（{st['A']:.1f}°）を選びました。</li>
</ul>

<h2>部品表（片翼。両翼ではこの2倍）</h2>
<div class="tw"><table><tr><th>部品</th><th>場所</th><th>数</th><th>備考</th></tr>{bomh}</table></div>

<h2>穴の位置（各部品の座標、mm）</h2>
<div class="tw"><table><tr><th>部品</th><th>関節</th><th>x</th><th>y</th><th>穴</th><th>傾き（°）</th><th>種類</th></tr>{ht}</table></div>
<p class="mute">{cad['notes'].get('hum', '')}。尺骨の W の穴は手首の軸の傾き {c['tw']:.1f}° に合わせて斜めに開けます。
肘レバーの B1 と B2 は同じ穴（1本のピンの下に玉、上にロッド2のベアリング）。</p>
{fig(R3 / CID / f'{CID}_drawings.png', '各部品の外形と穴（部品の座標、mm）。')}

<h2>図</h2>
{fig(RL / f'cand_{CID}.png', f'{CID} 畳んだ状態：青＝CGの畳んだ翼、橙＝モデルの羽根（平ら）、赤＝骨、黒＝リンク。')}
{fig(N / f'stack_{CID}.png', f'{CID} の層配置。上段：上から見た図（色＝高さ）。下段：横から見た図。')}
{fig(N / f'mirror_{CID}.png', f'{CID}：左右の翼が最も近づく姿勢（すき間 {mi[CID]:.1f} mm）。')}
<p>3D表示：results/stage3/{CID}/{CID}_viewer.html（ブラウザで開く。スライダーで開閉、左の翼・金具・中心面の表示切り替え）。</p>

<h2>未決・やっていないこと</h2>
<ul>
<li>両肩をモーター1つで動かす駆動（ギアかリンクか、モーターの置き場所）。今の CAD は片翼の肩の軸までです。</li>
<li>羽根の取り付け方（手と尺骨の板に羽根の根元をどう固定するか）。</li>
<li>3Dプリントの公差（⌀1 の穴の締まり具合、⌀3 のベアリング穴、ソケットへの玉の入れやすさ）。試し刷りで決めます。</li>
<li>2倍スケールでの試作（動きの確認用）。</li>
<li>ボールジョイントの購入と首振りの実測（B で ±{st['B1']:.0f}° 必要）。</li>
<li>ピンの長さ、ロッド4の玉ピン（板から玉の中心まで 3.4〜5.9 mm と長い）の曲がり・強度、竹串とソケットの接着。</li>
<li>シミュレータ（wing_linkage_simulator.html）はまだ {CID} を入れていません。</li>
</ul>
<p class="mute">体の形の参照：「Common Kingfisher」Alejandro Guzmán（Sketchfab）CC BY-SA 4.0 を縮尺・変形して使用。図に体の形が含まれるため、このページも CC BY-SA 4.0 です。</p>
</body></html>"""
out = RL / f'stage3_{CID}_report.html'
out.write_text(html, encoding='utf-8')
print(out, out.stat().st_size // 1024, 'KB')
