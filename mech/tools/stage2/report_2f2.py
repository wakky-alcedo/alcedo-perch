"""Stage-2f report, second part (2026-11-16): full post-check gate, candidates chosen by the user's priorities, no-bellcrank runs,
bowed feathers -> results/local/stage2_2f2_report.html (embeds CG-derived figures, CC BY-SA 4.0: stays out of git).
usage: MIDLINE=1 STACK_BUD_JOINT=8 STACK_NSUB=24 report_2f2.py
       (after gate_2f.py, analyze.py 45 2e, check_2e.py, crossing_check.py, mirror_check.py, loads.py 2e, feather_bend.py,
        stack_figs.py, cand_figs.py 2e, mirror_figs.py F1 ..)"""
import sys, json, base64, pathlib, collections, numpy as np
INTERIM = 'interim' in sys.argv; sys.argv = sys.argv[:1]
from problem import LAYER_T, GAP
from stack import BUD_BODY, BUD_WING, BUD_JOINT, R_SKEW, NSUB

ROOT = pathlib.Path(__file__).resolve().parents[2]
R2, RL = ROOT / 'results' / 'stage2', ROOT / 'results' / 'local'
Fd = R2 / '2f'
J = lambda p: json.load(open(p, encoding='utf-8'))
D = J(Fd / 'candidates.json'); summ = J(Fd / 'summary.json')['configs']
CHK, LD, MC, CR, FB = (J(Fd / f'{n}.json') for n in ('check_2e', 'loads', 'mirror_check', 'crossing_check', 'feather_bend'))
GT = J(Fd / 'gate.json')
img = lambda p: 'data:image/png;base64,' + base64.b64encode(p.read_bytes()).decode()
fig = lambda p, cap: f"<figure><img src='{img(p)}' loading='lazy'><figcaption>{cap}</figcaption></figure>" if p.exists() else ''
mc = lambda m: f"{m['p95']:.1f} / {m['max']:.1f}"
NAME = {'hum': '上腕骨', 'uln': '尺骨', 'hand': '手', 'bc1': 'ベルクランク', 'rod1': 'ロッド1（竹串）', 'rod2': 'ロッド2', 'rod4': 'ロッド4'}
pair = lambda k: '–'.join(NAME.get(p, p) for p in k.split('-'))
PICK = {'repeat': '再現性重視', 'shape': '形重視', 'open': '開きやすさ重視'}
CFGN = {'A:hum/direct': '案A・ベルクランク1個', 'B:hum/direct': '案B・ベルクランク1個', 'B:direct/direct': '案B・ベルクランク0個',
        'A:direct/direct': '案A・ベルクランク0個'}

# ---- gate statistics
st = collections.defaultdict(collections.Counter)
for r in GT:
    s = st[r['C']]; s['n'] += 1
    if r['unplaced']: s['層の配置が不成立'] += 1
    elif r['cross_min'] < 0: s['部品が交差'] += 1
    elif r['mirror_min'] < 0: s['左右が当たる'] += 1
    else: s['通過'] += 1
gate_rows = ''.join(f"<tr><td>{CFGN.get(k, k)}</td><td>{v['n']}</td><td>{v['通過']}</td><td>{v['層の配置が不成立']}</td><td>{v['部品が交差']}</td>"
                    f"<td>{v['左右が当たる']}</td></tr>" for k, v in sorted(st.items()))
crossfail = collections.Counter(r['cross_pair'] for r in GT if not r['unplaced'] and r['cross_min'] < 0 and 'hum/direct' in r['C'])
Z = [r for r in GT if 'direct/direct' in r['C'] and not r['unplaced']]
zbest = max(Z, key=lambda r: r['cross_min']) if Z else None
zpair = collections.Counter(r['cross_pair'] for r in Z)
zmc = np.array([r['mc020'] for r in Z]); zfit = np.array([r['fit_bent'] for r in Z]); zprot = np.array([r['F'][2] for r in Z])
zerr = np.array([r['F'][0] for r in Z]); zt = np.array([-r['F'][1] for r in Z])
hd = [r for r in GT if 'hum/direct' in r['C'] and r['pass']]
hmc = np.array([r['mc020'] for r in hd])


def thick(plan):
    lv = [v for k, v in plan.items() if k != 'rod1']
    return (max(lv) - min(lv)) * LAYER_T + (LAYER_T - GAP)


def cells(c):
    i = c['id']; F = c['F']; ch = CHK[i]; ld = LD[i]; sp = ld['spring']; fm = ld['force_max_with_spring_N']; fb = FB[i]
    cr = min(CR[i].values(), key=lambda t: t['clear'])['clear'] if CR[i] else 1
    dsg, cfg, pk = c['pick'].split(':')[0], c['pick'].split(':')[1], c['pick'].split(':')[-1]
    st_ = max(ch['studs_mm'].items(), key=lambda kv: kv[1])
    return [i, f"案{dsg}", PICK.get(pk, pk), f"{F[0]:.1f}", f"{-F[1]:.0f}", f"{c['gain_max']:.2f}",
            mc(c['mc_play']), mc(c['mc005']), mc(c['mc020']), f"{fb['fit_flat']:.1f} → <b>{fb['fit_bent']:.1f}</b>",
            f"{F[2]:.1f}", f"{F[3]:.2f}", f"{MC[i]['min_clear']:.1f}", f"{cr:.2f}", f"{thick(ch['plan']):.1f}",
            f"{pair(st_[0])} {st_[1]:.1f}", f"{max(c['rods']):.0f}", f"{c['lever_min']:.1f}",
            ' / '.join(f"{fm[k]:.2f}" if k in fm else '–' for k in ('Lc1', 'L2', 'L4')) + (f" （直結 {fm['Ld']:.2f}）" if 'Ld' in fm else ''), f"{sp['tau_E']:.2f} / {sp['tau_W']:.2f}", f"{sp['motor_extra_Nmm']:.1f}"]


HEAD = ['案', '設計', '選び方', '関節角誤差 °', '最小伝達角 °', 'ゲイン最大 °/°', 'ガタだけ 95% / 最大 °', '±0.05 mm 95% / 最大 °', '±0.20 mm 95% / 最大 °',
        '畳み当てはまり 平ら → 反らせる', 'はみ出し mm', '寸法感度 2σ °', '左右のすき間 mm', '部品交差の余裕 mm', '体側の重ね厚さ mm', '最長のピン mm',
        '最長ロッド mm', '最短レバー mm', '最大力 N ロッド1 / 2 / 4', 'ばね 肘 / 手首 N·mm', 'モーター追加 N·mm']
cand_tbl = '<tr>' + ''.join(f'<th>{h}</th>' for h in HEAD) + '</tr>' + ''.join('<tr>' + ''.join(f'<td>{v}</td>' for v in cells(c)) + '</tr>' for c in D)


def fbrow(c):
    f = FB[c['id']]; s, p = f['sec'], f['pri']
    R = lambda v: '平ら' if v is None else f'{v:.0f}'
    return (f"<tr><td>{c['id']}</td><td>{s['alpha_deg']:.0f}</td><td>{R(s['R_mm'])}</td><td>{100 * s['camber']:.0f}%</td><td>{s['rub_mm']:.1f}</td>"
            f"<td>{p['alpha_deg']:.0f}</td><td>{R(p['R_mm'])}</td><td>{100 * p['camber']:.0f}%</td><td>{p['rub_mm']:.1f}</td>"
            f"<td>{f['flat']['bury_max']:.1f} → {f['bent']['bury_max']:.1f}</td><td>{f['flat']['float_max']:.1f} → {f['bent']['float_max']:.1f}</td>"
            f"<td>{f['fit_flat']:.1f} → {f['fit_bent']:.1f}</td></tr>")


fb_tbl = ''.join(fbrow(c) for c in D)
gain = [(1 - FB[c['id']]['fit_bent'] / FB[c['id']]['fit_flat']) * 100 for c in D]
by = {c['id']: c for c in D}
def worst3(t):
    return '、'.join(f"{pair(k)} {v['clear']:.2f}" for k, v in sorted(t.items(), key=lambda kv: kv[1]['clear'])[:3])


cr_rows = ''.join(f"<tr><td>{i}</td><td style='text-align:left'>{worst3(t)}</td></tr>" for i, t in CR.items())
zs = summ.get('B:direct/direct', {}); za = summ.get('A:direct/direct', {})

ZP = [r for r in Z if r['pass']]
zpair_f = collections.Counter(r['cross_pair'] for r in Z if not r['pass'])
zpmc = np.array([r['mc020'] for r in ZP]); zpprot = np.array([r['F'][2] for r in ZP]); zperr = np.array([r['F'][0] for r in ZP])
zpfit = np.array([r['fit_bent'] for r in ZP])
hperr = np.array([r['F'][0] for r in hd]); hpprot = np.array([r['F'][2] for r in hd]); hpfit = np.array([r['fit_bent'] for r in hd])
z_html = f"""
<p>継ぎ手の厚さ枠を {BUD_JOINT:.0f} mm に緩め、層の判定を直して計算し直しました（案B、8シード）。今回は {zs.get('seeds_with_feasible', 0)}/{zs.get('seeds', 8)} シードで解が出ました。
確認の手順にかけた {len(Z)} 解のうち、{len(ZP)} 解がすべての確認を通りました。案Aは前回の計算のままで、解はありません（{za.get('seeds_with_feasible', 0)}/{za.get('seeds', 8)}）。</p>
<ul>
<li><b>直したこと：</b>前回は、傾いて通る竹串とロッド4を短い区間に分けて層を判定していました。そのため、尺骨のレバーの玉の近くで当たりを見逃していました（独立の交差チェックで −0.3〜−1.1 mm）。
 今回は、傾いた棒を各層の高さの範囲で切り取って、平らな部品と直接比べるようにしました。継ぎ手の周り 3 mm は、交差チェックと同じく除外します。
 それでも、層の配置は通ったのに交差チェックで当たった解が {len(Z) - len(ZP)} 解残りました。
 食い込みは {min(r['cross_min'] for r in Z):.2f}〜{max(r['cross_min'] for r in Z if not r['pass']):.2f} mm と小さく（{'、'.join(pair(k) + f' {v}解' for k, v in zpair_f.items())}）、確認の手順で落としています。
 玉継手の近くで、2つの判定の除外範囲が少し違うためと考えています（未確認）。</li>
<li><b>通った解の範囲（1個の構成で通った解との比較）：</b>
 <ul>
 <li>関節角誤差 {zperr.min():.1f}〜{np.median(zperr):.1f}°（最小〜中央値）。1個の構成は {hperr.min():.1f}〜{np.median(hperr):.1f}°。</li>
 <li>±0.20 mm の 95%：{zpmc.min():.1f}〜{np.median(zpmc):.1f}°。1個の構成は {hmc.min():.1f}〜{np.median(hmc):.1f}°。</li>
 <li>反らせた羽根での当てはまり {zpfit.min():.1f}〜{np.median(zpfit):.1f}。1個の構成は {hpfit.min():.1f}〜{np.median(hpfit):.1f}。</li>
 <li>はみ出し {zpprot.min():.1f}〜{np.median(zpprot):.1f} mm。1個の構成は {hpprot.min():.1f}〜{np.median(hpprot):.1f} mm。</li>
 </ul></li>
<li><b>はみ出しが約 10 mm と大きいままです。</b>体側のレバーと、竹串の体側の玉が胴から出ます。肘を直接ロッドで引くには、体側に長いレバーが要るためです。
 継ぎ手の厚さ枠を緩めても、この量はほとんど変わりませんでした。</li>
<li>左右のすき間は 9〜13 mm あります。ベルクランクが中心線の近くに来ないためです。</li>
<li>部品は少なくなります（ベルクランク・ロッド2がない）。部品の評価値は 13 前後で、1個の構成（17〜23）より小さい値です。</li>
</ul>"""

html = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>段階2f 候補の確認</title>
<style>
:root{{--bg:#fff;--ink:#1d2430;--mute:#5d6878;--line:#d9dee6;--hi:#fff4d6;--alt:#eef3fb;--bad:#c0392b}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#14181f;--ink:#e6eaf0;--mute:#9aa5b5;--line:#2c3440;--hi:#3a3220;--alt:#1d2633;--bad:#ff7b6b}} img{{background:#fff}}}}
body{{background:var(--bg);color:var(--ink);font:15px/1.65 system-ui,'Hiragino Sans','Yu Gothic UI',sans-serif;margin:0 auto;padding:16px;max-width:1200px}}
h1{{font-size:1.5em}} h2{{margin-top:2em;border-bottom:1px solid var(--line)}} .mute{{color:var(--mute)}} .bad{{color:var(--bad)}}
.tw{{overflow-x:auto}} table{{border-collapse:collapse;font-size:13px;white-space:nowrap}} td,th{{border:1px solid var(--line);padding:3px 7px;text-align:right}}
th{{background:var(--bg);white-space:normal;min-width:4.5em;vertical-align:bottom}} td:nth-child(2),td:nth-child(3){{text-align:left}}
img{{max-width:100%;height:auto}} figure{{margin:1.2em 0}} figcaption{{color:var(--mute);font-size:13px}} li{{margin:.3em 0}}
</style></head><body>
<h1>段階2f（続き）：すべての確認を通った候補と、反らせた羽根</h1>
<p class="mute">2026-11-17。数値はすべて計算値で、実測ではありません。
各シードの局所的なパレート解なので、大域最適の保証はありません。</p>

<h2>結論（先に）</h2>
<ul>
<li><b>すべての確認を通った候補は7つです（F1〜F7）。</b>ベルクランク0個の構成（F2〜F4）が、今回初めて確認を通りました。確認したのは次の5つです。
 <ul>
 <li>層の配置（全41姿勢）</li>
 <li>部品どうしの交差（竹串の実際の太さで判定）</li>
 <li>左右の翼の干渉</li>
 <li>寸法誤差のモンテカルロ</li>
 <li>反らせた羽根での当てはまり</li>
 </ul>
 選び方は、ご指定の優先順位（開きやすさ、畳んだときと開いたときの形、動きの滑らかさと再現性）に合わせました。
 案A・ベルクランク1個では、3通りの選び方がどれも同じ解（F1）になりました。</li>
<li><b>ベルクランク0個（F2〜F4）は、動きの精度と再現性で1個の構成より良く、はみ出しで劣ります。</b>
 <ul>
 <li>関節角誤差 1.5〜1.9°（1個の構成の候補は 2.9〜3.8°）。</li>
 <li>±0.20 mm の 95%：6.2〜7.4°（同 8.6〜9.9°）。</li>
 <li>はみ出し 11〜12 mm（同 1.5〜2 mm）。体側のレバーと玉が胴から出ます。</li>
 <li>反らせた羽根での当てはまりは、F3 が 12.5 で1個の構成と同程度、F2・F4 は 14.7 でやや劣ります。</li>
 <li>部品が2つ（ベルクランク、ロッド2）減ります。左右のすき間は 9〜13 mm と大きく余裕があります。</li>
 </ul></li>
<li><b>羽根を軸に沿って反らせると、畳み当てはまりが {min(gain):.0f}〜{max(gain):.0f}% 良くなります。</b>
 <ul>
 <li>根元の角度は 15〜35°、曲げ半径は 35〜70 mm です。</li>
 <li>開いた翼では、反りの高さが羽根の長さの 9〜18% になります。本物の翼より強く反っています。</li>
 <li>畳む途中で、隣り合う羽根の高さの差が 2〜4 mm 変わります。羽根の間隔か、羽根のしなりで吸収する必要があります。</li>
 </ul></li>
<li>最適化の中の層の判定を2か所直しました。
 <ul>
 <li>傾いて通る竹串・ロッド4と平らな部品の当たりを、厳密に判定するようにしました。</li>
 <li>玉継手のまわり 5 mm では、継ぎ手の厚さ枠（{BUD_JOINT:.0f} mm）を使うようにしました。前は翼の厚さ枠（{BUD_WING:.0f} mm）が掛かり、玉のすぐ横で成立しない場合がありました。</li>
 </ul>
 最適化の結果はそのまま使わず、前線の解すべてに後段の確認をかけてから選んでいます（下の「確認の手順」）。</li>
<li>シミュレータと Python の関節角の差は、7案とも 0.33° 以下です。</li>
</ul>

<h2>決めていただきたいこと</h2>
<ol>
<li>段階3（CAD）に進める候補。
 <ul>
 <li><b>はみ出しを 10 mm 程度まで許せるなら、F3（ベルクランク0個・形重視）を勧めます。</b>関節角誤差 1.9°、±0.20 mm の 95% が 7.4° で、当てはまりも 1個の構成と同程度です。部品も少なくなります。</li>
 <li>F2・F4（ベルクランク0個）：F3 より精度と再現性が少し良く（±0.20 mm で 6.2〜6.7°）、当てはまりがやや劣ります。</li>
 <li>はみ出しを抑えたいなら F1（案A・ベルクランク1個、手首が平面で作りやすい）です。ただし ±0.20 mm の 95% が 9.0° で、F2〜F4 より悪くなります。</li>
 <li>F5〜F7（案B・ベルクランク1個）は、F1 に比べて良い点が少ないです。</li>
 </ul>
 「ところどころ出っ張るのは仕方ない」とのことでしたが、約 10 mm は局所的な出っ張りとしては大きい量です。許せるかどうかをシミュレータで見て判断してください。</li>
<li>反らせた羽根の反りの量。
 <ul>
 <li>当てはまりが最良になる値では、開いた翼が強く反ります（9〜18%）。</li>
 <li>開いた形を優先するなら反りを弱めます（例：半径を大きくする）。当てはまりの改善はその分小さくなります。</li>
 <li>反りの量と当てはまりの関係を一覧にすることもできます（未計算）。</li>
 </ul></li>
</ol>

<h2>候補</h2>
<p class="mute">選び方（関節角誤差 4° 以下は共通）：
<br>・再現性重視＝反らせた当てはまり 15 以下の中で、±0.20 mm の 95% が最小。
<br>・形重視＝±0.20 mm の 95% が 9° 以下の中で、反らせた当てはまりが最良。
<br>・開きやすさ重視＝±0.20 mm が 9.5° 以下・当てはまり 15 以下の中で、最小伝達角が最大。
<br>選ぶときのモンテカルロは 200 回、表の値は 400 回で計算し直したものです。
<br>・部品交差の余裕＝平面図で重なる部品どうしの、高さ方向のすき間の最小値（0 は接していない、ちょうどの状態）。
<br>・最長のピン＝層をまたぐピン・玉継手の軸の長さ。</p>
<div class="tw"><table>{cand_tbl}</table></div>
<p class="mute">最大力は、ばねを最小必要量の2倍にしたとき、重力の向きを全方向に振った場合の最大値。質量は前腕＋次列風切 1.5 g、手＋初列 1.0 g（推定）。</p>

<h2>確認の手順と通過数</h2>
<ol>
<li>前線の解のうち、関節角誤差・寸法感度・当てはまりなどで足切りした解に、±0.05 / ±0.20 mm のモンテカルロ（各 200 回）をかけました。</li>
<li>その全員に、次の確認をかけました（gate_2f.py）。
 <ul>
 <li>層の配置：全41姿勢、探索の上限なし。</li>
 <li>部品どうしの交差：竹串は直径 {2 * R_SKEW:.0f} mm で判定。</li>
 <li>左右の干渉：鏡写しにして確認。</li>
 <li>ゲイン（モーター1°あたりの関節角）：最大値と、姿勢ごとの変化。</li>
 <li>反らせた羽根での当てはまり：反りは粗い格子から選びます。</li>
 </ul></li>
<li>通った解の中から、上の3通りの選び方で選びました。</li>
</ol>
<div class="tw"><table><tr><th>構成</th><th>確認した解</th><th>通過</th><th>層の配置が不成立</th><th>部品が交差</th><th>左右が当たる</th></tr>{gate_rows}</table></div>
<p class="mute">1個の構成で交差した組：{'、'.join(pair(k) + f' {v}' for k, v in crossfail.most_common())}。
厚さ枠は体の部分 {BUD_BODY:.0f} mm、翼の部分 {BUD_WING:.0f} mm、翼側の継ぎ手のところ {BUD_JOINT:.0f} mm（ご了承いただいた緩和）。</p>
<div class="tw"><table><tr><th>案</th><th>交差の余裕が小さい組（mm）</th></tr>{cr_rows}</table></div>

<h2>ベルクランク0個の構成（肘・手首とも直結ロッド）</h2>
{z_html}

<h2>反らせた羽根</h2>
<p>平らな羽根のまま畳むと、羽根は丸い胴を弦のように横切ります。根元と先は胴の表面近くにありますが、羽根の中ほどは胴の中に 3〜5 mm 埋まります。
そこで、羽根の軸と翼の面の法線を含む面の中で、羽根を次の形に曲げることにしました。</p>
<p style="text-align:center">z(r) = r·tan α − r² / (2R)</p>
<ul>
<li>r：羽根の根元から軸に沿った距離。</li>
<li>z：背側（上）への高さ。</li>
<li>α：根元の角度（印刷した根元の取り付け部で決まる）。</li>
<li>R：ジグの曲げ半径。</li>
</ul>
<p>次列風切・初列風切のそれぞれで α と R を共通にし、畳み当てはまりが最良になる組を格子から選びました。</p>
<div class="tw"><table><tr><th>案</th><th>次列 α °</th><th>次列 R mm</th><th>次列 反り</th><th>次列 隣との高さ差の変化 mm</th>
<th>初列 α °</th><th>初列 R mm</th><th>初列 反り</th><th>初列 隣との高さ差の変化 mm</th><th>最大の埋まり mm 平ら → 反らせる</th><th>最大の浮き mm</th><th>当てはまり</th></tr>{fb_tbl}</table></div>
<p class="mute">表の見方：
<br>・反り＝開いた翼での、根元と先を結ぶ線からの最大の高さ ÷ 羽根の長さ（次列 40 mm、初列 50 mm として計算）。
<br>・隣との高さ差の変化＝平面図で重なる隣の羽根との高さの差が、畳む途中でどれだけ変わるか（平らなら 0）。</p>
<ul>
<li>製作は、平らに印刷してから温め、ジグで曲げる前提です。ジグは半径 R の円筒（または R の型）で、根元の角度 α は取り付け部で付けます。
 PLA / PETG では、温めて曲げたあと少し戻ります。ジグは狙いより小さい半径で作り、戻り量を試し曲げで測る必要があります（未実測）。</li>
<li>反らせると胴への埋まりは減りますが、根元側が浮くことがあります（最大 0.4〜3.2 mm）。反りの量を変えると、浮きと埋まりの釣り合いが変わります。</li>
<li>羽根の重なりの順番（上下）と、畳むときに羽根どうしがこすれる量は、段階3のCADで確認します。</li>
</ul>

<h2>層の配置と左右の干渉</h2>
{''.join(fig(Fd / f"stack_{c['id']}.png", f"{c['id']}（案{c['dsg']}）の実高さの層配置。上段：上から見た図（色＝高さ）。下段：横から見た図（黒の縦線＝ピン・玉継手の軸）。") for c in D)}
{''.join(fig(Fd / f"mirror_{c['id']}.png", f"{c['id']}：左右の翼が最も近づく姿勢（すき間 {MC[c['id']]['min_clear']:.1f} mm、{pair('-'.join(MC[c['id']]['pair']))}）。") for c in D)}

<h2>畳んだ姿勢（平らな羽根）</h2>
{''.join(fig(RL / f"cand_{c['id']}.png", f"{c['id']}（案{c['dsg']}）畳んだ状態：青＝CGの畳んだ翼、橙＝モデルの羽根（平ら）、赤＝骨、黒＝リンク。") for c in D)}

<h2>計算結果（構成ごと）</h2>
<div class="tw"><table><tr><th>構成</th><th>解ありのシード</th><th>全体の前線に残った数</th></tr>
{''.join(f"<tr><td>{v['label']}</td><td>{v['seeds_with_feasible']}/{v['seeds']}</td><td>{v['n_in_global_front']}</td></tr>" for v in summ.values())}</table></div>
{fig(Fd / 'pareto.png', 'パレート前線（6目的のうち2つずつ）。')}

<h2>やっていないこと・限界</h2>
<ul>
<li>1つのモーターで左右の上腕骨をどう動かすか（駆動の取り回し）は、まだモデルに入っていません。段階3のCADで確認します。</li>
<li>強度（FEM）はまだです。</li>
<li>反らせた羽根は、当てはまりの評価だけに入れました。最適化の中では平らな羽根のままです。反らせる前提で最適化し直せば、当てはまりはもう少し良くなる可能性があります（未計算）。</li>
<li>層の配置と交差は、姿勢を 41 個に区切って確認しています。姿勢の間で一瞬当たる可能性は CAD で確認します。</li>
<li>シミュレータは羽根を平らに、ロッドをまっすぐに描きます。表示は片側の翼だけです。</li>
</ul>
<p class="mute">体の形の参照：「Common Kingfisher」Alejandro Guzmán（Sketchfab）CC BY-SA 4.0 を縮尺・変形して使用。図に体の形が含まれるため、このページも CC BY-SA 4.0 です。</p>
</body></html>"""
out = RL / 'stage2_2f2_report.html'
out.write_text(html, encoding='utf-8')
print(out, out.stat().st_size // 1024, 'KB', dict(st))
