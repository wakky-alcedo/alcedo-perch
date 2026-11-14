"""Stage-2e report (real-height layer plan in the optimisation, design A vs B, bellcrank count) as one self-contained HTML page
-> results/local/stage2_2e_report.html. It embeds figures with the CG body shape (CC BY-SA 4.0), so it stays out of git.
usage: report_2e.py   (after analyze.py 45 2e, loads.py 2e, check_2e.py, recheck_2d.py, cand_figs.py 2e, stack_figs.py)"""
import sys, json, glob, base64, pathlib, numpy as np
sys.argv = sys.argv[:1]
from problem import LAYER_T, GAP
from mech import HINGE_PLAY
from stack import BUD_BODY, BUD_WING, R_SKEW

ROOT = pathlib.Path(__file__).resolve().parents[2]
R2, RL = ROOT / 'results' / 'stage2', ROOT / 'results' / 'local'
E = R2 / '2e'
D = json.load(open(E / 'candidates.json', encoding='utf-8'))
summ = json.load(open(E / 'summary.json', encoding='utf-8'))['configs']
CHK = json.load(open(E / 'check_2e.json', encoding='utf-8'))
LD = json.load(open(E / 'loads.json', encoding='utf-8'))
RE = json.load(open(E / 'recheck_2d.json', encoding='utf-8'))
D2 = {c['id']: c for c in json.load(open(R2 / '2d' / 'candidates.json', encoding='utf-8'))}
D8H = json.load(open(E / 'd8_hinge_mc.json', encoding='utf-8'))['D8_with_hinge']   # D8 under the same play model (hinge play included)
img = lambda p: 'data:image/png;base64,' + base64.b64encode(p.read_bytes()).decode()
by = {c['id']: c for c in D}
REC = next(c['id'] for c in D if c['pick'] == 'B:robust-fit')
ALT = next(c['id'] for c in D if c['pick'] == 'A:robust-fit')
rec, alt = by[REC], by[ALT]
mc = lambda m: f"{m['p95']:.1f} / {m['max']:.1f}"

# graded follow-up runs of the two-bellcrank configuration (seeds >= 100)
grad = []
for f in sorted(glob.glob(str(R2 / 'runs_2e' / '*-hum_hum_w45_s10?.npz'))):
    z = np.load(f, allow_pickle=True); G = z['G']; cv = np.maximum(G, 0).sum(1); i = int(np.argmin(cv))
    grad.append(dict(name=pathlib.Path(f).stem, cv=float(cv[i]), stack=float(G[i, -1]), lever=float(G[i, 2]), res=float(G[i, 8]), prot=float(G[i, 7])))
g_ok = [g for g in grad if g['stack'] <= 0]


def thick(plan):
    lv = [v for k, v in plan.items() if k != 'rod1']
    return (max(lv) - min(lv)) * LAYER_T + (LAYER_T - GAP)


def row(c):
    F = c['F']; ch = CHK[c['id']]; ld = LD[c['id']]; fm = ld['force_max_with_spring_N']; sp = ld['spring']
    r4 = ch.get('rod4', {})
    cells = [c['id'], c['dsg'], c['pick'].replace('best-A:hum/direct', '総合（A）').replace('best-B:hum/direct', '総合（B）'), f"{c['tw']:.0f}",
             mc(c['mc_play']), mc(c['mc005']), mc(c['mc020']),
             f"{F[0]:.1f}", f"{-F[1]:.0f}", f"{F[2]:.1f}", f"{F[3]:.2f}", f"{F[4]:.1f}", f"{c['w40']['fit']:.1f} / {c['w50']['fit']:.1f}",
             f"{c['ct_pos']:.1f}", f"{c['ct_hand']:.0f}", f"{F[5]:.0f}", f"{max(c['rods']):.0f}", f"{c['lever_min']:.1f}", f"{c['gain_max']:.2f}",
             f"{thick(ch['plan']):.1f}", f"{r4.get('crank_offset_mm', 0):.1f}",
             f"{fm['Lc1']:.2f} / {fm['L2']:.2f} / {fm['L4']:.2f}", f"{sp['tau_E']:.2f} / {sp['tau_W']:.2f}", f"{sp['motor_extra_Nmm']:.1f}"]
    cls = ' class="rec"' if c['id'] == REC else ' class="alt"' if c['id'] == ALT else ''
    return f'<tr{cls}>' + ''.join(f'<td>{v}</td>' for v in cells) + '</tr>'


HEAD = ['案', '設計', '選び方', '手首軸の傾き °', 'ガタだけ 95% / 最大 °', '±0.05 mm 95% / 最大 °', '±0.20 mm 95% / 最大 °', '関節角誤差 °', '最小伝達角 °',
        'はみ出し mm', '寸法感度 2σ °', '畳み当てはまり', '胴幅40 / 50', 'CT骨位置ずれ mm', 'CT手の向きずれ °', '最大部品 mm', '最長ロッド mm', '最短レバー mm',
        'ゲイン最大 °/°', '体側の重ね厚さ mm', 'ロッド4のクランク mm', '最大力 N ロッド1 / 2 / 4', 'ばね 肘 / 手首 N·mm', 'モーター追加 N·mm']
NAME = {'hum': '上腕骨BC', 'direct': '直結'}
order = ['hum/direct', 'hum/hum', 'direct/hum', 'direct/direct']
nbc = lambda k: 2 - k.split('/').count('direct')
cfg_rows = ''
for k in order:
    for d in 'AB':
        s = summ[f'{d}:{k}']
        cfg_rows += (f"<tr><td>{d}</td><td>{' / '.join(NAME[p] for p in k.split('/'))}</td><td>{nbc(k)}</td><td>{s['seeds_with_feasible']}/{s['seeds']}</td>"
                     f"<td>{'—' if s['best_cheb_min'] is None else round(s['best_cheb_min'], 2)}</td><td>{s['n_in_global_front']}</td></tr>")
show = [REC, ALT]
stack_figs = ''.join(f"<figure><img src='{img(E / ('stack_' + i + '.png'))}' loading='lazy'><figcaption>{i}（案{by[i]['dsg']}）の実高さの層配置。"
                     "上段：上から見た図（色＝高さ）。下段：横から見た図（黒の縦線＝ピン・玉継手の軸、点＝玉の中心）。点線・紫の太線＝竹串の肘ロッド（ロッド1）。</figcaption></figure>"
                     for i in show)
cand_figs = ''.join(f"<figure><img src='{img(RL / ('cand_' + i + '.png'))}' loading='lazy'><figcaption>{i}（案{by[i]['dsg']}）畳んだ状態："
                    "青＝CGの畳んだ翼、橙＝モデルの羽根、赤＝骨、黒＝リンク。</figcaption></figure>" for i in show)
d8h = D8H['0.0']
re_rows = ''.join(f"<tr><td>{k}</td><td>{' / '.join(NAME.get(p, p) for p in (v['ec'], v['wc']))}</td><td>{v.get('unplaced', '—')}</td>"
                  f"<td>{'対象外（尺骨のベルクランク）' if 'skipped' in v else '成立' if v['unplaced'] == 0 else '不成立'}</td></tr>" for k, v in RE.items() if 'skipped' not in v)
grad_rows = ''.join(f"<tr><td>{g['name']}</td><td>{g['stack']:.0f}</td><td>{max(0, g['lever']):.2f}</td><td>{max(0, g['res']):.2f}</td><td>{g['cv']:.2f}</td></tr>" for g in grad)
F4 = [LD[c['id']]['force_max_with_spring_N']['L4'] for c in D]; F1 = [LD[c['id']]['force_max_with_spring_N']['Lc1'] for c in D]
F2 = [LD[c['id']]['force_max_with_spring_N']['L2'] for c in D]
r4a = CHK[ALT].get('rod4', {})

html = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>段階2e 実高さの層配置</title>
<style>
:root{{--bg:#fff;--ink:#1d2430;--mute:#5d6878;--line:#d9dee6;--hi:#fff4d6;--alt:#eef3fb}}
@media (prefers-color-scheme:dark){{:root{{--bg:#14181f;--ink:#e6eaf0;--mute:#9aa5b5;--line:#2c3440;--hi:#3a3220;--alt:#1d2633}} img{{background:#fff}}}}
body{{background:var(--bg);color:var(--ink);font:15px/1.65 system-ui,'Hiragino Sans','Yu Gothic UI',sans-serif;margin:0;padding:16px;max-width:1200px;margin:auto}}
h1{{font-size:1.5em}} h2{{margin-top:2em;border-bottom:1px solid var(--line)}} .mute{{color:var(--mute)}}
.tw{{overflow-x:auto}} table{{border-collapse:collapse;font-size:13px;white-space:nowrap}} td,th{{border:1px solid var(--line);padding:3px 7px;text-align:right}}
th{{position:sticky;top:0;background:var(--bg);white-space:normal;min-width:4.5em;vertical-align:bottom}} td:nth-child(2),td:nth-child(3){{text-align:left}}
tr.rec td{{background:var(--hi)}} tr.alt td{{background:var(--alt)}} img{{max-width:100%;height:auto}}
figure{{margin:1.2em 0}} figcaption{{color:var(--mute);font-size:13px}} li{{margin:.3em 0}}
</style></head><body>
<h1>段階2e：実際の高さで層を並べる条件を最適化に入れ、案A（平面）と案B（手首軸を傾ける）を比べ直し</h1>
<p class="mute">2026-11-15。NSGA-III、案A・B × 4構成 × 8シード、各 個体300 × 300世代、計64回（約4.4時間）。ベルクランク2個の構成は、制約の測り方を変えてさらに8回。
数値はすべて計算値で、実測ではありません。各シードの局所的なパレート解なので、大域最適の保証はありません。</p>

<h2>結論（先に）</h2>
<ul>
<li><b>実際の高さで層を並べられるのは「肘：上腕骨のベルクランク1個／手首：直結ロッド」の構成だけでした。</b>案A・Bとも 8 シード中 7 シードで解が出ました。
 <ul>
 <li>ベルクランク2個：{summ['A:hum/hum']['seeds'] + summ['B:hum/hum']['seeds']} 回＋追加 {len(grad)} 回、すべて不成立。置けないのは毎回手首ロッド（ロッド4）でした。ベルクランクの玉からの高さと手の玉の高さの差が大きく、上腕骨・ベルクランク・尺骨（翼側は厚さ枠 {BUD_WING:.0f} mm）のどれかと、枠の中で高さをずらせません。
 追加の {len(grad)} 回のうち {len(g_ok)} 回は層が成立しましたが、最短レバーが下限を {max(0, g_ok[0]['lever']) if g_ok else 0:.1f} mm 割り、畳み姿勢の残差も超えていました。</li>
 <li>肘を直結ロッド（竹串）にした構成：32 回すべて不成立。竹串が体から尺骨へ伸びる途中で、尺骨のどの高さとも当たります。曲げられない竹串にした以上、形として無理と判断しています。</li>
 </ul></li>
<li><b>推奨は {REC}（案B）。</b>寸法感度 2.3° 以下の中で畳み当てはまりが最良のものです。
 <ul>
 <li>同じ条件の案Aの最良 {ALT} と比べて：畳み当てはまり {rec['F'][4]:.1f} 対 {alt['F'][4]:.1f}、CTの手の向きずれ {rec['ct_hand']:.0f}° 対 {alt['ct_hand']:.0f}°、関節角誤差 {rec['F'][0]:.1f}° 対 {alt['F'][0]:.1f}°。
 ばらつき（±0.20 mm で 95% {rec['mc020']['p95']:.1f}° 対 {alt['mc020']['p95']:.1f}°）とガタ（{mc(rec['mc_play'])}° 対 {mc(alt['mc_play'])}°）はほぼ同じです。</li>
 <li>つまり<b>手首軸を傾ける利点は「畳んだ形が体に沿う」ことで、精度ではありません。</b>差は当てはまりで約 1.5（値の約 1 割）です。案Aの方が手首が平面で作りやすいので、作りやすさを優先するなら {ALT} も十分選べます。</li>
 <li>{REC} の手首ロッド（ロッド4）はまっすぐで済みます（クランク 0 mm）。{ALT} は 1 層分（{r4a.get('crank_offset_mm', 0):.1f} mm）曲げる必要があり、その伸びによる角度誤差は 0.9 mm 厚の板で約 {r4a.get('stretch', {}).get('plate 2.0 x 0.9', {}).get('angle_deg', 0):.2f}° と見積もりました。</li>
 </ul></li>
<li><b>高さを正しく扱った代わりに、精度は少し落ちました。</b>前回推奨の D8（層の数え方が粗かった）は、新しいルールでも手首ロッドが置けず不成立です。
 同じガタの条件（蝶番のガタ込み）で D8 と比べると、95% / 最大で、ガタだけ {mc(d8h)}° → {mc(rec['mc_play'])}°、±0.05 mm {mc(D8H['0.05'])}° → {mc(rec['mc005'])}°、±0.20 mm {mc(D8H['0.2'])}° → {mc(rec['mc020'])}° です。</li>
<li><b>ロッドにかかる力の見込みは当たっていました。</b>手首ロッド（ロッド4）は最大 {min(F4):.2f}〜{max(F4):.2f} N。肘のロッド1は {min(F1):.2f}〜{max(F1):.2f} N、ロッド2は {min(F2):.2f}〜{max(F2):.2f} N です（ばねを最小必要量の2倍にした場合）。
 推奨案ではロッド4もまっすぐなので、竹串にしても構いません。</li>
<li><b>ばね：</b>肘 約 {LD[REC]['spring']['tau_E']:.1f} N·mm、手首 約 {LD[REC]['spring']['tau_W']:.1f} N·mm の一定トルクで、どの向きに重力がかかってもロッドの力の向きが変わりません。モーターの追加トルクは最大 {LD[REC]['spring']['motor_extra_Nmm']:.1f} N·mm です（質量は推定値）。</li>
</ul>

<h2>決めていただきたいこと</h2>
<ol>
<li>段階3（CAD）に進める候補。推奨は {REC}（案B）。作りやすさ優先なら {ALT}（案A）。</li>
<li>精度の見込みを受け入れるか。寸法誤差が ±0.05 mm なら 95% で約 {rec['mc005']['p95']:.1f}°、±0.20 mm なら {rec['mc020']['p95']:.1f}° ずれます。
 ±0.20 mm の場合は、ロッド長の調整（ねじ・差し替え）で寸法誤差の分を消す前提になります。ガタの分（{rec['mc_play']['p95']:.1f}°）は調整しても残ります。</li>
<li>ベルクランク2個の構成を捨ててよいか。今回の条件（翼側 {BUD_WING:.0f} mm）では成立しません。翼側の厚さ枠を 4 mm 程度に緩めれば成立する可能性はあります（未計算）。</li>
</ol>

<h2>前回いただいた決定の反映</h2>
<ul>
<li><b>厚さは (b)：</b>リンクの板の重なりだけを数えます。足し方は「重なる2部品の高さの差＋板厚 {LAYER_T - GAP:.1f} mm」で、体の部分 {BUD_BODY:.0f} mm、翼の部分 {BUD_WING:.0f} mm 以内です。
 手と尺骨の間（手首軸の傾きで生じるすき間）は数えず、畳み当てはまりで判断します。玉継手の軸（スタッド）の長さも同じ枠で調べます。</li>
<li><b>案Aも同じモデルで：</b>案Bのモデルで、手首軸の傾きと初列のねじれを 0 に固定したものを案Aとしました。</li>
<li><b>曲げたロッド：</b>力の大きい肘のロッド1は、直径 {2 * R_SKEW:.0f} mm の竹串でまっすぐ、動かさない（運動学で決まる位置のまま）としました。
 手首のロッド4だけ、1.2 mm 単位で層を移る曲げ（クランク）を許しました。結果として、推奨案では曲げは不要でした。</li>
</ul>

<h2>層の並べ方の調べ方</h2>
<ul>
<li>板の部品（上腕骨、尺骨、手、ベルクランク、ロッド2、曲げてよいロッド4）を、それぞれ {LAYER_T} mm 単位で −3〜+3 層ずらせるとします。
 全部が当たらずに並ぶ組み合わせを、最適化の各個体ごとに探索します（21姿勢、探索の上限あり）。返す値は「置けなかった部品の数」です。</li>
<li>当たりの判定：外形（半径＋すき間 {GAP} mm）が重なる2部品は、高さの差が {LAYER_T} mm 未満なら当たり。ピンは2枚の板の間に、玉継手の軸は板から玉の中心までに伸びます。その間の高さに第三の部品があり、外形がピンに届けば当たりです。</li>
<li>候補は、全 41 姿勢・探索上限なしで調べ直しました（下の表はすべて成立）。</li>
</ul>

<h2>候補一覧</h2>
<p class="mute">黄＝推奨 {REC}（案B）、青＝案Aの対応する候補 {ALT}。「robust-fit」＝寸法感度 2.3° 以下・誤差 3.5° 以下・はみ出し 1.5 mm 以下・伝達角 25° 以上の中で畳み当てはまり最良。
<br>・ガタだけ：寸法は図面どおりで、関節と蝶番のガタだけを入れたモンテカルロ。・±a mm：全寸法を ±a mm 一様にずらし、ガタも加えたモンテカルロ。
<br>・体側の重ね厚さ：層の最上段と最下段の差＋板厚（ロッド1の竹串は除く）。・最大力：ばねを最小必要量の2倍にしたとき、重力の向きを全方向に振った最大値。</p>
<div class="tw"><table><tr>{''.join(f'<th>{h}</th>' for h in HEAD)}</tr>{''.join(row(c) for c in sorted(D, key=lambda c: (c['id'] != REC, c['id'] != ALT, c['dsg'], int(c['id'][1:]))))}</table></div>

<h2>構成ごとの結果</h2>
<div class="tw"><table><tr><th>設計</th><th>肘 / 手首</th><th>ベルクランク数</th><th>解ありのシード</th><th>最良の妥協点</th><th>全体の前線に残った数</th></tr>{cfg_rows}</table></div>
<p class="mute">ベルクランク2個の追加計算：はみ出した厚さに比例して値が減るよう制約を段階的にしたもの（シード100〜103）。</p>
<div class="tw"><table><tr><th>計算</th><th>置けない部品</th><th>レバー不足 mm</th><th>畳み残差の超過</th><th>違反の合計</th></tr>{grad_rows}</table></div>
<p class="mute">段階2dの候補を新しいルールで調べ直した結果（尺骨にベルクランクを載せる構成は対象外）：</p>
<div class="tw"><table><tr><th>案</th><th>肘 / 手首</th><th>置けない部品</th><th>判定</th></tr>{re_rows}</table></div>
<figure><img src="{img(E / 'seeds.png')}"><figcaption>シードごとの最良妥協点。</figcaption></figure>
<figure><img src="{img(E / 'pareto.png')}"><figcaption>パレート前線（6目的のうち2つずつ）。</figcaption></figure>

<h2>層の配置（実際の高さ）</h2>
{stack_figs}

<h2>畳んだ姿勢</h2>
{cand_figs}

<h2>仮定</h2>
<ul>
<li>板厚 {LAYER_T - GAP:.1f} mm＋すき間 {GAP} mm＝1層 {LAYER_T} mm。竹串は直径 {2 * R_SKEW:.0f} mm。</li>
<li>蝶番のガタ（半径方向）：肩 {HINGE_PLAY['S']} mm、肘 {HINGE_PLAY['E']} mm、手首 {HINGE_PLAY['W']} mm（樹脂ピン）。ロッド両端の関節のガタは段階2dと同じです。</li>
<li>手は手首軸に沿って動きますが、層の計算ではこれを高さ方向（z）の移動で近似しました。案Bの手首軸の傾きは 20° 前後なので、横方向へのずれは移動量の約 3 割です。</li>
<li>質量（ばね・力の計算）：前腕＋次列風切 1.5 g、手＋初列 1.0 g（推定）。力はこれに比例します。</li>
<li>曲げたロッドの伸び：PLA の縦弾性係数 3000 N/mm²、断面 2.0 × 0.9 mm、両端は玉継手（回転自由）、曲げ部は剛として計算しました。</li>
</ul>

<h2>やっていないこと・限界</h2>
<ul>
<li>強度（FEM）はまだです。段階3で、ロッド1・2、ベルクランク、蝶番まわりについて行います。</li>
<li>両端が玉継手のロッドは、自分の軸のまわりに回れます。まっすぐなロッドなら問題ありませんが、曲げたロッド（{ALT} のロッド4）は回り止めが必要です（CADで対応）。</li>
<li>シミュレータはロッドをまっすぐに描きます。曲げは表示されません。シミュレータと Python の運動学の差は全候補で 0.4° 以下です。</li>
<li>ベルクランク2個の構成は、翼側の厚さ枠 {BUD_WING:.0f} mm のもとで成立しない、という結論です。枠を緩めた計算はしていません。</li>
<li>層の探索は最適化中は 21 姿勢です。候補は 41 姿勢で確認しましたが、姿勢の間で一瞬当たる可能性は残ります。CADで確認します。</li>
</ul>
<p class="mute">体の形の参照：「Common Kingfisher」Alejandro Guzmán（Sketchfab）CC BY-SA 4.0 を縮尺・変形して使用。図に体の形が含まれるため、このページも CC BY-SA 4.0 です。</p>
</body></html>"""
out = RL / 'stage2_2e_report.html'
out.write_text(html, encoding='utf-8')
print(out, out.stat().st_size // 1024, 'KB')
