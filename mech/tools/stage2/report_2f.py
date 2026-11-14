"""Stage-2f report (left / right wings must not meet at the body midline) as one self-contained HTML page
-> results/local/stage2_2f_report.html. It embeds figures with the CG body shape (CC BY-SA 4.0), so it stays out of git.
usage: MIDLINE=1 report_2f.py   (after, with MIDLINE=1: analyze.py 45 2e, loads.py 2e, check_2e.py, mirror_check.py, mirror_figs.py,
       stack_figs.py, cand_figs.py 2e; and without MIDLINE: mirror_check.py, mirror_figs.py E1 E2 for the stage-2e evidence)"""
import sys, json, base64, pathlib, numpy as np
sys.argv = sys.argv[:1]
from problem import LAYER_T, GAP, MID_X, MID_GAP
from stack import BUD_BODY, BUD_WING, R_SKEW

ROOT = pathlib.Path(__file__).resolve().parents[2]
R2, RL = ROOT / 'results' / 'stage2', ROOT / 'results' / 'local'
E, Fd = R2 / '2e', R2 / '2f'
J = lambda p: json.load(open(p, encoding='utf-8'))
D = J(Fd / 'candidates.json'); summ = J(Fd / 'summary.json')['configs']
CHK, LD, MC = J(Fd / 'check_2e.json'), J(Fd / 'loads.json'), J(Fd / 'mirror_check.json')
D2E = {c['id']: c for c in J(E / 'candidates.json')}; MC2E = J(E / 'mirror_check.json')
img = lambda p: 'data:image/png;base64,' + base64.b64encode(p.read_bytes()).decode()
by = {c['id']: c for c in D}
ok = lambda i: MC[i]['min_clear'] >= GAP and CHK[i]['unplaced'] == 0
REC = next(c['id'] for c in D if c['pick'] == 'B:robust-mc')
ALT = next(c['id'] for c in D if c['pick'] == 'A:robust-mc')
RF = next(c['id'] for c in D if c['pick'] == 'B:robust-fit')
rf = by[RF]
SCR = J(Fd / 'screen_mc.json')
rec, alt = by[REC], by[ALT]
e1, e2 = D2E['E1'], D2E['E2']
mc = lambda m: f"{m['p95']:.1f} / {m['max']:.1f}"
NAME = {'hum': '上腕骨', 'uln': '尺骨', 'hand': '手', 'bc1': 'ベルクランク', 'rod1': 'ロッド1', 'rod2': 'ロッド2', 'rod4': 'ロッド4'}
pair = lambda k: '–'.join(NAME.get(p, p) for p in k.split('-'))


def thick(plan):
    lv = [v for k, v in plan.items() if k != 'rod1']
    return (max(lv) - min(lv)) * LAYER_T + (LAYER_T - GAP)


def row(c):
    F = c['F']; ch = CHK[c['id']]; ld = LD[c['id']]; fm = ld['force_max_with_spring_N']; sp = ld['spring']; m = MC[c['id']]
    r4 = ch.get('rod4', {})
    cells = [c['id'], c['dsg'], c['pick'].replace('best-A:hum/direct', '総合（A）').replace('best-B:hum/direct', '総合（B）'), f"{c['tw']:.0f}",
             f"{m['min_clear']:.1f}", '可' if ok(c['id']) else '<b>不可</b>',
             mc(c['mc_play']), mc(c['mc005']), mc(c['mc020']),
             f"{F[0]:.1f}", f"{-F[1]:.0f}", f"{F[2]:.1f}", f"{F[3]:.2f}", f"{F[4]:.1f}", f"{c['w40']['fit']:.1f} / {c['w50']['fit']:.1f}",
             f"{c['ct_pos']:.1f}", f"{c['ct_hand']:.0f}", f"{F[5]:.0f}", f"{max(c['rods']):.0f}", f"{c['lever_min']:.1f}", f"{c['gain_max']:.2f}",
             f"{thick(ch['plan']):.1f}", f"{r4.get('crank_offset_mm', 0):.1f}",
             f"{fm['Lc1']:.2f} / {fm['L2']:.2f} / {fm['L4']:.2f}", f"{sp['tau_E']:.2f} / {sp['tau_W']:.2f}", f"{sp['motor_extra_Nmm']:.1f}"]
    cls = ' class="rec"' if c['id'] == REC else ' class="alt"' if c['id'] == ALT else ''
    return f'<tr{cls}>' + ''.join(f'<td>{v}</td>' for v in cells) + '</tr>'


HEAD = ['案', '設計', '選び方', '手首軸の傾き °', '左右のすき間 mm', '左右判定', 'ガタだけ 95% / 最大 °', '±0.05 mm 95% / 最大 °', '±0.20 mm 95% / 最大 °',
        '関節角誤差 °', '最小伝達角 °', 'はみ出し mm', '寸法感度 2σ °', '畳み当てはまり', '胴幅40 / 50', 'CT骨位置ずれ mm', 'CT手の向きずれ °', '最大部品 mm',
        '最長ロッド mm', '最短レバー mm', 'ゲイン最大 °/°', '体側の重ね厚さ mm', 'ロッド4のクランク mm', '最大力 N ロッド1 / 2 / 4', 'ばね 肘 / 手首 N·mm', 'モーター追加 N·mm']
e2e_rows = ''.join(f"<tr><td>{k}</td><td>{D2E[k]['dsg']}</td><td>{v['min_clear']:.1f}</td><td>{v['u']:.2f}</td>"
                   f"<td style='text-align:left'>{'、'.join(pair(p) + f' {x:.1f}' for p, x in list(v['clashes'].items())[:4])}</td></tr>" for k, v in MC2E.items())
cfg_rows = ''.join(f"<tr><td>{d}</td><td>{summ[f'{d}:hum/direct']['seeds_with_feasible']}/{summ[f'{d}:hum/direct']['seeds']}</td>"
                   f"<td>{summ[f'{d}:hum/direct']['best_cheb_min']:.2f}</td><td>{summ[f'{d}:hum/direct']['best_cheb_median']:.2f}</td>"
                   f"<td>{summ[f'{d}:hum/direct']['n_in_global_front']}</td></tr>" for d in 'AB')
n_ok = sum(ok(c['id']) for c in D)
fig = lambda p, cap: f"<figure><img src='{img(p)}' loading='lazy'><figcaption>{cap}</figcaption></figure>"
F4 = [LD[c['id']]['force_max_with_spring_N']['L4'] for c in D]; F1 = [LD[c['id']]['force_max_with_spring_N']['Lc1'] for c in D]
F2 = [LD[c['id']]['force_max_with_spring_N']['L2'] for c in D]
d = lambda a, b, k=4: f"{a['F'][k]:.1f} → {b['F'][k]:.1f}"

html = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>段階2f 左右の翼の干渉</title>
<style>
:root{{--bg:#fff;--ink:#1d2430;--mute:#5d6878;--line:#d9dee6;--hi:#fff4d6;--alt:#eef3fb;--bad:#c0392b}}
@media (prefers-color-scheme:dark){{:root{{--bg:#14181f;--ink:#e6eaf0;--mute:#9aa5b5;--line:#2c3440;--hi:#3a3220;--alt:#1d2633;--bad:#ff7b6b}} img{{background:#fff}}}}
body{{background:var(--bg);color:var(--ink);font:15px/1.65 system-ui,'Hiragino Sans','Yu Gothic UI',sans-serif;margin:0;padding:16px;max-width:1200px;margin:auto}}
h1{{font-size:1.5em}} h2{{margin-top:2em;border-bottom:1px solid var(--line)}} .mute{{color:var(--mute)}} .bad{{color:var(--bad)}}
.tw{{overflow-x:auto}} table{{border-collapse:collapse;font-size:13px;white-space:nowrap}} td,th{{border:1px solid var(--line);padding:3px 7px;text-align:right}}
th{{position:sticky;top:0;background:var(--bg);white-space:normal;min-width:4.5em;vertical-align:bottom}} td:nth-child(2),td:nth-child(3){{text-align:left}}
tr.rec td{{background:var(--hi)}} tr.alt td{{background:var(--alt)}} img{{max-width:100%;height:auto}}
figure{{margin:1.2em 0}} figcaption{{color:var(--mute);font-size:13px}} li{{margin:.3em 0}}
</style></head><body>
<h1>段階2f：左右の翼が体の中心で当たらない条件を加えて計算し直し</h1>
<p class="mute">2026-11-15。NSGA-III、肘：上腕骨のベルクランク／手首：直結ロッドの構成、案A・B × 8シード、各 個体300 × 300世代。
数値はすべて計算値で、実測ではありません。各シードの局所的なパレート解なので、大域最適の保証はありません。</p>

<h2>結論（先に）</h2>
<ul>
<li><b class="bad">段階2eの候補 E1〜E11 は、すべて左右の翼の部品どうしが当たります。</b>
 前回推奨の E1 では、左右のベルクランクが体の中心線上で重なり、すき間は {MC2E['E1']['min_clear']:.1f} mm（部品の太さぶん完全に重なっている）でした。
 これまでの干渉チェックが片方の翼しか見ていなかったためで、確認漏れでした。</li>
<li><b>中心線の制約を入れても、解は十分ありました。</b>案A・Bとも 8 シード中 {summ['A:hum/direct']['seeds_with_feasible']}・{summ['B:hum/direct']['seeds_with_feasible']} シードで解が出ました。
 候補 {len(D)} 個のうち {n_ok} 個が、左右を鏡写しにした全 41 姿勢のチェックで当たりなし（すき間 {GAP} mm 以上）です。</li>
<li><b class="bad">代わりに、寸法誤差が大きいときの精度が落ちました。</b>ベルクランクを中心線の手前に収めるため、肘のリンクが短くなったためです
 （E1 → {RF}：ロッド1 {e1['rods'][0]:.0f} → {rf['rods'][0]:.0f} mm、ロッド2 {e1['rods'][1]:.0f} → {rf['rods'][1]:.0f} mm、最短レバー {e1['lever_min']:.1f} → {rf['lever_min']:.1f} mm）。
 E1 と同じ選び方（寸法感度 2.3° 以下で畳み当てはまり最良）の {RF} では、±0.20 mm の 95% が {e1['mc020']['p95']:.1f}° → {rf['mc020']['p95']:.1f}° に悪化しました。
 選び方に使っている「寸法感度」は小さなずれを前提にした近似で、この悪化をとらえていませんでした。</li>
<li><b>そこで、前線の {len(SCR)} 解に直接モンテカルロ（±0.05 / ±0.20 mm）をかけて選び直しました。</b>
 最も良いもので ±0.20 mm の 95% が {min(r['mc020']['p95'] for r in SCR):.1f}° でした（200回の試行。E1 は {e1['mc020']['p95']:.1f}°）。今回得られた解の中では、中心線の制約の代償として ±0.20 mm で最良でも約 {min(r['mc020']['p95'] for r in SCR) - e1['mc020']['p95']:.0f}°、推奨の {REC} で {rec['mc020']['p95'] - e1['mc020']['p95']:.1f}° 悪くなります。最適化の目的に非線形のばらつきを直接入れれば改善の余地はあります（未計算、時間がかかります）。
 ±0.05 mm なら差は小さく（E1 {e1['mc005']['p95']:.1f}° → {rec['mc005']['p95']:.1f}°）、ガタだけなら同等です（{mc(e1['mc_play'])}° → {mc(rec['mc_play'])}°）。</li>
<li><b>推奨は {REC}（案B）。</b>関節角誤差 4° 以下・±0.20 mm の 95% が 8.5° 以下の中で畳み当てはまり最良のものです。左右のすき間は最小 {MC[REC]['min_clear']:.1f} mm。
 E1 と比べて：畳み当てはまり {d(e1, rec)}、関節角誤差 {d(e1, rec, 0)}°、最小伝達角 {-e1['F'][1]:.0f}° → {-rec['F'][1]:.0f}°、
 ±0.20 mm の 95% / 最大 {mc(e1['mc020'])}° → {mc(rec['mc020'])}°。{RF} と比べると、関節角誤差が {rf['F'][0]:.1f}° → {rec['F'][0]:.1f}° と少し悪い代わりに、±0.20 mm で {rf['mc020']['p95']:.1f}° → {rec['mc020']['p95']:.1f}° です。</li>
<li>案Aの対応する候補は {ALT}（左右のすき間 {MC[ALT]['min_clear']:.1f} mm）。E2 と比べて畳み当てはまり {d(e2, alt)}。
 案Bとの差は当てはまりで {alt['F'][4] - rec['F'][4]:.1f}、CTの手の向きずれで {alt['ct_hand']:.0f}° 対 {rec['ct_hand']:.0f}° です。</li>
<li>ロッドにかかる力（ばねを最小必要量の2倍）：ロッド1 {min(F1):.2f}〜{max(F1):.2f} N、ロッド2 {min(F2):.2f}〜{max(F2):.2f} N、ロッド4 {min(F4):.2f}〜{max(F4):.2f} N。段階2eと同じ程度です。</li>
</ul>

<h2>決めていただきたいこと</h2>
<ol>
<li>段階3（CAD）に進める候補。推奨は {REC}（案B）。作りやすさ（手首が平面）優先なら {ALT}（案A）。関節角誤差を優先するなら {RF}（案B、±0.20 mm に弱い）。</li>
<li>精度の見込みを受け入れるか。±0.05 mm で 95% 約 {rec['mc005']['p95']:.1f}°、±0.20 mm で {rec['mc020']['p95']:.1f}°。±0.20 mm の印刷精度なら、ロッド長の調整（ねじ・差し替え）が前提です。
 中心線から逃がす量を減らす（体の中心でリンクを上下に重ねる、など）と精度は戻る可能性がありますが、左右が同じ高さの鏡写しなので、そのためには左右で層をずらす非対称な設計が必要です（未検討）。</li>
<li>中心線から各部品までのすき間 {MID_GAP} mm（左右合わせて {2 * MID_GAP:.0f} mm 以上）でよいか。体の中心にはモーターと左右をつなぐ駆動も入るので、
 そちらに場所が要るなら、この値を大きくして計算し直します（未計算）。</li>
</ol>

<h2>何が当たっていたか（段階2eの候補）</h2>
<p>左右の翼を体の中心線（x = {MID_X} mm、肩関節が 0）で鏡写しにし、両翼を同じだけ動かして全 41 姿勢で部品どうしのすき間を測りました。
肩関節は中心線から {-MID_X} mm しか離れていません。そこへ肘を動かすベルクランクの腕が内側へ伸び、反対側の翼の同じ部品と重なっていました。
左右は鏡写しで同じ高さにあるので、層（高さ）をずらしても避けられません。</p>
{fig(E / 'mirror_E1.png', 'E1（段階2eの推奨）。赤×＝左右の部品が重なる場所。左右のベルクランク（青）とロッド2（緑）、竹串のロッド1（紫）の端が中心線上で重なっています。')}
<div class="tw"><table><tr><th>案</th><th>設計</th><th>最小すき間 mm</th><th>その姿勢 u</th><th>当たる部品（すき間 mm）</th></tr>{e2e_rows}</table></div>
<p class="mute">すき間 −4.0 mm は、2つの部品の太さ（半径の和）ぶん完全に重なっている状態です。</p>

<h2>直し方</h2>
<ul>
<li>最適化に「どの部品の表面も、中心線から {MID_GAP} mm 以上自分の側にある」という制約を足しました（全 41 姿勢）。左右が鏡写しなので、これで左右のすき間は {2 * MID_GAP:.0f} mm 以上になります。</li>
<li>部品の位置は、層の配置で決めた高さを反映させて測ります。畳んだ姿勢では上腕骨の面が傾くので、高さ方向に最大 {3 * LAYER_T:.1f} mm ずらした部品は横にも動きます。</li>
<li class="mute">1回目の計算ではこの高さのずれを入れておらず、鏡写しのチェックで 12 候補中 6 個が当たっていました。それを直して計算し直したのが今回の結果です（1回目の最終集団から再開）。</li>
<li>対象は、段階2eで層の配置が成立した唯一の構成（肘：上腕骨のベルクランク1個、手首：直結ロッド）です。他の構成は段階2eの時点で不成立でした。</li>
</ul>
{fig(Fd / f'mirror_{REC}.png', f'{REC}（推奨）の左右の翼が最も近づく姿勢。')}

<h2>候補一覧</h2>
<p class="mute">黄＝推奨 {REC}（案B）、青＝案Aの対応する候補 {ALT}。「robust-fit」＝寸法感度 2.3° 以下・誤差 3.5° 以下・はみ出し 1.5 mm 以下・伝達角 25° 以上の中で畳み当てはまり最良。
<br>・「robust-mc」＝関節角誤差 4° 以下・±0.20 mm モンテカルロ 95% が 8.5° 以下（前線の選別、200回）の中で畳み当てはまり最良。<br>・左右のすき間：左右の翼を鏡写しにしたときの部品表面どうしの最小距離（全 41 姿勢、層の高さ込み）。{GAP} mm 以上を可としました。
<br>・ガタだけ：寸法は図面どおりで、関節と蝶番のガタだけを入れたモンテカルロ。・±a mm：全寸法を ±a mm 一様にずらし、ガタも加えたモンテカルロ。
<br>・体側の重ね厚さ：層の最上段と最下段の差＋板厚（ロッド1の竹串は除く）。・最大力：ばねを最小必要量の2倍にしたとき、重力の向きを全方向に振った最大値。</p>
<div class="tw"><table><tr>{''.join(f'<th>{h}</th>' for h in HEAD)}</tr>{''.join(row(c) for c in sorted(D, key=lambda c: (c['id'] != REC, c['id'] != ALT, c['dsg'], int(c['id'][1:]))))}</table></div>

<h2>案A・Bの計算結果</h2>
<div class="tw"><table><tr><th>設計</th><th>解ありのシード</th><th>最良の妥協点</th><th>シードの中央値</th><th>全体の前線に残った数</th></tr>{cfg_rows}</table></div>
{fig(Fd / 'seeds.png', 'シードごとの最良妥協点。')}
{fig(Fd / 'pareto.png', 'パレート前線（6目的のうち2つずつ）。')}

<h2>層の配置（実際の高さ）</h2>
{''.join(fig(Fd / ('stack_' + i + '.png'), f"{i}（案{by[i]['dsg']}）の実高さの層配置。上段：上から見た図（色＝高さ）。下段：横から見た図（黒の縦線＝ピン・玉継手の軸、点＝玉の中心）。") for i in (REC, ALT))}
{fig(Fd / f'mirror_{ALT}.png', f'{ALT}（案A）の左右の翼が最も近づく姿勢。')}

<h2>畳んだ姿勢</h2>
{''.join(fig(RL / ('cand_' + i + '.png'), f"{i}（案{by[i]['dsg']}）畳んだ状態：青＝CGの畳んだ翼、橙＝モデルの羽根、赤＝骨、黒＝リンク。") for i in (REC, ALT))}

<h2>仮定</h2>
<ul>
<li>体の中心線は x = {MID_X} mm（肩関節が 0）。CTの肩の位置の決め方と、CGの胴の断面の中心から決めました。胴幅 45 mm のとき。</li>
<li>左右の翼は同じモーターで同じだけ動く（同じ u）とし、鏡写しで比べました。左右で動きがずれると近づき方は変わります。</li>
<li>板厚 {LAYER_T - GAP:.1f} mm＋すき間 {GAP} mm＝1層 {LAYER_T} mm。竹串は直径 {2 * R_SKEW:.0f} mm。厚さ枠は体の部分 {BUD_BODY:.0f} mm、翼の部分 {BUD_WING:.0f} mm。</li>
<li>質量（ばね・力の計算）：前腕＋次列風切 1.5 g、手＋初列 1.0 g（推定）。力はこれに比例します。</li>
</ul>

<h2>やっていないこと・限界</h2>
<ul>
<li><b>1つのモーターで左右の上腕骨をどうつないで動かすか（駆動の取り回し）は、まだモデルに入っていません。</b>モーターと駆動の部品は中心線のすぐ近くに来るので、段階3のCADで、ロッド1の竹串の体側の端や左右のベルクランクと場所を取り合わないか確認します。</li>
<li>強度（FEM）はまだです。段階3で行います。</li>
<li>層の探索は最適化中は 21 姿勢です。候補は 41 姿勢で確認しましたが、姿勢の間で一瞬当たる可能性は残ります。CADで確認します。</li>
<li>シミュレータはロッドをまっすぐに描きます。左右の翼の表示はありません（片側のみ）。</li>
</ul>
<p class="mute">体の形の参照：「Common Kingfisher」Alejandro Guzmán（Sketchfab）CC BY-SA 4.0 を縮尺・変形して使用。図に体の形が含まれるため、このページも CC BY-SA 4.0 です。</p>
</body></html>"""
out = RL / 'stage2_2f_report.html'
out.write_text(html, encoding='utf-8')
print(out, out.stat().st_size // 1024, 'KB', 'ok', n_ok, '/', len(D), 'REC', REC, MC[REC]['min_clear'], 'ALT', ALT, MC[ALT]['min_clear'])
