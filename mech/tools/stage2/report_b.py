"""Design-B report (tilted wrist hinge) next to the planar candidates c5/c6, as one self-contained HTML page
-> results/local/stage2_B_report.html. It embeds figures with the CG body shape (CC BY-SA 4.0), so it stays out of git."""
import sys, json, base64, pathlib, numpy as np
sys.argv = sys.argv[:1]
from problem import Evaluator
from analyze import monte_carlo

ROOT = pathlib.Path(__file__).resolve().parents[2]
R2, RL = ROOT / 'results' / 'stage2', ROOT / 'results' / 'local'
B = json.load(open(R2 / 'tilt' / 'candidates.json', encoding='utf-8'))
summ = json.load(open(R2 / 'tilt' / 'summary.json', encoding='utf-8'))
P = [c for c in json.load(open(R2 / 'candidates.json', encoding='utf-8')) if c['id'] in ('c5', 'c6')]
for c in P:                                       # CT distances and the +-0.2 mm MC were not stored for the planar run
    _, _, e = Evaluator(c['ec'], c['wc'], 45)(np.array([c['x']]), extra=True)
    c['ct_pos'], c['ct_hand'] = float(e['fitx']['ct_pos'][0]), float(e['fitx']['ct_hand'][0])
    c['mc020'] = monte_carlo(Evaluator(c['ec'], c['wc'], 45), np.array(c['x']), amp=0.20)
img = lambda p: 'data:image/png;base64,' + base64.b64encode(p.read_bytes()).decode()
NAME = {'body': '体', 'hum': '上腕骨', 'ulna': '尺骨'}
REC = ('B6', 'B3')
mc = lambda m: f"{m['p95']:.1f} / {m['max']:.1f}"


def row(c):
    F = c['F']; t = c.get('tilt', False)
    cells = [c['id'], 'B（傾き）' if t else 'A（平面）', f"{NAME[c['ec']]} / {NAME[c['wc']]}", c['pick'], f"{F[0]:.1f}", f"{-F[1]:.0f}", f"{F[2]:.1f}", f"{F[3]:.1f}",
             mc(c['mc005']), mc(c['mc010']), mc(c['mc020']), f"{c['ct_pos']:.1f}", f"{c['ct_hand']:.0f}", f"{c['tip_off']:.0f}",
             f"{F[5]:.0f}", f"{max(c['rods']):.0f}", f"{c['lever_min']:.1f}", f"{c['swivel']:.0f}", f"{c['gain_max']:.2f}", f"{min(c['margins_mm']):.2f}",
             f"{c['theta']:.0f}", f"{c['E'][1]:.0f} / {c['M'][1]:.0f}",
             f"{c['tw']:.0f}" if t else '—', f"{c['phi']:.0f}" if t else '—', f"{c['dl']:.0f}" if t else '—', f"{c['w40']['fit']:.1f} / {c['w50']['fit']:.1f}" if t else '—']
    return ('<tr class="rec">' if c['id'] in REC else '<tr class="pl">' if not t else '<tr>') + ''.join(f'<td>{v}</td>' for v in cells) + '</tr>'


HEAD = ['案', '方式', '肘BC / 手首BC の載る骨', '選び方', '関節角誤差 °', '最小伝達角 °', 'はみ出し mm', '寸法感度 2σ °', 'MC ±0.05 mm 95% / 最大 °',
        'MC ±0.10 mm 95% / 最大 °', 'MC ±0.20 mm 95% / 最大 °', 'CT骨位置ずれ mm', 'CT手の向きずれ °', '翼端ずれ mm', '最大部品 mm', '最長ロッド mm', '最短レバー mm',
        '玉継手首振り °', 'ゲイン最大 °/°', '死点余裕 mm', 'モーター °', '畳み 肘/手首 °', '手首軸の傾き °', '初列のねじれ °', '初列の角度ずらし °', '胴幅40/50の当てはまり']
cfg_rows = ''.join(f"<tr><td>{k}</td><td>{v['seeds_with_feasible']}/{v['seeds']}</td><td>{v['best_cheb_min']:.2f}</td><td>{v['best_cheb_median']:.2f}</td>"
                   f"<td>{v['best_cheb_max']:.2f}</td><td>{v['n_in_global_front']}</td></tr>" for k, v in summ['configs'].items())
figs = ''.join(f"<figure><img src='{img(RL / ('cand_' + c['id'] + '.png'))}' loading='lazy'><figcaption>{c['id']}（{NAME[c['ec']]} / {NAME[c['wc']]}, {c['pick']}）"
               "：青＝CGの畳んだ翼、橙＝モデルの羽根、赤＝骨、黒＝リンク（畳んだ状態）。</figcaption></figure>" for c in [x for x in B if x['id'] in REC] + P
               + [x for x in B if x['id'] not in REC])
b6, b3 = [next(c for c in B if c['id'] == k) for k in REC]
c5, c6 = P

html = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>案B 手首軸の傾き</title>
<style>
:root{{--bg:#fff;--ink:#1d2430;--mute:#5d6878;--line:#d9dee6;--hi:#fff4d6;--pl:#eef3fb}}
@media (prefers-color-scheme:dark){{:root{{--bg:#14181f;--ink:#e6eaf0;--mute:#9aa5b5;--line:#2c3440;--hi:#3a3220;--pl:#1d2633}} img{{background:#fff}}}}
body{{background:var(--bg);color:var(--ink);font:15px/1.65 system-ui,'Hiragino Sans','Yu Gothic UI',sans-serif;margin:0;padding:16px;max-width:1200px;margin:auto}}
h1{{font-size:1.5em}} h2{{margin-top:2em;border-bottom:1px solid var(--line)}} .mute{{color:var(--mute)}}
.tw{{overflow-x:auto}} table{{border-collapse:collapse;font-size:13px;white-space:nowrap}} td,th{{border:1px solid var(--line);padding:3px 7px;text-align:right}}
th{{position:sticky;top:0;background:var(--bg);white-space:normal;min-width:4.5em;vertical-align:bottom}} td:nth-child(2),td:nth-child(3),td:nth-child(4){{text-align:left}}
tr.rec td{{background:var(--hi)}} tr.pl td{{background:var(--pl)}} .box{{background:var(--hi);padding:.6em 1em;border-radius:6px}} img{{max-width:100%;height:auto}}
figure{{margin:1.2em 0}} figcaption{{color:var(--mute);font-size:13px}} li{{margin:.3em 0}}
</style></head><body>
<h1>案B：手首の軸を傾けた立体の翼（平面の c5・c6 との比較）</h1>
<p class="mute">2026-11-14。NSGA-III、4構成 × 12シード、各 個体300 × 400世代、計48回（約3時間）。肘ベルクランクを体に載せる構成は、平面で全部 7〜17 mm はみ出したので外しました。
数値はすべて計算値で、実測ではありません。各シードの局所的なパレート解で、大域最適の保証はありません。</p>

<h2>案Bで変えたこと</h2>
<ul>
<li><b>手首の軸を尺骨まわりに傾ける</b>（傾き tw は最適化の変数、±85°）。開いた翼は今までどおり平面で、畳むと手が腕の面から下へ外れます。</li>
<li><b>ロッド4（手首を動かすロッド）は空間リンク</b>になり、両端がロッドエンド（玉継手）です。手首ベルクランク側の玉の位置と手のレバーに、面外の寸法（b4z、qz）を足しました。
 首振りは「ロッドエンドが最もよく合う平面からどれだけ傾くか」で測り、25°以下を制約にしています。</li>
<li><b>初列風切の付け根のねじれ phi</b>（±20°）と、<b>畳んだ時の初列の角度ずらし dl</b>を変数に追加しました。</li>
<li><b>CTの手の向きを畳み当てはまりに加えました</b>（10°までは罰則なし）。このため、畳み当てはまりの数値は平面の候補と比べられません。比べるときは「CT骨位置ずれ」「CT手の向きずれ」を見てください。</li>
<li>寸法の精度を、3Dプリンタを想定して <b>±0.20 mm</b> でも調べました（モンテカルロ400回、全寸法を独立に一様にずらす）。平面の c5・c6 も同じ条件で計算し直しています。</li>
</ul>

<h2>結論（先に）</h2>
<ul>
<li><b>手の向きは CT に近づきました：全候補で 19〜24°（平面は 45〜46°）。</b>ただし 0 にはなりません。初列のねじれが全候補で上限の 20° に張り付いており、ここが制約になっています。</li>
<li><b>肘・手首の位置ずれは改善しませんでした：8.1〜10.2 mm（平面の c5 8.7、c6 7.6）。</b>原因は姿勢だけの計算と同じで、CGの体形とCT標本の違いです。</li>
<li><b>3Dプリンタの精度（±0.2 mm）に耐えられるのは、肘も手首もベルクランクを上腕骨に載せる構成だけでした。</b>
 <ul>
 <li>B6（上腕骨/上腕骨）：95% が {b6['mc020']['p95']:.1f}°、最大 {b6['mc020']['max']:.1f}°。全候補で一番強く、平面の c5（{c5['mc020']['p95']:.1f}° / {c5['mc020']['max']:.1f}°）よりも良い値です。</li>
 <li>B3（上腕骨/上腕骨）：{b3['mc020']['p95']:.1f}° / {b3['mc020']['max']:.1f}°。</li>
 <li>どちらかのベルクランクを尺骨に載せる構成は、B・平面とも 95% で 14〜25°、最大 22〜48° ずれます（c6 は {c6['mc020']['p95']:.1f}° / {c6['mc020']['max']:.1f}°）。±0.2 mm で作るなら使えません。</li>
 </ul></li>
<li><b>推奨（私の判断）：B6。次点は B3。</b>
 <ul>
 <li><b>B6 の長所</b>：寸法のばらつきに一番強い（±0.05 mm で {b6['mc005']['p95']:.1f}°、±0.20 mm で {b6['mc020']['p95']:.1f}°）。手の向きのずれ {b6['ct_hand']:.0f}°、最短レバー {b6['lever_min']:.1f} mm、最大部品 {b6['F'][5]:.0f} mm（c5 は 28 mm）。</li>
 <li><b>B6 の弱点</b>：関節角の誤差 {b6['F'][0]:.1f}°（c5 は 1.2°）。CT骨位置ずれ {b6['ct_pos']:.1f} mm は全候補で最大です。最長ロッド {max(b6['rods']):.0f} mm で、上限 40 mm に近いです。最小伝達角 {-b6['F'][1]:.0f}° です。</li>
 <li><b>B3</b> は関節角の誤差 {b3['F'][0]:.1f}°、最大部品 {b3['F'][5]:.0f} mm です。ただし最小伝達角 {-b3['F'][1]:.0f}° は制約 35° ちょうどで、±0.20 mm で {b3['mc020']['p95']:.1f}° と B6 より弱いです。</li>
 </ul></li>
<li><b>「少しの角度ずれで翼が大きく動かない」こと</b>：B の全候補で、モーター1°あたりの関節の動きは最大 1.28〜1.56°/° でした（平面は 1.56〜2.19°/°）。0.5°刻みの1段では最大 0.8° 動きます。</li>
</ul>

<h2>決めていただきたいこと</h2>
<ol>
<li>段階3（CAD）に進める候補。推奨は B6（手の向きを実物に寄せる場合）。平面のままなら c5 です。どちらも上腕骨/上腕骨の構成です。</li>
<li>案Bの部品が立体になる点を受け入れるか。具体的には、ロッド4が両端ロッドエンドになり、手首の蝶番の軸が尺骨に対して約 20° 傾きます。</li>
<li>精度の想定。±0.2 mm がばらばらに出るという計算は厳しめの仮定です。実際の3Dプリンタの誤差は、収縮のように全体で同じ向きに出る分が多く、試し刷りで補正できます。
 ロッド長を調整できるようにすれば（ねじで長さを変えるなど）、さらに減らせます。ここはまだ計算していません。</li>
</ol>

<h2>候補一覧</h2>
<p class="mute">B1〜B8＝案B、青い行＝平面の c5・c6（同じ ±0.20 mm の計算を追加）、色付き＝推奨。関節角誤差＝目標（肘150→E1、手首160→M1 をモーター角に比例）からの最大ずれ。
寸法感度＝全寸法 ±0.05 mm＋関節ガタ＋モーターの0.5°刻みによる 2σ（線形化）。MC＝各寸法を ±a mm の範囲で一様に、ロッド長にはさらにガタの分を足した、モンテカルロ400回の関節角の最大ずれ。
CT骨位置ずれ＝肘と手首のCT位置からの距離（二乗平均）。CT手の向きずれ＝手の軸の向きの差。手首軸の傾き＝尺骨まわり。</p>
<div class="tw"><table><tr>{''.join(f'<th>{h}</th>' for h in HEAD)}</tr>{''.join(row(c) for c in B + P)}</table></div>

<h2>構成比較とシード間のばらつき（案B）</h2>
<p class="mute">各シードの最良の妥協点（Chebyshev 距離、小さいほど良い）。案Bの畳み当てはまりには手の向きの罰則が入っているので、平面の値（尺骨/上腕骨 2.55、尺骨/尺骨 3.48、上腕骨/尺骨 5.19、上腕骨/上腕骨 5.40）とは直接比べられません。
上腕骨/上腕骨は 12 回中 3 回、実行可能解が見つかりませんでした。</p>
<div class="tw"><table><tr><th>肘BC / 手首BC</th><th>解ありのシード</th><th>最良</th><th>中央値</th><th>最悪</th><th>全体の前線に残った数</th></tr>{cfg_rows}</table></div>
<figure><img src="{img(R2 / 'tilt' / 'seeds.png')}"><figcaption>シードごとの最良妥協点（案B）。</figcaption></figure>
<figure><img src="{img(R2 / 'tilt' / 'pareto.png')}"><figcaption>パレート前線（案B、6目的のうち2つずつ）。</figcaption></figure>

<h2>畳んだ姿勢の図</h2>
<p class="mute">推奨の B6・B3、平面の c5・c6、残りの B の順に並べています。案Bでは、手と初列風切（橙）が腕の面から下へ外れて、脇腹に沿っています。
断面図で、羽根の下の縁が体の輪郭の内側に入って見えるところがあります。これは許容（羽根が体に 4 mm まで沈んでよい）の範囲内です。</p>
{figs}

<h2>やっていないこと・限界</h2>
<ul>
<li>初列風切のねじれが全候補で上限 20° に張り付いています。上限を上げれば手の向きはさらに合いますが、その分だけ広げた翼もねじれます。広げた翼の形は評価していません。</li>
<li>羽根の畳み方は単純なモデルです。各羽根を手の上の1本の軸まわりに回すだけで、羽根どうしの重なりや滑りは扱っていません。</li>
<li>ロッドエンドの首振り角は、最もよく合う平面からの傾きで測っています。実際のロッドエンドの許容角（製品による。約 15〜30°）と、段階3で照合が必要です。</li>
<li>±0.20 mm のモンテカルロでは、各寸法を独立にずらしています。系統的な誤差（収縮など）の扱いと、ロッド長の調整機構は、まだ入れていません。</li>
<li>シミュレータと Python の運動学の差は、全候補で 0.3° 以下です。案Bの候補はシミュレータの版リストの先頭（案B候補 B1〜B8）にあります。</li>
<li>CTは1個体（マラカイトカワセミの液浸標本、拡大して使用）だけです。</li>
</ul>
<p class="mute">体の形の参照：「Common Kingfisher」Alejandro Guzmán（Sketchfab）CC BY-SA 4.0 を縮尺・変形して使用。図に体の形が含まれるため、このページも CC BY-SA 4.0 です。</p>
</body></html>"""
out = RL / 'stage2_B_report.html'
out.write_text(html, encoding='utf-8')
print(out, out.stat().st_size // 1024, 'KB')
