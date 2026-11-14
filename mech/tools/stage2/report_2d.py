"""Stage-2d report (link interference as layers, per-joint play, fewer bellcranks) as one self-contained HTML page
-> results/local/stage2_2d_report.html. It embeds figures with the CG body shape (CC BY-SA 4.0), so it stays out of git.
usage: report_2d.py   (after analyze.py 45 tilt 2d, cand_figs.py tilt 2d, layer_figs.py)"""
import sys, json, base64, pathlib, numpy as np
sys.argv = sys.argv[:1]
from problem import LAYER_T, LAYERS_ALL, LAYERS_WING, R_BONE, R_ROD, R_BOSS, R_PIN, GAP, TMIN_DIRECT
from mech import ROD_PLAY

ROOT = pathlib.Path(__file__).resolve().parents[2]
R2, RL = ROOT / 'results' / 'stage2', ROOT / 'results' / 'local'
D = json.load(open(R2 / '2d' / 'candidates.json', encoding='utf-8'))
summ = json.load(open(R2 / '2d' / 'summary.json', encoding='utf-8'))
OLD = json.load(open(R2 / '2d' / 'old_B_recheck.json', encoding='utf-8'))      # B6 / B3 of stage 2c under the new rules
img = lambda p: 'data:image/png;base64,' + base64.b64encode(p.read_bytes()).decode()
NAME = {'hum': '上腕骨', 'ulna': '尺骨', 'direct': '直結ロッド'}
NBC = lambda c: (c['ec'] != 'direct') + (c['wc'] != 'direct')
REC, ALT = 'D8', 'D6'
mc = lambda m: f"{m['p95']:.1f} / {m['max']:.1f}"
by = {c['id']: c for c in D}
rec, alt, d1, d3 = by[REC], by[ALT], by['D1'], by['D3']


def row(c):
    F = c['F']
    cells = [c['id'], f"{NBC(c)}", f"{NAME[c['ec']]} / {NAME[c['wc']]}", c['pick'], f"{c['layers']} / {c['layers_wing']}",
             f"{sum(v < 0 for v in c['clash'].values())}", mc(c['mc_play']), mc(c['mc005']), mc(c['mc020']),
             f"{F[0]:.1f}", f"{-F[1]:.0f}", f"{F[2]:.1f}", f"{F[3]:.2f}", f"{c['ct_pos']:.1f}", f"{c['ct_hand']:.0f}", f"{F[4]:.1f}",
             f"{c['w40']['fit']:.1f} / {c['w50']['fit']:.1f}", f"{F[5]:.0f}", f"{max(c['rods']):.0f}", f"{c['lever_min']:.1f}", f"{c['swivel']:.0f}",
             f"{c['gain_max']:.2f}", f"{min(c['margins_mm']):.2f}", f"{c['tw']:.0f}"]
    cls = ' class="rec"' if c['id'] == REC else ' class="alt"' if c['id'] == ALT else ''
    return f'<tr{cls}>' + ''.join(f'<td>{v}</td>' for v in cells) + '</tr>'


HEAD = ['案', 'ベルクランク数', '肘 / 手首 の駆動', '選び方', '層の数 全体 / 翼', '交差する部品の組（層で分離）', 'ガタだけ 95% / 最大 °', '±0.05 mm 95% / 最大 °',
        '±0.20 mm 95% / 最大 °', '関節角誤差 °', '最小伝達角 °', 'はみ出し mm', '寸法感度 2σ °', 'CT骨位置ずれ mm', 'CT手の向きずれ °', '畳み当てはまり',
        '胴幅40 / 50 の当てはまり', '最大部品 mm', '最長ロッド mm', '最短レバー mm', '玉継手首振り °', 'ゲイン最大 °/°', '死点余裕 mm', '手首軸の傾き °']
order = ['hum/direct', 'direct/direct', 'direct/hum', 'ulna/direct', 'hum/hum', 'ulna/hum']
lab = lambda k: ' / '.join(NAME[p] for p in k.split('/'))
cfg_rows = ''.join(f"<tr><td>{lab(k)}</td><td>{2 - k.split('/').count('direct')}</td><td>{summ['configs'][k]['seeds_with_feasible']}/{summ['configs'][k]['seeds']}</td>"
                   f"<td>{summ['configs'][k]['best_cheb_min']:.2f}</td><td>{summ['configs'][k]['best_cheb_median']:.2f}</td><td>{summ['configs'][k]['best_cheb_max']:.2f}</td>"
                   f"<td>{summ['configs'][k]['n_in_global_front']}</td></tr>" for k in order if k in summ['configs'])
show = [REC, ALT, 'D1', 'D3', 'D5', 'D4', 'D7']
lay_figs = ''.join(f"<figure><img src='{img(R2 / '2d' / ('layers_' + i + '.png'))}' loading='lazy'><figcaption>{i}（{NAME[by[i]['ec']]} / {NAME[by[i]['wc']]}）の層の割り当て。"
                   "色＝層。上腕骨の座標系で見た図なので、面の外への飛び出し（手首軸の傾き・ロッドエンドの高さ）は見えません。</figcaption></figure>" for i in show)
cand_figs = ''.join(f"<figure><img src='{img(RL / ('cand_' + i + '.png'))}' loading='lazy'><figcaption>{i}（{NAME[by[i]['ec']]} / {NAME[by[i]['wc']]}）畳んだ状態："
                    "青＝CGの畳んだ翼、橙＝モデルの羽根、赤＝骨、黒＝リンク。</figcaption></figure>" for i in show[:4])
b6, b3 = OLD['B6'], OLD['B3']
play_tab = ''.join(f"<tr><td>{k}</td><td>{v:.2f}</td></tr>" for k, v in ROD_PLAY.items())

html = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>段階2d 干渉とガタ</title>
<style>
:root{{--bg:#fff;--ink:#1d2430;--mute:#5d6878;--line:#d9dee6;--hi:#fff4d6;--alt:#eef3fb}}
@media (prefers-color-scheme:dark){{:root{{--bg:#14181f;--ink:#e6eaf0;--mute:#9aa5b5;--line:#2c3440;--hi:#3a3220;--alt:#1d2633}} img{{background:#fff}}}}
body{{background:var(--bg);color:var(--ink);font:15px/1.65 system-ui,'Hiragino Sans','Yu Gothic UI',sans-serif;margin:0;padding:16px;max-width:1200px;margin:auto}}
h1{{font-size:1.5em}} h2{{margin-top:2em;border-bottom:1px solid var(--line)}} .mute{{color:var(--mute)}}
.tw{{overflow-x:auto}} table{{border-collapse:collapse;font-size:13px;white-space:nowrap}} td,th{{border:1px solid var(--line);padding:3px 7px;text-align:right}}
th{{position:sticky;top:0;background:var(--bg);white-space:normal;min-width:4.5em;vertical-align:bottom}} td:nth-child(3),td:nth-child(4){{text-align:left}}
tr.rec td{{background:var(--hi)}} tr.alt td{{background:var(--alt)}} img{{max-width:100%;height:auto}}
figure{{margin:1.2em 0}} figcaption{{color:var(--mute);font-size:13px}} li{{margin:.3em 0}}
</style></head><body>
<h1>段階2d：リンクの干渉と関節のガタを入れて、ベルクランクを減らした構成と比較</h1>
<p class="mute">2026-11-14。NSGA-III、6構成 × 12シード、各 個体300 × 400世代、計72回（約3.7時間）。すべて手首軸を傾けた案B（立体）です。
数値はすべて計算値で、実測ではありません。各シードで見つかった局所的なパレート解なので、大域最適の保証はありません。</p>

<h2>結論（先に）</h2>
<ul>
<li><b>干渉の条件を入れると、前回の推奨 B6・B3 は成立しません。</b>層が B6 で {b6['layers']} 枚（翼の部分 {b6['layers_wing']} 枚）、B3 で {b3['layers']} 枚必要で、
 上限（全体 {LAYERS_ALL} 枚 ≈ {LAYERS_ALL * LAYER_T:.1f} mm、翼 {LAYERS_WING} 枚 ≈ {LAYERS_WING * LAYER_T:.1f} mm）を超えます。ロッドどうしが同じ面で {b6['ncross']}・{b3['ncross']} 組交差していたためです。</li>
<li><b>ベルクランクを1個にした構成（肘：上腕骨のベルクランク、手首：直結ロッド）が一番良くなりました。推奨は {REC}。</b>
 <ul>
 <li>層は {rec['layers']} 枚（翼 {rec['layers_wing']} 枚）＝ 約 {rec['layers'] * LAYER_T:.1f} mm。はみ出し {rec['F'][2]:.1f} mm。</li>
 <li>ばらつきに一番強い：±0.20 mm で 95% が {rec['mc020']['p95']:.1f}°、最大 {rec['mc020']['max']:.1f}°。
 ベルクランク2個のまま層の条件を満たした {d1['id']} は {d1['mc020']['p95']:.1f}° / {d1['mc020']['max']:.1f}° で、{REC} の方が良い値です。
 干渉を無視した前回の B6（{b6['mc020']['p95']:.1f}° / {b6['mc020']['max']:.1f}°）とほぼ同じです。</li>
 <li>ガタだけによるずれ：95% で {rec['mc_play']['p95']:.1f}°、最大 {rec['mc_play']['max']:.1f}°（全候補で最小）。</li>
 </ul></li>
<li><b>つまり、ベルクランクを減らしても精度はほとんど落ちませんでした。</b>干渉の条件を入れると、2個の構成は交差を避けるために形が制限され、利点がなくなります。</li>
<li><b>ベルクランクなし（{ALT}）も成立します。</b>層は {alt['layers']} 枚だけで、交差はありません。±0.20 mm で {alt['mc020']['p95']:.1f}° / {alt['mc020']['max']:.1f}° です。
 ただし、肘のロッドが体の輪郭から <b>{alt['F'][2]:.1f} mm はみ出し</b>、畳み当てはまりも {alt['F'][4]:.1f}（{REC} は {rec['F'][4]:.1f}）と悪くなります。胴幅40 mm では {alt['w40']['fit']:.1f} まで悪化します。
 肘のロッドは体の中の点から尺骨へ直接つなぐので、畳んだときに体の外を通ってしまいます。</li>
<li><b>ベルクランクを尺骨に載せる構成は、今回も ±0.20 mm で 95% が 16〜24° ずれます。</b>候補から外すことを勧めます。</li>
</ul>

<h2>いただいた3つの指摘への回答</h2>
<h3>1. ロッド長の調整に頼ると、軸のガタで死ぬのでは</h3>
<ul>
<li>その通りなので、ずれを2つに分けて計算しました。
 <ul>
 <li><b>寸法の誤差</b>（作ったときに決まる。組み立て後は変わらない）：±0.05 / ±0.20 mm のモンテカルロ。</li>
 <li><b>関節のガタ</b>（動かすたびに変わる。調整では直せない）：寸法は図面どおりとして、ガタだけを入れたモンテカルロ。</li>
 </ul></li>
<li>ガタの値は関節の種類ごとに決めました。681ベアリング 0.01 mm、玉継手・ロッドエンド 0.03 mm、手の樹脂ピン 0.05 mm です。
 ロッド長のガタは、両端の関節の値を足しています。</li>
<li>{REC} のガタだけのずれは 95% で {rec['mc_play']['p95']:.1f}°、最大 {rec['mc_play']['max']:.1f}° です。モーター 0.5° の1段で関節が動く量（最大 {0.5 * rec['gain_max']:.1f}°）と同じくらいです。
 ガタの影響は、最適化の目的（寸法感度）にも入れています。</li>
<li><b>ロッド長の調整は「寸法誤差」だけを減らすためのもので、ガタの分は調整に関係なく残ります。</b>{REC} ではその残る分が約1°です。
 ただし、ガタの値は仮定です。樹脂ピンのガタが 0.1 mm になると、手首側のずれはおよそ倍になる見込みです（線形として推定、未計算）。</li>
</ul>
<h3>2. リンク同士の干渉</h3>
<ul>
<li>各部品をカプセル（太さのある線分）で表しました。骨（取り付け部を含む）は半径 {R_BONE} mm、ロッドは {R_ROD} mm、ベアリングまわりは {R_BOSS} mm、ピンまわりは {R_PIN} mm です。
 畳む動きを 11 姿勢（0.1 刻み）に分けて調べています。姿勢の間で一瞬だけ当たる場合は見逃す可能性があります。</li>
<li>ピンでつながる部品どうし、または {GAP} mm 以内に近づく部品どうしは、別の層に置く必要があります。この条件で必要な層の数（グラフの彩色数）を、正確に計算しました。</li>
<li>制約は、全体 {LAYERS_ALL} 層以下（1層 {LAYER_T} mm として約 {LAYERS_ALL * LAYER_T:.1f} mm）、翼の部分 {LAYERS_WING} 層以下（約 {LAYERS_WING * LAYER_T:.1f} mm）です。翼の部分とは、尺骨・手に近い場所のことです。</li>
<li>表の「交差する部品の組」は、ピンでつながっていないのに同じ場所を通る組の数です。違う層に置けば当たりません。</li>
</ul>
<h3>3. ベルクランクを減らせないか</h3>
<ul>
<li>減らせます。上の結論のとおり、1個（{REC}）が一番良い結果でした。0個（{ALT}）は、はみ出しを許せるなら使えます。</li>
<li>直結ロッドの構成では、ベルクランクで力の向きを変えられません。このため最小伝達角の制約を 35° から {TMIN_DIRECT}° に緩めました。
 {REC} の最小伝達角は {-rec['F'][1]:.0f}° で、緩める前の 35° をやや下回ります。</li>
</ul>

<h2>決めていただきたいこと</h2>
<ol>
<li>段階3（CAD）に進める候補。推奨は {REC}（ベルクランク1個）です。はみ出し約 9 mm を羽根で隠せると考えるなら {ALT}（ベルクランク0個）も選べます。</li>
<li>{REC} の弱点を受け入れるか。
 <ul>
 <li>関節角の誤差が {rec['F'][0]:.1f}° です（{d1['id']} は {d1['F'][0]:.1f}°）。</li>
 <li>最小伝達角が {-rec['F'][1]:.0f}° です。</li>
 <li>畳み当てはまりが {rec['F'][4]:.1f} です（{d1['id']} は {d1['F'][4]:.1f}）。</li>
 <li>最長ロッドが {max(rec['rods']):.0f} mm です。</li>
 </ul></li>
<li>ガタの仮定。特に手の樹脂ピン 0.05 mm が実現できるか（穴とピンのはめあい）。段階3で試し刷りして測るのが確実です。</li>
</ol>

<h2>候補一覧</h2>
<p class="mute">色付き＝推奨 {REC}、青＝ベルクランクなし {ALT}。
<br>・ガタだけ：寸法は図面どおりで、関節のガタだけを入れたモンテカルロ400回の関節角の最大ずれ。
<br>・±a mm：全寸法を ±a mm の範囲で一様にずらし、さらにガタも加えたモンテカルロ。
<br>・寸法感度：±0.05 mm＋ガタ＋0.5°刻みによる 2σ（線形化）。
<br>・層の数は、各構成の最良妥協点で数えた値です。</p>
<div class="tw"><table><tr>{''.join(f'<th>{h}</th>' for h in HEAD)}</tr>{''.join(row(c) for c in sorted(D, key=lambda c: (c['id'] != REC, c['id'] != ALT, int(c['id'][1:]))))}</table></div>
<p class="mute">干渉の条件で比べた前回の候補（案B）：
<br>・B6：層 {b6['layers']}（翼 {b6['layers_wing']}）、ガタだけ {mc(b6['mc_play'])}°、±0.20 mm {mc(b6['mc020'])}°。
<br>・B3：層 {b3['layers']}（翼 {b3['layers_wing']}）、ガタだけ {mc(b3['mc_play'])}°、±0.20 mm {mc(b3['mc020'])}°。
<br>どちらも層の制約を外れます。</p>

<h2>構成ごとの比較（シード間のばらつき）</h2>
<p class="mute">各シードの最良の妥協点（Chebyshev 距離、小さいほど良い）。肘を直結ロッドにした構成は、はみ出しが約 9 mm あるため、この値が 9 前後になっています。
ベルクランク2個の構成では、層の条件を満たす解が見つからないシードがありました。上腕骨/上腕骨は 12 回中 8 回、尺骨/上腕骨は 5 回、直結/上腕骨は 1 回です。</p>
<div class="tw"><table><tr><th>肘 / 手首</th><th>ベルクランク数</th><th>解ありのシード</th><th>最良</th><th>中央値</th><th>最悪</th><th>全体の前線に残った数</th></tr>{cfg_rows}</table></div>
<figure><img src="{img(R2 / '2d' / 'seeds.png')}"><figcaption>シードごとの最良妥協点。</figcaption></figure>
<figure><img src="{img(R2 / '2d' / 'pareto.png')}"><figcaption>パレート前線（6目的のうち2つずつ）。</figcaption></figure>

<h2>層の割り当て</h2>
<p class="mute">開いた状態（u=0）、途中（u=0.5）、畳んだ状態（u=1）。線の太さは部品の外形のおおよその太さです。</p>
{lay_figs}

<h2>畳んだ姿勢</h2>
{cand_figs}

<h2>仮定</h2>
<ul>
<li>ガタ（半径方向）：681ベアリング 0.01 mm、玉継手・ロッドエンド 0.03 mm、手の樹脂ピン 0.05 mm。ロッドごとのガタ（両端の和、mm）は次のとおりです。
<table style="margin:.5em 0"><tr><th>ロッド</th><th>ガタ mm</th></tr>{play_tab}</table></li>
<li>層の厚さ 1.2 mm。部品の太さは上記のカプセルの半径です。</li>
<li>直結ロッドの構成だけ、最小伝達角の下限を {TMIN_DIRECT}° にしています（ほかは 35°）。</li>
</ul>

<h2>やっていないこと・限界</h2>
<ul>
<li>ピンが途中の層を貫く場合の干渉（ピンの軸と、間の層にある部品）は調べていません。段階3のCADで確認します。</li>
<li>層は「同じ層に置けない組」から数えただけで、実際の高さ（手首軸の傾きやロッドエンドの高さ）との整合はまだ取っていません。翼の部分の層の数は、翼の部分だけの組で数えた近似です。</li>
<li>肘・手首の蝶番そのもののガタは入れていません（ロッドの両端の関節だけです）。</li>
<li>以前に話した「プーリーと腱で手首を動かす案（b）」は計算していません。</li>
<li>ばねで片側に押し付けてガタを消す方法は、荷重の向きが途中で逆になると効きません。この確認はしていません。</li>
<li>シミュレータと Python の運動学の差は、全候補で 0.33° 以下です。候補はシミュレータの版リストの先頭（段階2d候補 D1〜D10）にあります。</li>
</ul>
<p class="mute">体の形の参照：「Common Kingfisher」Alejandro Guzmán（Sketchfab）CC BY-SA 4.0 を縮尺・変形して使用。図に体の形が含まれるため、このページも CC BY-SA 4.0 です。</p>
</body></html>"""
out = RL / 'stage2_2d_report.html'
out.write_text(html, encoding='utf-8')
print(out, out.stat().st_size // 1024, 'KB')
