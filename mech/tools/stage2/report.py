"""Stage-2 report as one self-contained HTML page (images embedded) -> results/local/stage2_report.html.
It embeds figures that show the CG body shape (CC BY-SA 4.0), so it stays out of git."""
import json, base64, pathlib, numpy as np
from problem import Evaluator

ROOT = pathlib.Path(__file__).resolve().parents[2]
R2, RL = ROOT / 'results' / 'stage2', ROOT / 'results' / 'local'
cands = json.load(open(R2 / 'candidates.json', encoding='utf-8'))
summ = json.load(open(R2 / 'summary.json', encoding='utf-8'))
for c in cands:                                   # distance of the folded bones from the CT landmarks (not stored in candidates.json)
    _, _, e = Evaluator(c['ec'], c['wc'], 45)(np.array([c['x']]), extra=True)
    c['ct_pos'], c['ct_hand'] = float(e['fitx']['ct_pos'][0]), float(e['fitx']['ct_hand'][0])
img = lambda p: 'data:image/png;base64,' + base64.b64encode(p.read_bytes()).decode()
SIGQ = 0.5 / np.sqrt(12)
NAME = {'body': '体', 'hum': '上腕骨', 'ulna': '尺骨'}


def row(c):
    F = c['F']; g = c['gain_max']
    cells = [c['id'], f"{NAME[c['ec']]} / {NAME[c['wc']]}", c['pick'], f"{F[0]:.1f}", f"{-F[1]:.0f}", f"{F[2]:.1f}", f"{F[3]:.1f}",
             f"{c['mc005']['p95']:.1f} / {c['mc005']['max']:.1f}", f"{c['mc010']['p95']:.1f} / {c['mc010']['max']:.1f}",
             f"{F[4]:.1f}", f"{c['ct_pos']:.1f}", f"{c['ct_hand']:.0f}", f"{c['tip_off']:.0f}", f"{F[5]:.0f}", f"{c['lever_min']:.1f}", f"{c['swivel']:.0f}",
             f"{g:.2f}", f"{0.5 * g:.2f}", f"{2 * SIGQ * g:.2f}", f"{min(c['margins_mm']):.2f}", f"{c['rodforce_N_per_Nmm']:.2f}",
             f"{c['theta']:.0f}", f"{c['E'][1]:.0f} / {c['M'][1]:.0f}", f"{c['fz']:.1f}", f"{c['w40']['fit']:.1f} / {c['w50']['fit']:.1f}"]
    return ('<tr class="rec">' if c['id'] in REC else '<tr>') + ''.join(f'<td>{v}</td>' for v in cells) + '</tr>'
REC = ('c5', 'c6')


HEAD = ['案', '肘BC / 手首BC の載る骨', '選び方', '関節角誤差 °', '最小伝達角 °', 'はみ出し mm', '寸法感度 2σ °', 'MC ±0.05 mm 95% / 最大 °',
        'MC ±0.10 mm 95% / 最大 °', '畳み当てはまり', 'CT骨位置ずれ mm', 'CT手の向きずれ °', '翼端ずれ mm', '最大部品 mm', '最短レバー mm', '玉継手首振り °', 'ゲイン最大 °/°',
        '0.5°刻み1段の動き °', 'モーター量子化 2σ °', '死点余裕 mm', 'ロッド力 N/(N·mm)', 'モーター °', '畳み 肘/手首 °', '羽根浮かせ mm', '胴幅40/50の当てはまり']
cfg_rows = ''.join(f"<tr><td>{k}</td><td>{v['seeds_with_feasible']}/{v['seeds']}</td><td>{v['best_cheb_min']:.2f}</td><td>{v['best_cheb_median']:.2f}</td>"
                   f"<td>{v['best_cheb_max']:.2f}</td><td>{v['n_in_global_front']}</td></tr>" for k, v in summ['configs'].items())
figs = ''.join(f"<figure><img src='{img(RL / ('cand_' + c['id'] + '.png'))}' loading='lazy'><figcaption>{c['id']}（{NAME[c['ec']]} / {NAME[c['wc']]}, {c['pick']}）"
               "：青＝CGの畳んだ翼、橙＝モデルの羽根、赤＝骨、黒＝リンク（畳んだ状態）。断面の実線は体、点線は胴だけ。</figcaption></figure>" for c in cands)

html = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>段階2 再最適化</title>
<style>
:root{{--bg:#fff;--ink:#1d2430;--mute:#5d6878;--line:#d9dee6;--hi:#fff4d6}}
@media (prefers-color-scheme:dark){{:root{{--bg:#14181f;--ink:#e6eaf0;--mute:#9aa5b5;--line:#2c3440;--hi:#3a3220}} img{{background:#fff}}}}
body{{background:var(--bg);color:var(--ink);font:15px/1.65 system-ui,'Hiragino Sans','Yu Gothic UI',sans-serif;margin:0;padding:16px;max-width:1200px;margin:auto}}
h1{{font-size:1.5em}} h2{{margin-top:2em;border-bottom:1px solid var(--line)}} .mute{{color:var(--mute)}}
.tw{{overflow-x:auto}} table{{border-collapse:collapse;font-size:13px;white-space:nowrap}} td,th{{border:1px solid var(--line);padding:3px 7px;text-align:right}}
th{{position:sticky;top:0;background:var(--bg);white-space:normal;min-width:4.5em;vertical-align:bottom}} td:nth-child(2),td:nth-child(3){{text-align:left}}
tr.rec td{{background:var(--hi)}} .box{{background:var(--hi);padding:.6em 1em;border-radius:6px}} img{{max-width:100%;height:auto}} figure{{margin:1.2em 0}} figcaption{{color:var(--mute);font-size:13px}}
li{{margin:.3em 0}}
</style></head><body>
<h1>段階2（CT反映版）：畳み姿勢とリンク寸法の同時最適化（胴幅45 mm）</h1>
<p class="mute">2026-11-13。NSGA-III（pymoo 0.6.2）、6構成 × 12シード、各 個体300 × 400世代、計72回。うち <b>25回は実行可能解なし</b>（制約が厳しくなったため。構成ごとの数は下の表）。
数値はすべて計算値で、実測ではありません。得られたのは各シードの局所的なパレート解で、大域最適の保証はありません。</p>

<h2>前回からの変更</h2>
<ul>
<li><b>肩の位置の決め方をCTの実測に変更</b>：マラカイトカワセミのCT（DigiMorph, FMNH 319990、液浸標本）を尺骨長でカワセミに拡大（×1.47）して、
肩の左右位置 12.7 mm（前回 9）、手首の前端から肩までの前後距離 8.7 mm（前回 4）、背中の表面からの深さ 5.0 mm（前回 7。皮膚まで実測 2.8 mm ＋ 羽毛 2 mm は仮定）。</li>
<li><b>畳んだ骨の位置もCTに合わせる</b>：肘と手首の位置がCTから 3 mm 以上離れると罰則（畳み当てはまりに加算）。</li>
<li><b>羽根の浮かせ量 fz の上限を 6 → 3 mm</b>：CTで骨の中心から外側の表面まで中央値 1.3 mm（拡大後 約2 mm）だったため。前回の 6 mm は実物より厚すぎました。</li>
<li><b>最短レバー 4 → 8 mm、最小伝達角 35° 以上を制約に</b>（前回は伝達角を目的にだけ入れていて 27〜29° の候補が出たため）。</li>
<li>手首の畳み角 M1 の上限を 50° に（CTでは 46°）。</li>
</ul>
<p class="mute">前回（旧ルール）の結果は results/stage2/runs_old_rule/ に残してあります。目標と制約が違うので、数値をそのまま比べることはできません。</p>

<h2>結論（先に）</h2>
<ul>
<li><b>構成の順位が前回と入れ替わりました。</b>肘のベルクランクを尺骨に載せ、手首のベルクランクを上腕骨に載せる構成（尺骨/上腕骨）が、シードの中央値で一番良い妥協点でした（Chebyshev 中央値 2.55）。次が 尺骨/尺骨（3.48）、上腕骨/尺骨（5.19）、上腕骨/上腕骨（5.40）です。前回1位の上腕骨/尺骨は、肩が深く後ろになったため、はみ出し 4.2 mm、最大部品 32 mm（尺骨と同じ長さ）になりました。</li>
<li><b>肘ベルクランクを体に載せる構成（c1, c4）は、今回も 7〜17 mm はみ出しました。</b>勧めません。</li>
<li><b>推奨（私の判断）：c5 と c6 の2つを比較用に。</b>
 <ul>
 <li><b>c5</b>（上腕骨/上腕骨）：寸法のばらつきに一番強い（±0.05 mm で 95% が 2.0°、±0.10 mm で 4.0°）。関節角の誤差 1.2°、はみ出し 1.8 mm。弱点は畳み当てはまり 19.6 で、胴幅40 mmだと 23.9 まで悪くなること。</li>
 <li><b>c6</b>（尺骨/上腕骨）：部品が一番小さく（最大 20 mm）、畳み当てはまり 13.8、胴幅40〜50 mmで安定（15.8 / 14.2）、ゲイン最小 1.56°/°、最短レバー 10.3 mm。弱点は寸法感度が c5 の約2倍（±0.05 mm で 95% が 4.6°、±0.10 mm で 8.8°、最大 13.8°）。</li>
 </ul>
 製作精度が ±0.05 mm 程度出せるなら c6、出せる自信がなければ c5、というのが私の考えです。</li>
<li>「少しの角度ずれで翼が大きく動かない」こと：全候補でモーター1°あたりの関節の動きは最大 1.56〜2.19°/°（制約 2.5°/°）。0.5°刻み1段で最大 1.1°です。</li>
<li>伝達角は全候補 35° 以上、最短レバーは全候補 8 mm 以上になりました。</li>
</ul>

<h2>重要な限界：平面の翼ではCTの手の向きを再現できない</h2>
<p class="box">CTの畳んだ翼では、手（掌骨＋指）が上腕骨と尺骨の作る面から <b>約40° 下へ</b>外れて、脇腹に沿って垂れています。今のモデルは肘・手首が同じ面内で曲がる平面の翼なので、
全候補で手の向きが CT と <b>44〜48°</b> ずれます（表の「CT手の向きずれ」）。肘と手首の位置のずれは 6.4〜8.7 mm です。このため今回は手の向きを罰則に入れず、位置だけ合わせています。
手首の軸を傾けると（試算、姿勢だけの最適化 3 シード）、当てはまりが 18.5 → 14.3、骨位置ずれが 8.3 → 3.4 mm に改善しました。ただし手の向きは 40° ずれたままで、
これは羽根の並びを手の軸に沿わせているモデルの作りと、CGの翼端位置とがぶつかるためです。</p>

<h2>決めていただきたいこと</h2>
<ol>
<li>段階3に進める候補（推奨は c5 と c6 の比較。1つなら製作精度で選ぶ）。</li>
<li>平面の翼のまま進めるか（A）、手首の軸を傾けた立体の翼に作り直すか（B：手首ロッドを空間リンクにして玉継手、羽根の畳み方のモデルも作り直し。作業 約半日＋計算 約3時間）。
  A なら手の向きが実物と約45°違うまま、B なら位置は合うが部品が立体になり、製作と調整が難しくなります。</li>
<li>羽毛の厚さ 2 mm（仮定）でよいか。液浸標本では羽毛が寝ているので、生きた鳥の厚さは測れていません。</li>
</ol>

<h2>候補一覧</h2>
<p class="mute">関節角誤差＝目標（肘150→E1、手首160→M1 をモーター角に比例）からの最大ずれ。寸法感度＝全寸法 ±0.05 mm（一様）＋関節ガタ 0.01 mm＋モーターの0.5°刻みによる関節角の 2σ（線形化）。
MC＝同じばらつきでのモンテカルロ400回。はみ出し＝CGメッシュの体から外へ出るリンクの最大量。畳み当てはまり＝CGの畳んだ翼との差＋CTの肘・手首位置から3 mmを超えた分（小さいほど良い。姿勢だけ最適化したときの最良が 13.9）。CT骨位置ずれ＝肘と手首のCT位置からの距離（二乗平均）。CT手の向きずれ＝手の軸の向きの差。行の色付きは推奨。
ゲイン＝モーター1°あたりの関節角の変化の最大値。ロッド力＝モーター1 N·mm あたりの最大ロッド力。</p>
<div class="tw"><table><tr>{''.join(f'<th>{h}</th>' for h in HEAD)}</tr>{''.join(row(c) for c in cands)}</table></div>


<h2>構成比較とシード間のばらつき</h2>
<p class="mute">各シードで最良の妥協点（目標値で正規化した Chebyshev 距離、小さいほど良い）の最小・中央値・最大と、全72回を合わせたパレート前線に残った解の数。</p>
<div class="tw"><table><tr><th>肘BC / 手首BC</th><th>解ありのシード</th><th>最良</th><th>中央値</th><th>最悪</th><th>全体の前線に残った数</th></tr>{cfg_rows}</table></div>
<figure><img src="{img(R2 / 'seeds.png')}"><figcaption>シードごとの最良妥協点。</figcaption></figure>
<figure><img src="{img(R2 / 'pareto.png')}"><figcaption>パレート前線（6目的のうち2つずつ）。濃い点は全体で非劣、丸が候補。</figcaption></figure>

<h2>却下案の失敗との照合</h2>
<ul>
<li>r1（単段、伝達角不足）：全候補が2段で、最小伝達角は 35.7〜44.3°（制約 35°）。</li>
<li>r2（部品が翼より大きい）：c1（56 mm）、c4（43 mm）、c2（32 mm）は該当または境界。推奨の c5 は 28 mm、c6 は 20 mm。</li>
<li>r3（手のレバー 2 mm で感度が致命的）：最短レバーは全候補 8 mm 以上。</li>
<li>r4（感度が致命的）：±0.10 mm のモンテカルロ最大ずれは 6.6〜26.5°。c7・c9・c10 は 24〜27° と大きく、勧めません。</li>
<li>v4（玉継手 60°、体に埋まる）：首振りは 16〜25°（制約 25°）。</li>
</ul>

<h2>CTで確かめたこと（図）</h2>
<figure><img src="{img(RL / 'ct_interpretation.png')}" loading="lazy"><figcaption>CTの骨の読み取り（赤＝上腕骨、青＝尺骨、黄＝手）。</figcaption></figure>
<figure><img src="{img(RL / 'ct_wing_sections.png')}" loading="lazy"><figcaption>翼の骨（赤＋）から外側の表面（水色）までの距離 mm。</figcaption></figure>

<h2>畳んだ姿勢の図（候補ごと）</h2>
{figs}

<h2>やっていないこと・限界</h2>
<ul>
<li>CTは1個体（マラカイトカワセミ、液浸標本、カワセミより小さい）だけです。左右の位置の差（の半分）は肘 0.6 mm、手首 3.2 mm。体の軸の取り方（胴の主軸か背骨の線か）で 13° 違い、肘の目標位置が約 6 mm 変わります。</li>
<li>CT の元データ（MorphoSource）は申請制で、今回使ったのは DigiMorph の断面動画（圧縮あり）です。縮尺は再構成円から求め、他の2方向の動画と約4%以内で一致しました。</li>
<li>25回は実行可能解が見つかりませんでした（体/上腕骨は12回中9回）。制約の境界付近の解なので、段階3で寸法を丸めると制約を外れる可能性があります。</li>
<li>c10 はシミュレータが最初に別の組み立て方（手首側の枝）を選んでいたため、枝を指定して合わせました。開いた姿勢の近くにもう一つの組み立て方があるということなので、実機では組み立て時に注意が要ります。</li>
<li>ホモトピー法による厳密合成はしていません。アルゴリズムは NSGA-III だけです。</li>
<li>はみ出しは CG メッシュの体で評価しています。シミュレータの楕円体では値が違います（例：c1 は楕円体で 7.0 mm）。</li>
<li>シミュレータと Python の運動学の差は全候補で 0.35° 以下です（モーター角を 0.5° に丸めたことによる差）。</li>
</ul>
<p class="mute">体の形の参照：「Common Kingfisher」Alejandro Guzmán（Sketchfab）CC BY-SA 4.0 を縮尺・変形して使用。図に体の形が含まれるため、このページも CC BY-SA 4.0 です。</p>
</body></html>"""
out = RL / 'stage2_report.html'
out.write_text(html, encoding='utf-8')
print(out, out.stat().st_size // 1024, 'KB')
