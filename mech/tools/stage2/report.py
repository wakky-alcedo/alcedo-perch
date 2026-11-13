"""Stage-2 report as one self-contained HTML page (images embedded) -> results/local/stage2_report.html.
It embeds figures that show the CG body shape (CC BY-SA 4.0), so it stays out of git."""
import json, base64, pathlib, numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
R2, RL = ROOT / 'results' / 'stage2', ROOT / 'results' / 'local'
cands = json.load(open(R2 / 'candidates.json', encoding='utf-8'))
summ = json.load(open(R2 / 'summary.json', encoding='utf-8'))
img = lambda p: 'data:image/png;base64,' + base64.b64encode(p.read_bytes()).decode()
SIGQ = 0.5 / np.sqrt(12)
NAME = {'body': '体', 'hum': '上腕骨', 'ulna': '尺骨'}


def row(c):
    F = c['F']; g = c['gain_max']
    cells = [c['id'], f"{NAME[c['ec']]} / {NAME[c['wc']]}", c['pick'], f"{F[0]:.1f}", f"{-F[1]:.0f}", f"{F[2]:.1f}", f"{F[3]:.1f}",
             f"{c['mc005']['p95']:.1f} / {c['mc005']['max']:.1f}", f"{c['mc010']['p95']:.1f} / {c['mc010']['max']:.1f}",
             f"{F[4]:.1f}", f"{c['tip_off']:.0f}", f"{F[5]:.0f}", f"{c['lever_min']:.1f}", f"{c['swivel']:.0f}",
             f"{g:.2f}", f"{0.5 * g:.2f}", f"{2 * SIGQ * g:.2f}", f"{min(c['margins_mm']):.2f}", f"{c['rodforce_N_per_Nmm']:.2f}",
             f"{c['theta']:.0f}", f"{c['E'][1]:.0f} / {c['M'][1]:.0f}", f"{c['fz']:.1f}", f"{c['w40']['fit']:.1f} / {c['w50']['fit']:.1f}"]
    return '<tr>' + ''.join(f'<td>{v}</td>' for v in cells) + '</tr>'


HEAD = ['案', '肘BC / 手首BC の載る骨', '選び方', '関節角誤差 °', '最小伝達角 °', 'はみ出し mm', '寸法感度 2σ °', 'MC ±0.05 mm 95% / 最大 °',
        'MC ±0.10 mm 95% / 最大 °', '畳み当てはまり', '翼端ずれ mm', '最大部品 mm', '最短レバー mm', '玉継手首振り °', 'ゲイン最大 °/°',
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
tr.rec td{{background:var(--hi)}} img{{max-width:100%;height:auto}} figure{{margin:1.2em 0}} figcaption{{color:var(--mute);font-size:13px}}
li{{margin:.3em 0}}
</style></head><body>
<h1>段階2：畳み姿勢とリンク寸法の同時最適化（胴幅45 mm）</h1>
<p class="mute">2026-11-13。NSGA-III（pymoo 0.6.2）、6構成 × 12シード、各 個体300 × 400世代、計72回（全回で実行可能解あり）。数値はすべて計算値で、実測ではありません。
得られたのは各シードの局所的なパレート解で、大域最適の保証はありません。</p>

<h2>結論（先に）</h2>
<ul>
<li><b>肘のベルクランクを上腕骨に載せ（ワット型）、手首のベルクランクを尺骨に載せる構成</b>が、12シードとも他の構成より良い妥協点に収まりました（下の構成比較表、Chebyshev距離の中央値 1.34、他は 2.3〜6.6）。これは却下案 r4 と同じ配置ですが、r4 を却下した理由（寸法感度：0.1 mm で100°以上）は、感度を目的に、レバー4 mm以上・ゲイン2.5°/°以下を制約に入れたことで解消しています（c2 で ±0.10 mm のモンテカルロ 95% が 4.6°、最大 6.6°）。</li>
<li>v4 と同じ「肘ベルクランクを体に載せる」構成（c1, c4）は、どのシードでもベルクランクが背中から 4〜9 mm はみ出しました。肩の上は体が薄いためです。</li>
<li>候補の推奨（私の判断）：<b>c8</b>（寸法感度が最小、±0.05 mm で 95% が 1.9°、レバー 12 mm）か <b>c2</b>（構成内の妥協点、誤差 1.8°、はみ出し 0.1 mm）。伝達角は c8 が 27°、c2 が 29° で、一般的な目安の 30〜40° を少し下回ります。伝達角を優先するなら c9（35°）です。</li>
<li>ご要望の「少しの角度ずれで翼が大きく動かない」こと：全候補でモーター1°あたりの肘・手首の動きは <b>最大 1.34〜1.61°/°</b>（制約は 2.5°/°）。0.5°刻み1段で最大 0.8°動きます。シミュレータのモーターは 0.5°刻みに戻してあります。</li>
</ul>

<h2>決めていただきたいこと</h2>
<ol>
<li>段階3に進める候補（1つ、または比較用に2つ）。</li>
<li>伝達角 27〜30° を許容するか。許容しないなら c9 系か、伝達角の下限を制約にして再計算します（その構成だけなら約30分）。</li>
<li>羽根の浮かせ量 fz（骨の面から羽根を背側へ浮かせる量）が全候補で上限の 6 mm に張り付いています。羽根の付け根を 6 mm 以上浮かせる構造が作れるなら、上限を上げると畳み姿勢がさらに合います。</li>
<li>仮定の確認：玉継手の首振り上限 25°、最短レバー 4 mm（実寸）。段階3で部品を選ぶ際に確定します。</li>
</ol>

<h2>候補一覧</h2>
<p class="mute">関節角誤差＝目標（肘150→E1、手首160→M1 をモーター角に比例）からの最大ずれ。寸法感度＝全寸法 ±0.05 mm（一様）＋関節ガタ 0.01 mm＋モーターの0.5°刻みによる関節角の 2σ（線形化）。
MC＝同じばらつきでのモンテカルロ400回。はみ出し＝CGメッシュの体から外へ出るリンクの最大量。畳み当てはまり＝CGの畳んだ翼との差（小さいほど良い、v4 は 18.8）。
ゲイン＝モーター1°あたりの関節角の変化の最大値。ロッド力＝モーター1 N·mm あたりの最大ロッド力。</p>
<div class="tw"><table><tr>{''.join(f'<th>{h}</th>' for h in HEAD)}</tr>{''.join(row(c) for c in cands)}</table></div>
<p class="mute">比較：v4 を同じ基準で評価すると、誤差 4.1°、伝達角 32°、はみ出し 0.6 mm、感度 4.0°、当てはまり 18.8、首振り 60°（制約違反）、±0.05 mm での最悪ずれ 13.5°、ゲイン最大 1.56°/° です。</p>

<h2>構成比較とシード間のばらつき</h2>
<p class="mute">各シードで最良の妥協点（目標値で正規化した Chebyshev 距離、小さいほど良い）の最小・中央値・最大と、全72回を合わせたパレート前線に残った解の数。</p>
<div class="tw"><table><tr><th>肘BC / 手首BC</th><th>解ありのシード</th><th>最良</th><th>中央値</th><th>最悪</th><th>全体の前線に残った数</th></tr>{cfg_rows}</table></div>
<figure><img src="{img(R2 / 'seeds.png')}"><figcaption>シードごとの最良妥協点。</figcaption></figure>
<figure><img src="{img(R2 / 'pareto.png')}"><figcaption>パレート前線（6目的のうち2つずつ）。濃い点は全体で非劣、丸が候補。</figcaption></figure>

<h2>却下案の失敗との照合</h2>
<ul>
<li>r1（単段、伝達角不足）：全候補が2段で、最小伝達角は 11〜35°。c1（17°）と c4（11°）は r1 と同じく伝達角が低く、勧めません。</li>
<li>r2（部品が翼より大きい）：最大部品は 12〜20 mm（尺骨 32 mm より短い）。</li>
<li>r3（手のレバー 2 mm で感度が致命的）：最短レバーは全候補で約 5 mm 以上（最小 c4 の 4.97 mm）。</li>
<li>r4（感度が致命的）：同じ配置の c2/c7〜c10 で ±0.10 mm の最大ずれは 4.7〜11°。r4 の 100°以上から大きく改善しています。</li>
<li>v4（玉継手 60°、体に埋まる）：首振り 9〜25°、はみ出しと埋まりは CG メッシュで評価。</li>
</ul>

<h2>畳んだ姿勢の図（候補ごと）</h2>
{figs}

<h2>やっていないこと・限界</h2>
<ul>
<li>ホモトピー法による厳密合成は行っていません（任意項目）。</li>
<li>アルゴリズムは NSGA-III だけで、NSGA-II とは比べていません。</li>
<li>製作性・応力集中は「最大部品寸法」と「最短レバー」で代用しています。</li>
<li>畳み姿勢の当てはまりは、どの構成でも約 9.5 より良くなりません。原因は翼を平面と仮定していることで（段階1の結果）、翼端は CG から 7〜12 mm ずれます。</li>
<li>前後方向の動きの制約（sweep ≥ −4 mm）も多くの候補で境界に張り付いています。</li>
<li>はみ出しは CG メッシュの体で評価しています。シミュレータの楕円体では値が違います（例：c8 は楕円体で 3.4 mm、メッシュで 1.2 mm）。</li>
<li>シミュレータと Python の運動学の差は全候補で 0.34° 以下です。この差はモーター角を 0.5° に丸めたことによるものです。</li>
</ul>
<p class="mute">体の形の参照：「Common Kingfisher」Alejandro Guzmán（Sketchfab）CC BY-SA 4.0 を縮尺・変形して使用。図に体の形が含まれるため、このページも CC BY-SA 4.0 です。</p>
</body></html>"""
out = RL / 'stage2_report.html'
out.write_text(html, encoding='utf-8')
print(out, out.stat().st_size // 1024, 'KB')
