"""Stage 2f, swivel trial (user 2026-11-20: 'try 20 deg'): the ball-joint swivel limit lowered from 25 to 20 deg, the three
configurations that had solutions rerun (8 seeds each, warm start from the 25-deg final populations), the same post-checks.
Compares the new picks F20-n with F1-F3 / Z1-Z3 (swivel 25) -> results/local/stage2_2f_sw20_report.html
(embeds CG-derived figures, CC BY-SA 4.0: stays out of git).
usage: report_2f_sw20.py   (after run_2f_sw20.sh and export_sim.py 2e with SWIVEL=20 STAGE_TAG=_sw20)"""
import json, base64, pathlib, collections, glob, numpy as np
from problem import LAYER_T, GAP

ROOT = pathlib.Path(__file__).resolve().parents[2]
R2, RL = ROOT / 'results' / 'stage2', ROOT / 'results' / 'local'
J = lambda p: json.load(open(p, encoding='utf-8'))
img = lambda p: 'data:image/png;base64,' + base64.b64encode(p.read_bytes()).decode()
fig = lambda p, cap: f"<figure><img src='{img(p)}' loading='lazy'><figcaption>{cap}</figcaption></figure>" if p.exists() else ''
NAME = {'hum': '上腕骨', 'uln': '尺骨', 'hand': '手', 'bc1': 'ベルクランク', 'rod1': 'ロッド1（竹串）', 'rod2': 'ロッド2', 'rod4': 'ロッド4'}
pair = lambda k: '–'.join(NAME.get(p, p) for p in k.split('-'))
PICK = {'repeat': '再現性重視', 'shape': '形重視', 'open': '開きやすさ重視'}
CFGN = {'A:hum/direct': '案A・ベルクランク1個', 'B:hum/direct': '案B・ベルクランク1個', 'B:direct/direct': '案B・ベルクランク0個'}


def load(sub):
    d = R2 / sub
    D = J(d / 'candidates.json')
    X = {n: J(d / f'{n}.json') for n in ('check_2e', 'loads', 'mirror_check', 'crossing_check', 'feather_bend', 'gate', 'summary')}
    return d, D, X


DIRS = {'25': load('2f'), '20': load('2f_sw20')}


def thick(plan):
    lv = [v for k, v in plan.items() if k != 'rod1']
    return (max(lv) - min(lv)) * LAYER_T + (LAYER_T - GAP)


def row(sw, c):
    _, _, X = DIRS[sw]; i = c['id']; F = c['F']; fm = X['loads'][i]['force_max_with_spring_N']; fb = X['feather_bend'][i]
    cr = min([t['clear'] for t in X['crossing_check'][i].values()], default=1.0)
    dsg, cfg, pk = c['pick'].split(':')
    nbc = '0' if cfg == 'direct/direct' else '1'
    st_ = max(X['check_2e'][i]['studs_mm'].items(), key=lambda kv: kv[1])
    return [i, f'{sw}°', f'案{dsg}・{nbc}個', PICK.get(pk, pk), f"<b>{c['swivel']:.1f}</b>", f"{F[0]:.1f}", f"{-F[1]:.0f}", f"{c['gain_max']:.2f}",
            f"{c['mc_play']['p95']:.1f}", f"{c['mc020']['p95']:.1f} / {c['mc020']['max']:.1f}", f"{fb['fit_flat']:.1f} → {fb['fit_bent']:.1f}",
            f"{F[2]:.1f}", f"{F[5]:.0f}", f"{max(c['rods']):.0f}", f"{c['lever_min']:.1f}", f"{pair(st_[0])} {st_[1]:.1f}",
            f"{X['mirror_check'][i]['min_clear']:.1f}", f"{cr:.2f}", f"{thick(X['check_2e'][i]['plan']):.1f}",
            f"{max(fm.values()):.2f}"]


HEAD = ['案', '首振り上限', '構成', '選び方', '首振り（実際の最大）°', '関節角誤差 °', '最小伝達角 °', 'ゲイン最大 °/°', 'ガタだけ 95% °',
        '±0.20 mm 95% / 最大 °', '畳み当てはまり 平ら → 反らせる', 'はみ出し mm', '最大部品 mm', '最長ロッド mm', '最短レバー mm', '最長のピン mm',
        '左右のすき間 mm', '部品交差の余裕 mm', '体側の重ね厚さ mm', '最大ロッド力 N']
rows = [(sw, c) for sw in ('25', '20') for c in DIRS[sw][1]]
tbl = '<tr>' + ''.join(f'<th>{h}</th>' for h in HEAD) + '</tr>' + ''.join(
    f"<tr{' class=new' if sw == '20' else ''}>" + ''.join(f'<td>{v}</td>' for v in row(sw, c)) + '</tr>' for sw, c in rows)

# gate statistics per configuration
def gate_stats(sw):
    st = collections.defaultdict(collections.Counter)
    for r in DIRS[sw][2]['gate']:
        s = st[r['C']]; s['n'] += 1; s['pass'] += bool(r['pass'])
    return st


G25, G20 = gate_stats('25'), gate_stats('20')
S25, S20 = DIRS['25'][2]['summary']['configs'], DIRS['20'][2]['summary']['configs']
cfg_rows = ''.join(
    f"<tr><td>{CFGN[k]}</td><td>{S25.get(k, {}).get('seeds_with_feasible', 0)}/{S25.get(k, {}).get('seeds', 8)}</td><td>{G25[k]['pass']}/{G25[k]['n']}</td>"
    f"<td>{S20.get(k, {}).get('seeds_with_feasible', 0)}/{S20.get(k, {}).get('seeds', 8)}</td><td>{G20[k]['pass']}/{G20[k]['n']}</td></tr>" for k in CFGN)

# no-bellcrank runs at 20 deg: what is violated in the best design of each seed (constraint 5 = swivel - SWIVEL, 2 = LMIN - lever)
zrows, zsw = '', []
for f in sorted(glob.glob(str(R2 / 'runs_2f_sw20' / 'B-direct_direct_w45_s*.npz'))):
    z = np.load(f, allow_pickle=True); Gz = z['G']; cv = np.maximum(Gz, 0).sum(1); i = int(cv.argmin()); g = np.maximum(Gz[i], 0)
    zsw.append(20 + g[5])
    zrows += f"<tr><td>{f[-5]}</td><td>{20 + g[5]:.1f}</td><td>{8 - g[2]:.2f}</td></tr>"

# picks in the new runs
F20 = DIRS['20'][1]; F25 = [c for c in DIRS['25'][1] if c['id'].startswith('F')]
rng = lambda cs, f, n: f"{min(f(c) for c in cs):.{n}f}〜{max(f(c) for c in cs):.{n}f}"
P20 = [r for r in DIRS['20'][2]['gate'] if r['pass']]

html = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>首振り20°の再計算</title>
<style>
:root{{--bg:#fff;--ink:#1d2430;--mute:#5d6878;--line:#d9dee6;--hi:#fff4d6;--alt:#eef3fb;--bad:#c0392b}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#14181f;--ink:#e6eaf0;--mute:#9aa5b5;--line:#2c3440;--hi:#3a3220;--alt:#1d2633;--bad:#ff7b6b}} img{{background:#fff}}}}
:root[data-theme="dark"]{{--bg:#14181f;--ink:#e6eaf0;--mute:#9aa5b5;--line:#2c3440;--hi:#3a3220;--alt:#1d2633;--bad:#ff7b6b}}
body{{background:var(--bg);color:var(--ink);font:15px/1.65 system-ui,'Hiragino Sans','Yu Gothic UI',sans-serif;margin:0 auto;padding:16px;max-width:1200px}}
h1{{font-size:1.5em}} h2{{margin-top:2em;border-bottom:1px solid var(--line)}} .mute{{color:var(--mute)}}
.tw{{overflow-x:auto}} table{{border-collapse:collapse;font-size:13px;white-space:nowrap}} td,th{{border:1px solid var(--line);padding:3px 7px;text-align:right}}
th{{background:var(--bg);white-space:normal;min-width:4.5em;vertical-align:bottom}} tr.new td{{background:var(--hi)}}
img{{max-width:100%;height:auto}} figure{{margin:1.2em 0}} figcaption{{color:var(--mute);font-size:13px}} li{{margin:.3em 0}}
</style></head><body>
<h1>ボールジョイントの首振り上限を20°にした再計算</h1>
<p class="mute">2026-11-20。数値はすべて計算値で、実測ではありません。各シードの局所解なので、大域最適の保証はありません。
変えたのは首振りの上限（25° → 20°）だけです。ボールの外形（直径3 mm）、ガタ、層の規則は前回と同じです。</p>

<h2>結論（先に）</h2>
<ul>
<li><b>ベルクランク0個（Z系）は、20°では成立しませんでした（0/8シード）。</b>各シードの一番ましな解でも、首振りが {min(zsw):.1f}〜{max(zsw):.1f}° 必要でした。
Z1〜Z3 を使うなら、23〜25°以上振れるボールジョイントが要ります。</li>
<li><b>ベルクランク1個は、20°でも成立します。</b>確認をすべて通った候補を4つ選びました（F20-1〜F20-4）。
実際の首振りは {rng(F20, lambda c: c['swivel'], 1)}° で、20°にも余裕があります。15°程度のボールジョイントでも足りる計算です。</li>
<li>F1〜F3（25°）と比べると、得たものと失ったものがあります。
<ul>
<li>良くなった点：はみ出し {rng(F25, lambda c: c['F'][2], 1)} → <b>{rng(F20, lambda c: c['F'][2], 1)} mm</b>、最大部品 {rng(F25, lambda c: c['F'][5], 0)} → <b>{rng(F20, lambda c: c['F'][5], 0)} mm</b>、
最長のピン（玉継手の軸）が短くなりました。</li>
<li>悪くなった点：最小伝達角 {rng(F25, lambda c: -c['F'][1], 0)} → <b>{rng(F20, lambda c: -c['F'][1], 0)}°</b>（力の伝わり方が悪く、押し引きの力やガタの影響が増える方向）、
最長ロッドが 32 → 35〜36 mm。</li>
<li>ほぼ同じ点：関節角誤差 {rng(F25, lambda c: c['F'][0], 1)} → {rng(F20, lambda c: c['F'][0], 1)}°、±0.20 mm の誤差で95%が {rng(F25, lambda c: c['mc020']['p95'], 1)} → {rng(F20, lambda c: c['mc020']['p95'], 1)}°、
反らせた羽根での当てはまり {rng(F25, lambda c: DIRS['25'][2]['feather_bend'][c['id']]['fit_bent'], 1)} → {rng(F20, lambda c: DIRS['20'][2]['feather_bend'][c['id']]['fit_bent'], 1)}、左右のすき間 1.3〜1.6 mm。</li>
</ul></li>
<li>首振りを小さくしたのに、はみ出しと部品の大きさが良くなった理由は確かめていません。前回の最終集団から続けて計算したので、計算量が増えた効果の可能性があります。
25°のまま同じだけ続けて計算すれば区別できます（未実施）。</li>
<li>シミュレータでの再現は、Python の計算と 0.37° 以内で一致しました（F20-1〜F20-4）。</li>
</ul>

<h2>決めていただきたいこと</h2>
<ul>
<li>候補の絞り込み：F20系（首振り小・はみ出し0・伝達角25〜28°）、F1〜F3（首振り25°・伝達角30〜34°）、Z1〜Z3（首振り25°・はみ出し約11 mm）のどれで段階3に進むか。</li>
<li>買う予定のボールジョイントが実際に何度振れるかを測ると、Z系を残すかどうかが決まります。</li>
</ul>

<h2>候補の比較（黄色＝首振り20°の新しい候補）</h2>
<div class="tw"><table>{tbl}</table></div>
<p class="mute">首振り（実際の最大）は全姿勢の中での最大値です。ガタだけ・±0.20 mm は寸法誤差のモンテカルロの95%点（関節角のずれ）です。
最大部品はベルクランクやレバーの最大寸法、最長のピンは層をまたぐ軸の長さです。</p>

<h2>構成ごとの結果</h2>
<div class="tw"><table><tr><th>構成</th><th>25°：解ありのシード</th><th>25°：全確認を通過</th><th>20°：解ありのシード</th><th>20°：全確認を通過</th></tr>{cfg_rows}</table></div>
<p class="mute">「全確認」＝層の配置（41姿勢）、部品の交差、左右の干渉。20°で案B・ベルクランク1個の2シードに解がないのは、層の配置が成立しなかったためで、首振りの制約ではありません。</p>

<h3>ベルクランク0個（20°）：各シードの一番ましな解</h3>
<div class="tw"><table><tr><th>シード</th><th>必要な首振り °</th><th>最短レバー mm（下限 8）</th></tr>{zrows}</table></div>

<h2>図（F20-1〜F20-4）</h2>
{''.join(fig(RL / f"cand_{c['id']}.png", f"{c['id']}（案{c['dsg']}）畳んだ状態：青＝CGの畳んだ翼、橙＝モデルの羽根（平ら）、赤＝骨、黒＝リンク。") for c in F20)}
{''.join(fig(DIRS['20'][0] / f"stack_{c['id']}.png", f"{c['id']} の実高さの層配置。上段：上から見た図（色＝高さ）。下段：横から見た図（黒の縦線＝ピン・玉継手の軸）。") for c in F20)}
{''.join(fig(DIRS['20'][0] / f"mirror_{c['id']}.png", f"{c['id']}：左右の翼が最も近づく姿勢（すき間 {DIRS['20'][2]['mirror_check'][c['id']]['min_clear']:.1f} mm）。") for c in F20)}

<h2>やっていないこと・限界</h2>
<ul>
<li>ボールの外形は直径3 mmのままです。アリエクの M2 ロッドエンドは玉の幅が約4 mmあるので、実物では交差の余裕が減ります。</li>
<li>25°で同じ計算量を追加した比較はしていません（上記）。</li>
<li>駆動の取り回し、強度（FEM）、姿勢の間での一瞬の接触は未確認です（段階3の CAD で確認）。</li>
</ul>
<p class="mute">体の形の参照：「Common Kingfisher」Alejandro Guzmán（Sketchfab）CC BY-SA 4.0 を縮尺・変形して使用。図に体の形が含まれるため、このページも CC BY-SA 4.0 です。</p>
</body></html>"""
out = RL / 'stage2_2f_sw20_report.html'
out.write_text(html, encoding='utf-8')
print(out, out.stat().st_size // 1024, 'KB')
