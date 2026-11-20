"""Stage 2f, dihedral trial (user 2026-11-21: '+8 deg dihedral, tip slightly lowered'): the open wing raised 8 deg about the
body's fore-aft axis (kin.DIHEDRAL), swivel limit 20, the two bellcrank-1 configurations rerun (8 seeds each, warm start from the
2f_sw20 finals with the folded frame kept), the same post-checks; primaries' root-tip chord >= 3 deg below the wing plane (DROOP=3).
Compares the new picks V8-n with F20-n (flat open wing) -> results/local/stage2_2f_dh8_report.html
(embeds CG-derived figures, CC BY-SA 4.0: stays out of git). Also writes results/stage2/2f_dh8/primary_profile.png (model only).
usage: report_2f_dh8.py   (after run_2f_dh8.sh, the CAMBER_MAX sweep of feather_bend.py, export_sim.py 2e)"""
import json, base64, pathlib, collections, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from problem import LAYER_T, GAP
from kin import PRIM

ROOT = pathlib.Path(__file__).resolve().parents[2]
R2, RL = ROOT / 'results' / 'stage2', ROOT / 'results' / 'local'
J = lambda p: json.load(open(p, encoding='utf-8'))
img = lambda p: 'data:image/png;base64,' + base64.b64encode(p.read_bytes()).decode()
fig = lambda p, cap: f"<figure><img src='{img(p)}' loading='lazy'><figcaption>{cap}</figcaption></figure>" if p.exists() else ''
NAME = {'hum': '上腕骨', 'uln': '尺骨', 'hand': '手', 'bc1': 'ベルクランク', 'rod1': 'ロッド1（竹串）', 'rod2': 'ロッド2', 'rod4': 'ロッド4'}
pair = lambda k: '–'.join(NAME.get(p, p) for p in k.split('-'))
PICK = {'repeat': '再現性重視', 'shape': '形重視', 'open': '開きやすさ重視'}
CFGN = {'A:hum/direct': '案A・ベルクランク1個', 'B:hum/direct': '案B・ベルクランク1個'}


def load(sub):
    d = R2 / sub
    X = {n: J(d / f'{n}.json') for n in ('check_2e', 'loads', 'mirror_check', 'crossing_check', 'feather_bend', 'gate', 'summary')}
    return d, J(d / 'candidates.json'), X


DIRS = {'0': load('2f_sw20'), '8': load('2f_dh8')}
SW = {c: J(DIRS['8'][0] / f'feather_bend_cmax{c}.json') for c in ('0.10', '0.06')}


def thick(plan):
    lv = [v for k, v in plan.items() if k != 'rod1']
    return (max(lv) - min(lv)) * LAYER_T + (LAYER_T - GAP)


LP = max(p[1] for p in PRIM)
bow = lambda r, a, R: r * np.tan(np.radians(a)) - (0 if R is None else r ** 2 / (2 * R))
tipdeg = lambda p: np.degrees(np.arctan(bow(LP, p['alpha_deg'], p['R_mm']) / LP))


def row(dh, c):
    _, _, X = DIRS[dh]; i = c['id']; F = c['F']; fm = X['loads'][i]['force_max_with_spring_N']; fb = X['feather_bend'][i]
    cr = min([t['clear'] for t in X['crossing_check'][i].values()], default=1.0)
    dsg, cfg, pk = c['pick'].split(':')
    st_ = max(X['check_2e'][i]['studs_mm'].items(), key=lambda kv: kv[1])
    return [i, f'{dh}°', f'案{dsg}', PICK.get(pk, pk), f"{c['swivel']:.1f}", f"{F[0]:.1f}", f"{-F[1]:.0f}", f"{c['gain_max']:.2f}",
            f"{c['mc_play']['p95']:.1f}", f"{c['mc020']['p95']:.1f} / {c['mc020']['max']:.1f}", f"{fb['fit_flat']:.1f} → {fb['fit_bent']:.1f}",
            f"{tipdeg(fb['pri']):+.1f}", f"{fb['pri']['camber'] * 100:.0f}",
            f"{F[2]:.1f}", f"{F[5]:.0f}", f"{max(c['rods']):.0f}", f"{c['lever_min']:.1f}", f"{pair(st_[0])} {st_[1]:.1f}",
            f"{X['mirror_check'][i]['min_clear']:.1f}", f"{cr:.2f}", f"{thick(X['check_2e'][i]['plan']):.1f}", f"{max(fm.values()):.2f}"]


HEAD = ['案', '上反角', '構成', '選び方', '首振り（実際の最大）°', '関節角誤差 °', '最小伝達角 °', 'ゲイン最大 °/°', 'ガタだけ 95% °',
        '±0.20 mm 95% / 最大 °', '畳み当てはまり 平ら → 反らせる', '初列の先端の向き °', '初列の反り %', 'はみ出し mm', '最大部品 mm',
        '最長ロッド mm', '最短レバー mm', '最長のピン mm', '左右のすき間 mm', '部品交差の余裕 mm', '体側の重ね厚さ mm', '最大ロッド力 N']
rows = [(dh, c) for dh in ('0', '8') for c in DIRS[dh][1]]
tbl = '<tr>' + ''.join(f'<th>{h}</th>' for h in HEAD) + '</tr>' + ''.join(
    f"<tr{' class=new' if dh == '8' else ''}>" + ''.join(f'<td>{v}</td>' for v in row(dh, c)) + '</tr>' for dh, c in rows)


def gate_stats(dh):
    st = collections.defaultdict(collections.Counter)
    for r in DIRS[dh][2]['gate']:
        s = st[r['C']]; s['n'] += 1; s['pass'] += bool(r['pass'])
    return st


G0, G8 = gate_stats('0'), gate_stats('8')
S0, S8 = DIRS['0'][2]['summary']['configs'], DIRS['8'][2]['summary']['configs']
cfg_rows = ''.join(
    f"<tr><td>{CFGN[k]}</td><td>{S0[k]['seeds_with_feasible']}/{S0[k]['seeds']}</td><td>{G0[k]['pass']}/{G0[k]['n']}</td>"
    f"<td>{S8[k]['seeds_with_feasible']}/{S8[k]['seeds']}</td><td>{G8[k]['pass']}/{G8[k]['n']}</td></tr>" for k in CFGN)

V8, F20 = DIRS['8'][1], DIRS['0'][1]
FB8 = DIRS['8'][2]['feather_bend']
rng = lambda cs, f, n: f"{min(f(c) for c in cs):.{n}f}〜{max(f(c) for c in cs):.{n}f}"

# camber sweep table
OPT = [('制限なし（今の計算）', FB8), ('反り 10% 以下', SW['0.10']), ('反り 6% 以下', SW['0.06'])]
p0 = lambda fb: next(iter(fb.values()))['pri']
sw_head = '<tr><th>初列の反りの上限</th><th>根元の角度 °</th><th>治具半径 mm</th><th>反り %</th><th>中ほどの盛り上がり mm</th><th>先端の向き °</th>' + \
          ''.join(f"<th>{c['id']}</th>" for c in V8) + '</tr>'


def hump(p):
    r = np.linspace(0, LP, 400); z = bow(r, p['alpha_deg'], p['R_mm']); return float(np.max(z - z[-1] * r / LP))


sw_rows = ''.join(f"<tr><td>{n}</td><td>{p0(fb)['alpha_deg']:.0f}</td><td>{p0(fb)['R_mm']:.0f}</td><td>{p0(fb)['camber'] * 100:.1f}</td>"
                  f"<td>{hump(p0(fb)):.1f}</td><td>{tipdeg(p0(fb)):+.1f}</td>" +
                  ''.join(f"<td>{fb[c['id']]['fit_bent']:.2f}</td>" for c in V8) + '</tr>' for n, fb in OPT)

# primary profile figure (model only, no CG data)
pf = DIRS['8'][0] / 'primary_profile.png'
f_, ax = plt.subplots(figsize=(9.5, 2.8), dpi=130)
r = np.linspace(0, LP, 200)
for (n, fb), col in zip(OPT + [('F20（上反角0°）', DIRS['0'][2]['feather_bend'])], ('C3', 'C1', 'C2', 'C7')):
    p = p0(fb); ax.plot(r, bow(r, p['alpha_deg'], p['R_mm']), color=col, lw=2,
                        label=f"{n}: {p['alpha_deg']:.0f}°/R{p['R_mm']:.0f}, 先端 {tipdeg(p):+.1f}°")
ax.axhline(0, color='k', lw=.8); ax.text(LP, .4, '翼の面（上反角 +8° で傾いている）', fontsize=8, ha='right')
ax.set_xlabel('根元からの距離 mm（最長の初列風切 57 mm）'); ax.set_ylabel('翼の面からの高さ mm（上が背側）')
ax.set_aspect('equal'); ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.01, 1)); ax.grid(alpha=.3)
plt.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'sans-serif']
for t in ax.texts + [ax.xaxis.label, ax.yaxis.label] + ax.get_legend().get_texts(): t.set_fontfamily(['Yu Gothic', 'Meiryo', 'sans-serif'])
f_.tight_layout(); f_.savefig(pf); plt.close(f_)

html = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>上反角8°の再計算</title>
<style>
:root{{--bg:#fff;--ink:#1d2430;--mute:#5d6878;--line:#d9dee6;--hi:#fff4d6;--alt:#eef3fb}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#14181f;--ink:#e6eaf0;--mute:#9aa5b5;--line:#2c3440;--hi:#3a3220;--alt:#1d2633}} img{{background:#fff}}}}
:root[data-theme="dark"]{{--bg:#14181f;--ink:#e6eaf0;--mute:#9aa5b5;--line:#2c3440;--hi:#3a3220;--alt:#1d2633}}
body{{background:var(--bg);color:var(--ink);font:15px/1.65 system-ui,'Hiragino Sans','Yu Gothic UI',sans-serif;margin:0 auto;padding:16px;max-width:1200px}}
h1{{font-size:1.5em}} h2{{margin-top:2em;border-bottom:1px solid var(--line)}} .mute{{color:var(--mute)}}
.tw{{overflow-x:auto}} table{{border-collapse:collapse;font-size:13px;white-space:nowrap}} td,th{{border:1px solid var(--line);padding:3px 7px;text-align:right}}
th{{background:var(--bg);white-space:normal;min-width:4.5em;vertical-align:bottom}} tr.new td{{background:var(--hi)}}
img{{max-width:100%;height:auto}} figure{{margin:1.2em 0}} figcaption{{color:var(--mute);font-size:13px}} li{{margin:.3em 0}}
</style></head><body>
<h1>開いた翼に上反角 +8° を付けた再計算</h1>
<p class="mute">2026-11-21。数値はすべて計算値で、実測ではありません。各シードの局所解なので、大域最適の保証はありません。
変えたのは、開いた翼の向き（体の前後軸まわりに +8° 持ち上げる）と、初列風切の先端を3°以上下げる条件だけです。
首振り上限20°、ボールの外形、ガタ、層の規則は F20系と同じです。畳んだ状態の目標（CGの畳んだ翼）は変えていません。</p>

<h2>結論（先に）</h2>
<ul>
<li><b>上反角 +8° でも成立します。</b>解のあるシードは案A 8/8、案B 6/8 でした（案Bの2シードは解なし。F20系でも同じ2シードは層の配置が成立していません）。
確認をすべて通った候補を4つ選びました（V8-1〜V8-4）。</li>
<li>F20系（上反角0°）と比べて、大きな差はありません（首振りは少し増えますが、上限20°の内側です）。
<ul>
<li>関節角誤差 {rng(F20, lambda c: c['F'][0], 1)} → {rng(V8, lambda c: c['F'][0], 1)}°、最小伝達角 {rng(F20, lambda c: -c['F'][1], 0)} → {rng(V8, lambda c: -c['F'][1], 0)}°、
首振り {rng(F20, lambda c: c['swivel'], 1)} → {rng(V8, lambda c: c['swivel'], 1)}°。</li>
<li>±0.20 mm の誤差で95%が {rng(F20, lambda c: c['mc020']['p95'], 1)} → {rng(V8, lambda c: c['mc020']['p95'], 1)}°、左右のすき間 {rng(V8, lambda c: DIRS['8'][2]['mirror_check'][c['id']]['min_clear'], 1)} mm、
最大ロッド力 {rng(V8, lambda c: max(DIRS['8'][2]['loads'][c['id']]['force_max_with_spring_N'].values()), 2)} N。</li>
<li>はみ出しは V8-1〜V8-3 で 0〜0.3 mm、V8-4（開きやすさ重視）だけ 1.6 mm です。</li>
</ul></li>
<li><b>初列の先端を下げる件：</b>当てはまりが一番良い反らせ方で、初列の先端は <b>{tipdeg(p0(FB8)):+.1f}°</b>（翼の面より下）を向きます。
3°以上下げる条件は、当てはまりを悪くしていません。F20系の反らせ方でも、先端はもともと {tipdeg(p0(DIRS['0'][2]['feather_bend'])):+.1f}° でした。</li>
<li><b>注意：その反らせ方は、羽根がかなり反ります。</b>最長の初列（57 mm）の中ほどが、根元と先端を結ぶ線より約 {hump(p0(FB8)):.0f} mm 盛り上がります（反り {p0(FB8)['camber'] * 100:.0f}%）。
F20系の反らせ方も反り9%（約5 mm）でした。鳥の羽根の反りは数%程度です。反りを10%以下に抑えると当てはまりは {rng(V8, lambda c: SW['0.10'][c['id']]['fit_bent'] - FB8[c['id']]['fit_bent'], 1)}、
6%以下では {rng(V8, lambda c: SW['0.06'][c['id']]['fit_bent'] - FB8[c['id']]['fit_bent'], 1)} 悪くなります（下の表）。どちらでも先端は3°以上下がります。</li>
<li>シミュレータでの再現は、Python の計算と 0.36° 以内で一致しました。</li>
</ul>

<h2>決めていただきたいこと</h2>
<ul>
<li>初列の反りの強さ：当てはまり重視（反り14%、中ほど約9 mm盛り上がる）か、見た目重視（反り7%や3%）か。羽根を治具で曲げるなら、試しに1本曲げて見ていただくのが早いと思います。</li>
<li>段階3に進む候補：V8系（上反角+8°）にするか、F20系（0°）に戻すか。機構の数値はほぼ同じなので、見た目で決めてよいと考えます。</li>
</ul>

<h2>候補の比較（黄色＝上反角 +8° の新しい候補）</h2>
<div class="tw"><table>{tbl}</table></div>
<p class="mute">初列の先端の向き：最長の初列風切の根元と先端を結ぶ線が、翼の面から何度上（＋）か下（−）か。反り：根元と先端を結ぶ線からの最大の盛り上がり ÷ 50 mm。
首振りは全姿勢の中での最大値です。ガタだけ・±0.20 mm は寸法誤差のモンテカルロの95%点（関節角のずれ）です。</p>

<h2>初列の反りの強さと当てはまり（先端を3°以上下げる条件のもと）</h2>
<div class="tw"><table>{sw_head}{sw_rows}</table></div>
<p class="mute">V8-n の列は「反らせた羽根での畳み当てはまり」（小さいほど良い）。参考：平らな羽根では {rng(V8, lambda c: FB8[c['id']]['fit_flat'], 1)}。
反りの候補は根元の角度5°刻み・治具半径の飛び飛びの値なので、6%以下の上限では次に小さい3%が選ばれています。次列風切は上限なしで、反りは
{rng(V8, lambda c: FB8[c['id']]['sec']['camber'] * 100, 0)}%（こちらも強めです）。</p>
{fig(pf, '最長の初列風切の形（翼の面からの高さ）。根元で上向きに出し、治具の半径で下向きに曲げる。先端が0より下なら翼の面より下がる。')}

<h2>構成ごとの結果</h2>
<div class="tw"><table><tr><th>構成</th><th>0°：解ありのシード</th><th>0°：全確認を通過</th><th>+8°：解ありのシード</th><th>+8°：全確認を通過</th></tr>{cfg_rows}</table></div>
<p class="mute">「全確認」＝層の配置（41姿勢）、部品の交差、左右の干渉。+8° の計算は、0° の最終集団を「畳んだ姿勢がそのまま保たれるように」変換して始めています。</p>

<h2>図（V8-1〜V8-4）</h2>
{''.join(fig(RL / f"cand_{c['id']}.png", f"{c['id']}（案{c['dsg']}）畳んだ状態：青＝CGの畳んだ翼、橙＝モデルの羽根（平ら）、赤＝骨、黒＝リンク。") for c in V8)}
{''.join(fig(DIRS['8'][0] / f"stack_{c['id']}.png", f"{c['id']} の実高さの層配置。上段：上から見た図（色＝高さ）。下段：横から見た図（黒の縦線＝ピン・玉継手の軸）。") for c in V8)}
{''.join(fig(DIRS['8'][0] / f"mirror_{c['id']}.png", f"{c['id']}：左右の翼が最も近づく姿勢（すき間 {DIRS['8'][2]['mirror_check'][c['id']]['min_clear']:.1f} mm）。") for c in V8)}

<h2>やっていないこと・限界</h2>
<ul>
<li>上反角 +8° は資料（ホバリング中の写真1枚で約8〜10°）からの仮定で、カワセミの実測ではありません。滑空中の猛禽類はむしろ下反角（−6〜−13°）です。</li>
<li>シミュレータは羽根を平らに描くので、初列の反りや先端の下がりは表示されません（上の図のみ）。</li>
<li>シミュレータのはみ出しは体を楕円体で近似しているので、V8-4 では Python（メッシュ）の 1.6 mm より大きく表示されます（7.3 mm）。</li>
<li>ボールの外形は直径3 mmのままです（実物の M2 ロッドエンドは玉の幅が約4 mm）。駆動の取り回し、強度、姿勢の間での一瞬の接触は段階3の CAD で確認します。</li>
</ul>
<p class="mute">体の形の参照：「Common Kingfisher」Alejandro Guzmán（Sketchfab）CC BY-SA 4.0 を縮尺・変形して使用。図に体の形が含まれるため、このページも CC BY-SA 4.0 です。</p>
</body></html>"""
out = RL / 'stage2_2f_dh8_report.html'
out.write_text(html, encoding='utf-8')
print(out, out.stat().st_size // 1024, 'KB')
