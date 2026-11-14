# カワセミ翼 開閉リンク

止まり木の姿勢で翼を広げる・畳む動きを、モーター1個のリンク機構で再現する展示用ロボットの設計リポジトリ。

## 中身
- `kingfisher_wing_fold.html` — シミュレータ（ブラウザで開くだけ）。右上で全バージョンと却下案を切り替え可能。
- `wing-fold-linkage-design.md` — 設計メモ（根拠データ、寸法、関節の種類、検証結果、最適性の注意、先行研究）。
- `v4_views.png` — v4 の4視点の画像。
- `wing_linkage_simulator.html` — v1（羽ばたき中の折れ込み機構、別物）。
- `tools/` — 設計に使ったスクリプト。
  - `v4pose.py`：畳み姿勢を胴の輪郭に合わせて最適化し、肩の斜め1軸を求める → `results/v4_pose.json`
  - `v4opt.py elbow|wrist`：肘・手首系統の寸法合成（差分進化） → `results/v4_*_all.json`
  - `v4optB.py`：肘ベルクランクを上腕骨に載せる別構成（却下案 r4）
  - `mkv4.py`：結果からシミュレータのバージョン項目を作り、`part1.html` に差し込む。シミュレータ本体は `cat tools/part1.html tools/part2.html` で組み立てる。
  - `v3opt.py` ほか：v3 以前の合成・検証スクリプト。
  - `*.js`：Playwright による検証（全バージョンの指標、±0.1 mm 感度、スクリーンショット、視点操作）。
  - `stage1/`：段階1（3Dモデルで胴と畳み姿勢を確認）。`ck_align.py`（CGを肩座標へ）、`eval_v4_ck.py`（v4をメッシュで評価・図）、`sensitivity.py`、`proportions.py`、`fit_ellipsoid.py`、`bk_*.py`（ヤマセミのスキャン）、`export_mesh_js.py`（シミュレータ用メッシュ）、`dump_path.js`（シミュレータから動きを書き出す）、`shots.js`。
  - `stage2/`：段階2（畳み姿勢とリンク寸法の多目的最適化）。`body.py`（CG体のSDF）、`kin.py`/`mech.py`（運動学・合成、6構成）、`problem.py`（目的と制約）、`run.py`/`batch.py`（NSGA-III、シード並列）、`pose_only.py`、`analyze.py`（パレート統合・候補・MC）、`cand_figs.py`、`export_sim.py`（候補をシミュレータへ＋照合）、`standalone.py`、`report.py`（結果ページ、手元のみ）、`ct_volume.py`/`ct_wing.py`/`ct_bones.py`/`ct_interpret.py`/`ct_frame.py`（CTの計測。CTデータは `ref/models/` の手元のみ）、`pose_tilt.py`（案B：手首軸を傾けた姿勢だけの計算）、`report_b.py`（案Bの結果ページ、手元のみ）。`layer_figs.py`（段階2d：部品の層の割り当て図）、`report_2d.py`（段階2dの結果ページ、手元のみ）。`stack.py`（段階2e：実高さの層配置。最適化の制約）、`stack_check.py`・`loads.py`（段階2dの追加検証。`loads.py 2e` で段階2e）、`check_2e.py`（候補の全41姿勢の層確認とロッド4の曲げの伸び）、`recheck_2d.py`、`stack_figs.py`（層配置の図）、`report_2e.py`（段階2eの結果ページ、手元のみ）。段階2eは `batch.py 300 300 8 15 45 2e`、`analyze.py 45 2e` など引数 `2e`（出力は `results/stage2/runs_2e/`、`results/stage2/2e/`）。環境変数 `STACK_GRADED=1` で層の制約を段階的にする（`graded_hh.sh`）。段階2f（左右の翼が体の中心線で当たらない制約）は環境変数 `MIDLINE=1` を付けて同じスクリプトを実行する（`batch.py 300 300 8 16 45 2f`、`analyze.py 45 2e` など。出力は `results/stage2/runs_2f/`、`results/stage2/2f/`、候補名 F1…）。`mirror_check.py`（左右の翼を鏡写しにした干渉チェック）、`mirror_figs.py`（その図）、`screen_mc.py`（前線をモンテカルロで選別）、`report_2f.py`（段階2fの結果ページ、手元のみ）。`run.py`/`batch.py`/`analyze.py`/`cand_figs.py`/`export_sim.py` は引数 `tilt` で案B（出力は `results/stage2/runs_tilt/`、`results/stage2/tilt/`）。`tilt 2d`（`batch.py` は `tilt` のみ）で段階2d（干渉・ガタ・直結ロッドの構成、出力は `results/stage2/runs_2d/`、`results/stage2/2d/`）。
- `ref/` — 参考3Dモデルと派生物（git には入れない。手元に置く）。作者・ライセンスは `ref/README.md`。
- `results/` — 最適化の出力（シード別）。`results/stage1/` は段階1の数値、`results/stage2/` は段階2の候補・要約・パレート図（個々の実行は `runs/`）。CGの形を含む図は `results/local/`（git 外）。

## バージョン
タグ `v2` `v3` `v4`。それ以降のコミットは表示・操作の改善。

## 必要なもの
Python 3 + numpy + scipy、Node.js + playwright（検証用のみ）。v4 以前のスクリプトは作業フォルダで書いたもので、パスはスクリプトと同じ場所に入力ファイルがある前提。

段階1以降（`tools/stage1/`）はリポジトリ内のパスを自分で解決する。準備：
```
uv venv .venv --python 3.13 && uv pip install --python .venv numpy scipy trimesh matplotlib rtree shapely networkx scikit-image fast_simplification pymoo
cd mech/tools && npm install && npx playwright install chromium
node stage1/dump_path.js v4             # シミュレータから v4 の動きを書き出す
../../.venv/Scripts/python stage1/eval_v4_ck.py
cd stage2 && ../../../.venv/Scripts/python batch.py 400 300 12 15 45   # 72回、16コアで約3時間
../../../.venv/Scripts/python analyze.py 45 && ../../../.venv/Scripts/python export_sim.py
```
シミュレータの「体の形」で CG メッシュを選べる（`ref/derived/ck_mesh.js` を読み込む。CC BY-SA 4.0）。
