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
- `results/` — 最適化の出力（シード別）。

## バージョン
タグ `v2` `v3` `v4`。それ以降のコミットは表示・操作の改善。

## 必要なもの
Python 3 + numpy + scipy、Node.js + playwright（検証用のみ）。スクリプトは作業フォルダで書いたもので、パスはスクリプトと同じ場所に入力ファイルがある前提。
