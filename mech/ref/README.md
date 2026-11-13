# 参考3Dモデル（サードパーティ）

| ファイル | モデル | 作者 | ライセンス | 種類 | 入手元 |
|---|---|---|---|---|---|
| `models/common_kingfisher.glb` | Common Kingfisher（ヨーロッパカワセミ *Alcedo atthis*） | Alejandro Guzmán | CC BY-SA 4.0 | 手作りCG（骨格なし、部品ごとに閉じたメッシュ） | Sketchfab（URL 未確認：要記入） |
| `models/belted_kingfisher.glb` | Belted Kingfisher（アメリカヤマセミ *Megaceryle alcyon*） | RISD Nature Lab（Edna Lawrence Nature Lab, Rhode Island School of Design） | CC BY 4.0 | 剥製の3Dスキャン（止まり木・台座つき、穴の多い開いたメッシュ） | Sketchfab（URL 未確認：要記入。作者ページ https://sketchfab.com/RISDNaturelab/models） |

- 作者・ライセンスは、モデルを置いた人（リポジトリの所有者）が Sketchfab のページで確認した内容。glb ファイル自体にはメタデータが入っていない。
- URL は確認できていない。ダウンロードしたページの URL を表に書き足すこと。
- **git には入れない**（ライセンスの扱いを簡単にするため）。`models/` と `derived/` は `.gitignore` 済み。手元で使う人は Sketchfab から各自ダウンロードして `mech/ref/models/` に上の名前で置く。

## 派生物とライセンス（どれも git に入れない）
- `derived/ck_mesh.js`：Common Kingfisher を間引き・座標変換したもの（シミュレータ表示用）。**CC BY-SA 4.0**（元の作品の継承条件）。作者表示はファイル先頭のコメントにもある。
- `../results/stage1/` の図のうち、Common Kingfisher の形状を描いたもの（`ck_vs_v4.png`、`sim_*.png` など）も CC BY-SA 4.0 の派生物として扱う。Belted Kingfisher を描いたもの（`bk_sections.png`）は CC BY 4.0 の派生物（作者表示が必要）。
- モデルの形を描いた画像は `results/stage1/*.png` と `results/local/` に置き、どちらも git に入れない。
- シミュレータ本体（`kingfisher_wing_fold.html`）は、別ファイルの `derived/ck_mesh.js` を読み込んで表示するだけで、メッシュを含まない。ファイルが無ければ楕円体だけで動く。作り直すには `tools/stage1/export_mesh_js.py`。

## 変換方法（どちらも推定を含む）
- Common Kingfisher：翼の殻（左右の閉じたメッシュ）の最長の弦を翼長 78.8 mm（BTO、生体成鳥）に合わせて縮尺 ×0.677。胴の第1主成分を前後軸、z=0 面を正中面とした。モデルに骨格がないので、肩の位置は規則で置いた推定（`tools/stage1/ck_align.py`）。
- Belted Kingfisher：スキャンの単位を mm とみなした（翼長の目安 ≈ 150〜160、実物 155〜165 mm と矛盾しない）。胴の左右対称面を当てはめて体の座標系を作った（`tools/stage1/bk_sym.py`）。

## DigiMorph: Alcedo cristata（マラカイトカワセミ）FMNH 319990
- 出典: DigiMorph Staff, 2004, "Alcedo cristata" (On-line), Digital Morphology. https://digimorph.org/specimens/Alcedo_cristata/whole/
  スキャン: Matthew Colbert（UT High-Resolution X-ray CT Facility）、画像処理: Amy Balanoff。標本: Field Museum of Natural History。
- 使ったもの: 公開されているスライス動画・回転動画（mp4）。`models/digimorph_acristata/` に手元のみ（git に入れない）。著作権は DigiMorph。
- 生データ（CT 885枚）は MorphoSource media 000712732（CC BY-NC 4.0、申請制）。未取得。
- 派生物: CT の図は `results/local/` のみ。数値（`results/stage2/ct_wing.json`）だけ git に入れる。
