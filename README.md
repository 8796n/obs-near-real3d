# near Real 3D

**単眼深度推定で 2D 映像をリアルタイムに Side-by-Side 立体化する OBS 32 用ネイティブ動画フィルター。**
XREAL One シリーズの「REAL 3D」(平面映像を擬似立体で見せるモード)を、専用ハードなしに OBS の中だけで擬似的に再現する。

> モジュール名/バイナリ: `obs-near-real3d` ／ フィルター表示名: **near Real 3D (SBS)**

ソースに適用すると、そのソースのプレビュー/出力が **左右2枚(SBS)の立体映像**になる。
1枚絵から単眼深度を推定し、深度から左右の視差を作って GPU シェーダで warp する。
`obs-backgroundremoval` と同じ「ONNX 推論を OBS フィルター内で回す」構造で、外部アプリ不要。

---

## 仕組み: 単眼深度から視差をつくる

立体視の原理は単純で、**左右の眼で見える像が水平にわずかにずれている**こと(視差/parallax)。
近い物ほどずれが大きく、無限遠ではずれない。このフィルターは 1 枚の 2D フレームから

```
RGB フレーム ──▶ ① 単眼深度推定 ──▶ ② 深度→視差 ──▶ ③ バックワード warp ──▶ 左右 SBS 出力
                     (ONNX/GPU)        (収束面+強度)      (各眼で逆方向)
```

という流れで擬似的な左右像を合成する。

### ① 単眼深度推定 (monocular depth)

- **Depth Anything V2 (Small)** を ONNX 化し、**ONNX Runtime + DirectML**(DX12 GPU)で推論。
- 入力は **392×392** の正方に縮小(`INFER_SIZE`)。縮小は**2段の面積平均**（大縮小の一発バイリニアはエイリアス＆圧縮バンディングを通すため）＋任意で**軽い平滑**を掛け、トーンジャンプが深度の段差として拾われるのを抑える。出力は **相対的な逆深度**(近いほど大)を `[0,1]` に正規化したもの。絶対距離ではなく「手前/奥の順序」を表す。
- OBS の描画スレッドとは別の**ワーカースレッドで非同期推論**し、結果を深度テクスチャ(`R32F`)としてアップロード。**warp(表示)は毎フレーム、深度の更新だけを指定レート**(既定 15fps)で回すことで負荷を抑える。

### ② 深度 → 視差 (disparity)

各画素の深度 `d` から、水平方向のずらし量(視差)を作る:

```hlsl
disp = strength * (d - convergence)
```

- **convergence(収束面)**: 視差ゼロになる深度 = スクリーン面。`d > convergence`(手前)は正の視差で**飛び出し**、`d < convergence`(奥)は負で**引っ込む**。
- **strength**: 基線長(両眼間隔)に相当するスケール。`0` で視差なし=完全フラット(深度処理は走ったまま立体だけ無効。A/B 比較用)。
- **左右で符号を反転**: 左眼と右眼は逆方向にずらす(`swap` で入れ替え可)。

### ③ バックワード warp と SBS 合成

出力フレームを左右半分に割り、**各出力画素について「ソースのどこをサンプルすべきか」を逆算**する(inverse/backward warp):

```hlsl
// 出力 uv からどちらの眼か判定し、その眼のフル画面 uv(euv)に復元
// 深度から視差 disp を求め、サンプル元 uv を眼ごとに逆方向へ水平シフト
suv = euv.x + sign(eye) * disp * 0.5;
color = image.Sample(linClamp, suv);   // ソースを引いてくる
```

- 画素を**飛ばす** forward warp ではなく、出力から**引いてくる** backward warp なので穴が空かず実装が単純・GPU 1 パスで済む。
- 代償として、**前景の陰に隠れていた背景(disocclusion)は黒穴でなく引き伸ばし**になる。急な深度段差(前景シルエット vs 遠景)で視差を強くすると輪郭が裂ける——DIBR 特有の "rubber-sheet" アーティファクト。**Edge softening**(深度エッジの Gaussian 平滑)で深度段差をなだらかにして緩和する(mesh-warp 相当の近似)。視差量に比例して悪化するので、効果は控えめが基本。

### 補足: 時間安定化と色空間

- **時間安定化(flicker 対策)**: 単眼深度モデルはフレーム独立推論なので、画面のどこかが動くと**静止領域の深度まで毎フレーム揺れる**(breathing。MiDaS/DPT/DA V2 共通の限界)。対策として **自前実装の軽量オプティカルフロー(ピラミッド型 Lucas-Kanade、OpenCV 非依存)で前フレーム深度を動き補償ワープ → 残差に応じて適応合成**し、加えて正規化レンジを EMA 平滑。動く前景の残像を抑えるため、**現在深度の局所範囲へ過去深度を制限する history clipping** と **高速移動・フロー失敗付近の履歴を捨てる reactive mask** を常時併用する(`flow_stabilizer.hpp`)。
- **シーンチェンジ対策**: 深度は非同期で遅れるため、カット直後は前シーンの深度で新フレームを歪めて視差が一瞬乱れる。対策として描画スレッド側でフレーム間差分から**カットをキャンバス FPS で検出**し、(1)**視差を一瞬フラット化(2D)**して前シーンの深度で歪めない、(2)推論レートを無視して**追加推論を即投入**、新シーンの深度が届いたら視差をなめらかに戻す。
- **深度と映像の一致(任意)**: 通常は最新フレームに*少し過去の*深度を当てるため、動き中は深度シルエットが画像から遅れる。設定の **深度と映像を一致** を有効にすると、直近のフルレスフレームをリング保持して**いま手元にある深度に対応するフレームを warp**し、この定常ズレを解消する。実測レイテンシぶん映像出力が遅れ、**音声もソースの同期オフセットで自動的に同じだけ遅延**して A/V を保つ。追加 VRAM と出力遅延と引き換え(ライブ配信では遅延は配信側バッファに吸収されやすい)。
- **色空間(sRGB)**: フィルターは `OBS_SOURCE_SRGB` を宣言し、リニアライト・パイプラインの状態を取得/描画の各パスで整合させる。これにより sRGB タグ付きソース(動画など)が **decode されっぱなしで暗く沈むことなく**正しい明るさで立体化される。

---

## インストール（Windows）

[Releases](https://github.com/8796n/obs-near-real3d/releases) から最新版を入手:

- **インストーラ（推奨）**: `near-real3d-x.y.z-windows-x64-installer.exe` を実行。`%PROGRAMDATA%\obs-studio\plugins\obs-near-real3d\` へ配置されます（管理者不要）。**OBS を閉じてから**実行し、終わったら OBS を再起動。
- **ポータブル ZIP**: `near-real3d-x.y.z-windows-x64.zip` を展開し、中の `obs-near-real3d` フォルダを **`%PROGRAMDATA%\obs-studio\plugins\`** に置く（→ `…\plugins\obs-near-real3d\bin\64bit\…`）。

導入後、ソースを右クリック → **フィルタ** → **+** → **near Real 3D (SBS)**。

> 署名前のため、初回は SmartScreen/AV の警告が出ることがあります（「詳細情報」→「実行」）。
> 必要環境: **Windows x64 / DirectML が動く DX12 GPU**。VC++ ランタイムは OBS 同梱で通常そのまま動作。

## 動作環境(検証済)

- **OBS Studio 32.1.2**(ABI 一致のヘッダをこのバージョンで取得)
- **Windows / DirectML が動く DX12 GPU**(検証: RTX 3060) — DirectML EP のため現状 **Windows x64 専用**
- ビルド: **MSVC**(VS 2022/18, cl 19.5x)+ 同梱 **CMake / Ninja**
- 深度モデル `depth_anything_v2_small.onnx`(Apache-2.0)を同梱(リリース成果物 / `models/`)

## ソースからビルド (Windows)

```powershell
.\setup.ps1     # 依存取得: obs ヘッダ + obs.lib / ONNX Runtime(DirectML)
.\build.ps1     # MSVC + Ninja でビルド → build\obs-near-real3d.dll
.\install.ps1   # %PROGRAMDATA%\obs-studio\plugins\obs-near-real3d\ へ配置(管理者不要)
```

`install.ps1` は plugin DLL・`onnxruntime.dll`・`DirectML.dll`・
`near-real3d.effect`・深度モデルを所定の `bin\64bit` / `data` に配置する。

> Windows の OBS ユーザープラグインパスは `%APPDATA%` ではなく
> **`%PROGRAMDATA%\obs-studio\plugins\<module>\bin\64bit`**
> (`frontend/widgets/OBSBasic.cpp` の `GetProgramDataPath` + `/bin/64bit`)。

## 使い方

1. OBS を起動 → ソースを右クリック → **フィルタ** → **+** → **near Real 3D (SBS)**。
2. そのソースのプレビュー/出力が Half-SBS の立体になる。
3. フィルタ設定(**3D / 安定化 / デバッグ** の3グループ。各項目はホバーで詳細ツールチップ):
   - **3D strength** 0–4(視差量。**0=視差なし(フラット)** で深度処理は走ったまま立体だけ無効化 → A/B 比較用)
   - **Convergence**(スクリーン面=視差ゼロの深度。上げるほど手前に飛び出す)
   - **Swap left/right**(立体が反転して見えるとき)
   - **Full-SBS**(**既定 on**。off=Half-SBS。下記参照。横2倍出力なのでベース/出力解像度も横2倍に)
   - **SBS size**(片眼の出力解像度。`ソース基準`＝従来どおりソース寸法 / `1080p`＝片眼 **1920×1080 固定**。Full-SBS なら **3840×1080**、Half-SBS なら **1920×1080** を出力し、ソース寸法に依らずグラス向けの SBS フレームになる。OBS のキャンバス/出力も同じサイズにすると 1:1)
   - **Letterbox each eye**(既定 off。アスペクト維持で各眼にソースを収め余白を黒帯に。**非16:9 ソースを固定 SBS サイズで扱うとき**用。眼=ソースのアスペクトが一致なら無効)
   - **Depth update rate**(GPU 負荷。10/15/24/30/60fps、既定 15。**24fps は映画ソース向け**)
   - **Stop inference when static**(既定 on。ほぼ静止している間は ONNX 推論をスキップしキャッシュ深度を再利用。表示=warp は毎フレーム継続)
   - **Static threshold**(0–8、0.1刻み、既定 1.0。縮小フレームの平均色差がこれ以下なら静止扱い。画面キャプチャはほぼ 0 で即スキップ、カメラ入力はノイズに合わせ上げる)
   - **Stabilize depth** / **Stabilization strength**(時間平滑化、チェック付きグループの**ヘッダーチェックが ON/OFF**。既定 on / 強さ 0.4。上げるほど安定・動きは僅かに遅延・残像寄り。残像対策(history clipping + reactive mask)は常時適用)
   - **Edge softening**(0–1、既定 0.3。深度エッジを平滑化し、強い視差での輪郭破綻=ラバーシートを低減)
   - **Smooth inference input**(推論入力の平滑、既定 on。深度モデルに渡す**縮小画像だけ**を軽くぼかし、圧縮のトーンジャンプ/ブロックノイズが深度の段差として拾われるのを抑える。**最終出力の解像感には影響しない**=warp はフル解像度を使用。ノイズの少ない素材ではオフ可)
   - **Frame-matched depth(深度と映像を一致)**(既定 off。映像表示を実測の深度レイテンシ(~0.1s)ぶん遅らせ、各フレームを**その時点の深度**で warp して動き中の“深度が遅れる”ズレを解消。**追加 VRAM＋出力遅延**と引き換え。音声は自動で同じだけ遅延し A/V を保つ。ライブ配信では遅延は配信側バッファに吸収されやすい)
   - **Show depth map**(デバッグ: **左=深度マップ / 右=元画像**を並べて表示。深度と原画を見比べられ、静止スキップ中は深度が固定・原画のみ更新されるので、静止判定しきい値の効きを目視確認しやすい)

### Half-SBS と Full-SBS

- **Half-SBS**: 出力はソースと同寸 W×H。各眼を W/2 に 2:1 横圧縮した真のアナモルフィック SBS(3D TV / VR / SBS 対応グラスが左右に引き伸ばして見る標準形式)。フル解像度ソースをサンプルするので拾いは綺麗。
- **Full-SBS（既定）**: 出力 2W×H、1 眼フル解像度。シェーダは同じで `get_width` が 2W を返すだけ。
  **注意**: OBS のキャンバス/出力解像度が W のままだと 2W ソースが縮小され結局 Half 相当に戻る。効果を出すにはチェーン全体を 2W 幅に(例: ベース/出力 3840×1080)。

**グラス（XREAL 等）へ送るには**: 最終フレームのサイズを決めるのは **OBS のベース/出力解像度**（プラグインは変えられない）。`SBS size = 1080p` にすると片眼が常に 1920×1080 になり、**Full-SBS → キャンバス 3840×1080**／**Half-SBS → 1920×1080** に合わせれば、ソース寸法に依らずグラスにそのまま出せる。非16:9 ソースは **Letterbox each eye** で各眼を 16:9 に収める（黒帯）。16:9 ソースなら `ソース基準` のままでも 3840×1080 キャンバスで正しく出る。

SBS 対応のグラス/ディスプレイへ送るなら、SBS 出力したシーンを全画面表示し、デバイス側の SBS モードで見る(その場合は Half-SBS が標準)。

## パフォーマンス(GPU 負荷)

深度推定は**連続でニューラルネットを GPU 推論**するため本質的に重い(GPU を OBS の描画と共有)。RTX 3060 / DirectML 実測。

**精度(量子化)別**(DA V2 small 392²):

| 形式 | 1 回 | サイズ | FP32 との誤差 | 備考 |
|---|---|---|---|---|
| FP32 | 20.0 ms | 99MB | — | |
| **FP16(既定)** | **10.8 ms** | 50MB | 0.012 | ~1.85x 速・無劣化・FP32 I/O ドロップイン |
| INT8 動的 | 25.8 ms | 27MB | 0.67 | DirectML では**遅い**(非加速) |
| INT8 静的 QDQ | 569 ms | 27MB | 1.63 | DirectML では激遅。NPU 向けの正しい形式 |

→ **PC/DirectML では FP16 が最適**。**入力解像度別**(FP32): 518²=42.9ms / 392²=20.6ms / 308²=11.1ms。
**dynamic-axes は DirectML で約 5 倍遅いので固定サイズ必須**。

**深度ワーカーは「ONNX 推論 → 時間安定化」を直列処理**する点に注意。RTX 3060 実測で時間安定化（自前オプティカルフロー）は約 **28ms**、安定化全体で約 **35ms**。ONNX(FP16 10.8ms)と合わせると **1 深度フレーム ≈ 45ms** になり、**時間安定化 ON では実効の深度更新は ~20fps が上限**（それ以上を選んでもワーカーが追いつかず一部フレームは間引かれる）。**30/60fps を活かすなら時間安定化を OFF**（ONNX のみ ~11ms）にするか、より高速な GPU/CPU が要る。warp（表示）は常に毎フレーム。

軽くしたいとき: **Stop inference when static**(既定 on)/ **Depth update rate を下げる** / **時間安定化を OFF** / 入力ソース解像度を下げる / 推論サイズを小さく再エクスポート(`INFER_SIZE` を合わせる)。

> 推論解像度を下げても **SBS の出力解像度は不変**(深度マップの精細さだけが変わる)。深度は低周波情報なので 392→308 でも立体感への影響は小さい。

> 計測の見方: 静止スキップが効くと GPU の**絶対負荷・消費電力(W)は実際に下がる**(クロックが例 1900→700MHz)が、タスクマネージャ等の **使用率%** はクロック低下に伴い数秒で元水準へ戻って見える(DVFS による見かけ上の上昇)。実負荷は **GPU Clock(MHz) / Power(W)**(GPU-Z / HWiNFO64 / Afterburner)で見る。

## 既知の制約 / 今後

- フィルタ作成時に ONNX セッション初期化 + モデルロードで **~1.3 秒**(インスタンスごと)。共有セッション化で短縮余地。
- 深度は **正方(392²)** で推論し正方深度を warp に使用(相対深度なので可)。アスペクト厳密化は今後。
- backward warp ゆえ disocclusion は黒穴でなく引き伸ばし。**急な深度段差 × 強い視差で輪郭が裂ける(DIBR ラバーシート)**。`Edge softening` で緩和、根本解決は occlusion-aware forward warp + inpainting(重い)。
- **深度 flicker は単眼深度モデル固有の限界**(時間的一貫性なし)。フロー誘導の時間安定化で実用域まで抑制(上記「仕組み」参照)。
- **平坦/構造の乏しい映像(空・星空・霧・グラデーション・抽象)は苦手**: そこに*実際の奥行きが無い*ため、単眼深度は輝度やテクスチャから深度を捏造する(星空なら天の川の帯を「近い」と誤推定する等)。視差を付けると不自然になりやすいので、この種のソースでは **3D強度を下げる/オフ**を推奨。逆に**前景・背景が明確な被写体**(人物・物・室内・風景)が得意領域。
- **真の動画深度モデル(Video Depth Anything)は不採用**: 時間一貫性が KV-cache(ステートフル)依存で公式に ONNX 書き出し不可、性能的にも 3060+DirectML で非現実的。よってフロー誘導で代替。
- 深度推論は既定 15fps なので激しい動きで段付きが残りうる(warp は毎フレーム)。Depth update rate を上げると緩和。
- sRGB 厳密化は対応済。DA V2↔DPT 切替 / 共有セッション化 / 非同期 readback は今後の候補。
- 現状 Depth Anything V2 Small 固定(別サイズ/別モデルへ差し替え可能)。

## ファイル構成

- `src/plugin-main.cpp` — フィルター本体(フレーム取得・推論オーケストレーション・warp 描画・SBS 出力)
- `src/depth_infer.hpp` — ONNX Runtime(DirectML)単眼深度推論ラッパー
- `src/flow_stabilizer.hpp` — フロー誘導の時間安定化 + 正規化
- `data/near-real3d.effect` — 深度→視差 backward warp + SBS シェーダ(OBS effect)
- `setup.ps1` / `build.ps1` / `install.ps1` — 依存取得 / ビルド / ローカル配置
- `package.ps1` / `installer/obs-near-real3d.iss` — 配布物（ZIP＋Inno Setup インストーラ）生成
- `.github/workflows/release.yml` — タグ push で CI ビルド→Release 添付
- `deps/get_onnxruntime.ps1` — ONNX Runtime(DirectML)の取得（PowerShell のみ）

## リリース（メンテナ向け）

配布物は **GitHub Actions** が自動生成します（`.github/workflows/release.yml`）。

1. **一度だけ準備**: 深度モデル `depth_anything_v2_small.onnx` を安定した URL（GitHub Release アセット等）に置き、リポジトリの **Actions 変数**（Settings → Secrets and variables → Actions → Variables）に2つ設定。モデルは巨大なので Git には含めない。
   - `MODEL_URL` … モデルの直リンク
   - `MODEL_SHA256` … モデルの SHA-256（**タグリリースでは必須**。未設定だと CI が失敗）。現行モデルの値:
     `593878883CF16CA39A58895390A251F1BC2858B9C1256FC167E65DBBB230435D`
     （`Get-FileHash depth_anything_v2_small.onnx -Algorithm SHA256` で確認可）
2. **リリース**: `v0.1.0` のような **タグを push** すると、CI が 依存取得 → `-DPLUGIN_VERSION=<タグ>` でビルド → `package.ps1` で ZIP＋インストーラ生成 → その Release に添付（バージョンはタグ由来。手動実行時は `0.1.<run_number>`）。
3. **ローカルでも生成可**: `setup.ps1` → `build.ps1` → `package.ps1 -Version 0.1.0` で `dist/` に ZIP（＋Inno Setup があればインストーラ）。初回や CI 整備前の手動アップロードに。

## ライセンス

**GPL-2.0-or-later**(OBS 本体と同じ)。GPLv2 と非互換な Apache-2.0 の OpenCV を外し、オプティカルフローを自前実装したことで GPLv2 を選択できる。
同梱物とライセンス: ONNX Runtime(MIT)/ DirectML(再配布可)/ Depth Anything V2 Small(Apache-2.0)。
詳細は [`LICENSE`](LICENSE)(GPLv2 全文)と [`THIRD_PARTY_LICENSES`](THIRD_PARTY_LICENSES) を参照（各全文はリリース成果物の `licenses/` に同梱）。
