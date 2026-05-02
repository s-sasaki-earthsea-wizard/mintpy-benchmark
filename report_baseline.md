# FernandinaSenDT128 ベースライン実行時間レポート

実施日: 2026-04-26 (JST)
対象: `smallbaselineApp.py FernandinaSenDT128.txt` の全 18 ステップ
計測スクリプト: [benchmark/run_bench.sh](run_bench.sh)
ログ一式: [benchmark/logs_baseline/](logs_baseline/)

---

## 1. マシン情報

| 項目 | 値 |
|---|---|
| OS | Ubuntu 24.04.3 LTS (kernel 6.17.0-20-generic) |
| CPU | Intel Core Ultra 9 285H — 16 コア / 16 スレッド, max 5.4 GHz |
| L2 / L3 cache | 28 MiB / 24 MiB |
| メモリ | 93 GiB (Swap 8 GiB) |
| GPU | NVIDIA GeForce RTX 5080 (16 GiB, Blackwell sm_120) |
| GPU driver / CUDA | 590.48.01 / CUDA 13.1 driver, nvcc 13.0 |
| データ置き場 | CIFS マウント `//192.168.10.132/EW-NAS-Atoll` (NAS 経由) |
| Python | 3.12.3 (`.venv/`, uv 管理) |
| 主要ライブラリ | numpy 2.4.4, scipy 1.17.1, h5py 3.16.0, dask 2026.3.0, torch 2.11.0+cu128 (未使用), cupy-cuda12x 14.0.1 (未使用) |

詳細は [logs_baseline/machine_info.txt](logs_baseline/machine_info.txt) 参照。

> **注意**: 入力データ約 1 GB は NAS (CIFS / 1GbE) 上にあり、I/O がボトルネックになるステップ (`load_data` など) はローカル NVMe より遅い可能性があります。

---

## 2. データセット規模

[FernandinaSenDT128/mintpy/inputs/ifgramStack.h5](../FernandinaSenDT128/mintpy/inputs/) の主な寸法:

| 項目 | 値 |
|---|---|
| 干渉ペア数 (interferograms) | 288 |
| 取得日数 (acquisitions) | 98 |
| ピクセル数 | 270,000 (おそらく 450 × 600) |
| 反転対象有効ピクセル | 157,667 (58.4%) |

---

## 3. ステップ別実行時間

`wall` は `/usr/bin/time` 計測の壁時計時間（Python プロセス起動含む）、`internal` は MintPy 自身が出力した `Time used:` 値（純粋な処理時間）。`startup` ≈ wall − internal は 1 ステップあたりの CLI ブート + import コスト。

| # | step | wall (s) | internal (s) | startup (s) | max RSS (MB) | 割合 (internal) |
|--:|---|--:|--:|--:|--:|--:|
| 1 | load_data | 149.76 | 88.6 | 61.2 | 387 | 17.7% |
| 2 | modify_network | 64.77 | 43.9 | 20.9 | 1204 | 8.8% |
| 3 | reference_point | 45.51 | 15.4 | 30.1 | 1096 | 3.1% |
| 4 | quick_overview | 52.39 | 25.1 | 27.3 | 1886 | 5.0% |
| 5 | correct_unwrap_error | 26.86 | 0.1 | 26.8 | 364 | 0.0% (disabled) |
| 6 | **invert_network** | **237.85** | **218.0** | 19.9 | 1436 | **43.5%** |
| 7 | correct_LOD | 29.93 | 0.9 | 29.0 | 371 | 0.2% (Sentinel: skip) |
| 8 | correct_SET | 28.05 | 2.1 | 26.0 | 371 | 0.4% |
| 9 | correct_ionosphere | 27.92 | 1.5 | 26.4 | 371 | 0.3% (no config) |
| 10 | correct_troposphere | 30.47 | 7.3 | 23.2 | 665 | 1.5% |
| 11 | deramp | 39.16 | 16.4 | 22.8 | 387 | 3.3% |
| 12 | correct_topography | 45.77 | 14.9 | 30.9 | 656 | 3.0% |
| 13 | residual_RMS | 50.70 | 25.8 | 24.9 | 449 | 5.1% |
| 14 | reference_date | 42.72 | 17.7 | 25.0 | 538 | 3.5% |
| 15 | velocity | 37.13 | 5.1 | 32.0 | 673 | 1.0% |
| 16 | geocode | 30.27 | 8.8 | 21.5 | 825 | 1.8% |
| 17 | google_earth | 30.21 | 9.1 | 21.1 | 2044 | 1.8% |
| 18 | hdfeos5 | 20.28 | 0.1 | 20.2 | 364 | 0.0% (disabled) |
| | **合計** | **991.0** | **500.8** | **489.5** | — | 100% |

### 計測上の注意

`--dostep` で 1 ステップずつ別プロセスで起動したため、Python interpreter + MintPy 全モジュールの import コスト (≈20〜30 s/step、合計 489.5 s) が `wall` に含まれています。これは通常ユーザが連続実行 (`smallbaselineApp.py FernandinaSenDT128.txt`) する場合は **1 度だけ**発生するので、本番 wall は概ね `internal 合計 + 30 s ≒ 8.8 分` と見るのが妥当です。

---

## 4. ボトルネック分析（GPU 化候補の優先順位）

純粋計算時間 (`internal`) 上位 3 ステップが全体の **70%** を占めます。

| 順位 | step | 内部時間 | 占有率 | 想定処理 (GPU 適性) |
|--:|---|--:|--:|---|
| 1 | **invert_network** | 218.0 s | 43.5% | 各ピクセルで SBAS 最小二乗反転 (大規模並列、GPU 最適) |
| 2 | **load_data** | 88.6 s | 17.7% | I/O + フォーマット変換主体 (NAS 経由、GPU 化効果は小) |
| 3 | modify_network | 43.9 s | 8.8% | 干渉ペアのフィルタリング、コヒーレンス計算 |
| 4 | residual_RMS | 25.8 s | 5.1% | 残差 RMS 集計 |
| 5 | quick_overview | 25.1 s | 5.0% | 平均コヒーレンス・速度等の俯瞰 |
| 6 | reference_date | 17.7 s | 3.5% | 時系列の基準日参照 |
| 7 | deramp | 16.4 s | 3.3% | 平面/二次フィット除去 |
| 8 | reference_point | 15.4 s | 3.1% | 参照点抽出 |
| 9 | correct_topography | 14.9 s | 3.0% | DEM 残差補正 |
| 10 | google_earth | 9.1 s | 1.8% | 出力フォーマット |
| 11 | geocode | 8.8 s | 1.8% | 地理座標化 (リサンプリング) |
| 12 | correct_troposphere | 7.3 s | 1.5% | ERA5 補正 (既キャッシュ) |
| 13 | velocity | 5.1 s | 1.0% | 線形トレンド回帰 |
| - | correct_SET | 2.1 s | 0.4% | 固体地球潮汐 |
| - | correct_ionosphere | 1.5 s | 0.3% | (無効) |
| - | correct_LOD | 0.9 s | 0.2% | (Sentinel: 不要) |
| - | correct_unwrap_error | 0.1 s | ≈0 | (無効) |
| - | hdfeos5 | 0.1 s | ≈0 | (無効) |

### GPU 化の第一候補: `invert_network`

- 単独で **全体の 43.5%** を占める最大ホットスポット
- 処理の本質: 157,667 ピクセル × (288 × 98 系の最小二乗) — 完全に独立な並列ループで、CuPy / PyTorch の **バッチ化線形代数 (`torch.linalg.lstsq`, `cp.linalg.lstsq`)** に置き換えやすい
- 実装本体: `src/mintpy/ifgram_inversion.py` (次ステップで読み込む)

### 二番手候補

- **load_data** は I/O 主体なので、まずローカル SSD にデータをコピーして再計測すべき (NAS の影響を切り分け)。GPU 化の効果は限定的。
- **modify_network / residual_RMS / quick_overview** はそれぞれ 25〜45 s 規模。invert_network 改善後に再評価し、「次の山」として GPU 化検討。
- **deramp / correct_topography** は最小二乗フィット系で GPU 化と相性良いが、規模は小さい (15 s 前後)。

---

## 5. 次のアクション提案

1. `src/mintpy/ifgram_inversion.py` を読み、最小二乗反転のループ構造とデータ I/O パターンを把握
2. CuPy / PyTorch でのバッチ反転プロトタイプを別モジュール (`mintpy/ifgram_inversion_gpu.py` など) に実装
3. テンプレートに切替フラグ (例: `mintpy.networkInversion.backend = cpu|cupy|torch`) を追加
4. 同じベンチマークスクリプト ([run_bench.sh](run_bench.sh)) を再実行し、`benchmark/logs_gpu/` に保存して比較レポート (`report_gpu.md`) を作成
