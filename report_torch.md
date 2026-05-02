# FernandinaSenDT128 GPU 化レポート (torch backend)

実施日: 2026-04-26 (JST)
対象: `mintpy.networkInversion.backend = torch` を有効化した状態での 18 ステップ計測
計測スクリプト: [benchmark/run_bench.sh](run_bench.sh)
GPU テンプレート: [FernandinaSenDT128_torch.txt](FernandinaSenDT128_torch.txt)

ログ一式:

| ラベル | ストレージ | backend | ログディレクトリ | 反転対象 pixel |
|---|---|---|---|--:|
| **NAS-CPU** (参考) | NAS (CIFS) | cpu | [logs_baseline/](logs_baseline/) | 157,667 |
| **SSD-CPU** | ローカル NVMe | cpu | [logs_cpu_local/](logs_cpu_local/) | 269,999 |
| **SSD-Torch** | ローカル NVMe | torch | [logs_torch/](logs_torch/) | 269,999 |

実装: [`src/mintpy/ifgram_inversion_gpu.py`](../src/mintpy/ifgram_inversion_gpu.py) + dispatch in [`src/mintpy/ifgram_inversion.py`](../src/mintpy/ifgram_inversion.py)
コミット: [8ab560fb](https://github.com/s-sasaki-earthsea-wizard/MintPy/commit/8ab560fb)

---

## 1. 計測条件

| 項目 | 値 |
|---|---|
| マシン | Intel Core Ultra 9 285H (16C) / 93 GiB RAM / NVIDIA RTX 5080 (16 GiB, Blackwell sm_120) |
| OS | Ubuntu 24.04.3 LTS, kernel 6.17.0-20-generic |
| Python | 3.12.3, `.venv/` (uv 管理) |
| 主要ライブラリ | torch 2.11.0+cu128, numpy 2.4.4, scipy 1.17.1, h5py 3.16.0 |
| GPU driver / CUDA | 590.48.01 / CUDA 13.1 driver, nvcc 13.0 |
| 実行 | 18 step 全部を `--dostep` で 1 ステップ 1 プロセス (cold-start) |

詳細は各 `logs_*/machine_info.txt` 参照。

---

## 2. データセット規模 — NAS と SSD で `invert_network` の workload が違う件

NAS baseline (`logs_baseline/`) と SSD bench (`logs_cpu_local/`, `logs_torch/`) では `avgSpatialCoh.h5` (品質マスクの一部) の中身が異なり、`invert_network` の反転対象 pixel 数が違う:

| バージョン | 反転対象 pixel | 比率 |
|---|--:|--:|
| NAS baseline | 157,667 | 58.4% |
| SSD CPU / Torch | 269,999 | 100.0% |

NAS baseline 取得時の `avgSpatialCoh.h5` は過去の処理結果が pre-existed で、約 11 万 pixel を zero マークしていたため有効ピクセルが少なくなっていた。一方 SSD bench は cold-start なので `quick_overview` が新規生成し、ほぼ全 pixel が有効と判定された。

> **したがって NAS-CPU と SSD-CPU の `invert_network` 時間を直接比較するのは不公平**。本レポートでは **同条件で走っている SSD-CPU と SSD-Torch を apples-to-apples の比較対象**とし、NAS-CPU は参考値として併記する。

---

## 3. ステップ別 wall 時間

`/usr/bin/time -v` 計測の壁時計時間 (Python 起動含む)。`exit=1` の step は ERA5 grib キャッシュ不在による cascade failure ([§7](#7-bench-環境上の制約)) で、bench の主軸とは無関係。

| # | step | NAS-CPU (s) | SSD-CPU (s) | SSD-Torch (s) | exit |
|--:|---|--:|--:|--:|:-:|
| 1 | load_data | 149.76 | 39.54 | 37.04 | 0 |
| 2 | modify_network | 64.77 | 33.43 | 35.63 | 0 |
| 3 | reference_point | 45.51 | 30.96 | 32.49 | 0 |
| 4 | quick_overview | 52.39 | 34.48 | 38.89 | 0 |
| 5 | correct_unwrap_error | 26.86 | 21.83 | 24.44 | 0 |
| 6 | **invert_network** | **237.85** ⚠ | **386.09** | **280.39** | 0 |
| 7 | correct_LOD | 29.93 | 18.70 | 25.35 | 0 |
| 8 | correct_SET | 28.05 | 20.38 | 22.79 | 0 |
| 9 | correct_ionosphere | 27.92 | 24.08 | 23.34 | 0 |
| 10 | correct_troposphere | 30.47 | 33.80 | 32.05 | NAS:0 / SSD:1 |
| 11 | deramp | 39.16 | 22.15 | 23.08 | NAS:0 / SSD:1 |
| 12 | correct_topography | 45.77 | 23.95 | 19.81 | NAS:0 / SSD:1 |
| 13 | residual_RMS | 50.70 | 21.65 | 17.93 | NAS:0 / SSD:0 |
| 14 | reference_date | 42.72 | 22.20 | 18.06 | 0 |
| 15 | velocity | 37.13 | 17.95 | 19.29 | NAS:0 / SSD:1 |
| 16 | geocode | 30.27 | 17.85 | 19.88 | NAS:0 / SSD:1 |
| 17 | google_earth | 30.21 | 18.57 | 20.45 | NAS:0 / SSD:1 |
| 18 | hdfeos5 | 20.28 | 19.14 | 28.85 | 0 |
| | **合計 wall** | **991** | **808** | **720** | |

⚠ NAS-CPU の `invert_network` は別 workload (157k vs 269k pixels) なので比較対象から除外。

---

## 4. `invert_network` の詳細 (apples-to-apples 比較)

`internal` は MintPy 自身が出力する `Time used:` 値。Python startup を除いた純粋処理時間。

| バージョン | pixel | wall (s) | internal (s) | per-pixel internal (ms) | speedup vs SSD-CPU |
|---|--:|--:|--:|--:|--:|
| SSD-CPU (基準) | 269,999 | 386.09 | 367.0 | 1.359 | 1.00× |
| **SSD-Torch** | 269,999 | 280.39 | **257.4** | **0.953** | **1.43×** |
| NAS-CPU (別 workload) | 157,667 | 237.85 | 218.0 | 1.382 | — |

**結果**: `invert_network` の internal で **1.43×** 高速化、wall で **1.38×** 高速化。

### GPU 経路の動作確認

[logs_torch/invert_network.log](logs_torch/invert_network.log) より:

```
estimating time-series via torch backend (batched, GPU)
GPU auto chunk_size = 19403 pixels (free VRAM 15.1 GiB)
estimating time-series via torch batched WLS in 14 chunk(s) of up to 19403 pixels ...
```

`mintpy.networkInversion.gpuChunkSize = 0` (auto) を採用、free VRAM 15.1 GiB から 19,403 pixel/chunk が選ばれ 14 chunks で完了。

---

## 5. 数値等価性

### 単体ユニットテスト ([tests/test_ifgram_inversion_gpu.py](../tests/test_ifgram_inversion_gpu.py))

合成データ (FernandinaSenDT128 と同形状の 98 dates × 288 pairs network) で:

| ケース | 期待精度 | 結果 |
|---|---|:-:|
| WLS, NaN なし | rms < 1e-5 × signal | ✅ |
| WLS, 3% NaN (冗長 network) | rms < 1e-4 × signal | ✅ |
| OLS, NaN なし | rms < 1e-5 × signal | ✅ |
| min_norm_phase | rms < 1e-5 × signal | ✅ |
| chunk size invariance | rtol 1e-6 | ✅ |
| 未対応 backend → エラー | ValueError | ✅ |

すべて float32 round-off レベルで CPU per-pixel scipy.lstsq と一致。

### 実データ smoke test (157,667 pixels, FernandinaSenDT128)

| 項目 | 値 |
|---|---|
| ts 出力中の NaN/Inf | **0 / 0** (15.4M 値中) |
| tcoh 出力中の NaN/Inf | **0 / 0** |
| tcoh の CPU との RMS 差 | **2.33e-7** (max 1.87e-5) |

実データで rank-deficient pixel は出ず、CPU と float32 限界で一致。**`gels` driver の full-rank 仮定**は今回の dataset では問題なし。

---

## 6. なぜ 1.4× にとどまったか

`Time used: 257.4 s` (SSD-Torch internal) は単純な lstsq 時間ではなく、`run_ifgram_inversion_patch` 全体を含む:

```
read_stack_obs + ref_phase 引き当て + mask 構築   ┐
                                                  ├─ ここは CPU の numpy / h5py のまま
calc_weight_sqrt (coherence → weight 変換, chunked)┘

estimate_timeseries_batch (本コミットで GPU 化)   ── 主要 hot loop

phase → meter 単位変換 + reshape                  ── ほぼ 0 秒
```

`estimate_timeseries_batch` 内では:
- 14 chunks × 各 chunk で host→device コピー (`y`, `weight_sqrt`, `valid` mask)
- `torch.linalg.lstsq` (CUDA `gels` driver, batched)
- tcoh 計算 + cumsum (GPU)
- device→host コピー (`ts`, `tcoh`, `nobs`)

`gels` driver は QR factorization で full-rank 前提の高速 path だが、batched でも cuSOLVER 上では `(num_pair, num_unknown) = (288, 97)` の行列を 19403 個並列に解く形なので、純粋な GEMM ほど効率は出ない (workspace allocation overhead, NaN-mask 処理用 weight 0 行の wasted FLOPs)。

**つまり「lstsq ループ自体は GPU 化されたが、その前後の CPU 処理 (read, weight) と GPU 計算内の overhead が無視できないサイズ」になっており、1.4× が現状の上限**。プロファイリングで分解すれば内訳が出る (§8 follow-up)。

---

## 7. bench 環境上の制約

`SSD-CPU` / `SSD-Torch` 両方で `correct_troposphere` 以降の一部 step が `exit=1` (cascade failure) となった。原因は **PyAPS が必要とする ERA5 grib キャッシュ**が SSD copy には存在せず、`number of grib files used: 0` で `progressBar` が ZeroDivisionError を投げたため (`src/mintpy/objects/progress.py:109`)。NAS baseline では何らかのキャッシュ経路で grib が解決していた。

bench の主軸 (`invert_network`) には影響なし。GPU bench/CPU bench で同じ step が同じ理由で落ちており、比較性は保たれている。

---

## 8. 次のアクション (follow-ups)

| # | タスク | 期待効果 |
|--:|---|---|
| 1 | `invert_network` を py-spy / cProfile で profile し、read / weight / lstsq / コピー の内訳を出す | speedup の上限を数値で説明 |
| 2 | `calc_weight_sqrt` の GPU 化 (chunked numpy → torch) | weight 計算 (~25 s/run 推定) の削減 |
| 3 | `--backend cupy` 実装 + bench 比較 | CuPy vs PyTorch の特性比較 |
| 4 | rank-deficient pixel に対する CPU fallback | 別 dataset で必要になった場合の対策 |
| 5 | ERA5 grib キャッシュの環境整備 | bench full-pipeline 化 |
| 6 | upstream PR 化 (Wiki [Upstream-PR-Checklist](https://github.com/s-sasaki-earthsea-wizard/MintPy/wiki/Upstream-PR-Checklist) 準拠) | insarlab/MintPy への還元 |

---

## 9. 結論

- `mintpy.networkInversion.backend = torch` は実データで **NaN/Inf を発生させず、CPU と float32 round-off レベルで数値一致**
- `invert_network` の internal time で **1.43× / wall で 1.38×** の高速化
- 純粋 lstsq 部分はほぼ GPU 化済み。残る改善余地は **計算前の weight 構築 (CPU bound)** と **chunk 切り替えの host↔device コピー**
- defaults は upstream 互換 (`mintpy.networkInversion.backend = cpu`)、`--backend torch` または template flag で opt-in
