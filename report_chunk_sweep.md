# `invert_network` chunk_size sweep (torch backend)

実施日: 2026-04-26 (JST)
対象: `mintpy.networkInversion.gpuChunkSize` の sweet spot 検証 + GPU lstsq 固有時間 vs 周辺 overhead の比率の事前分離 ([Issue #2](https://github.com/s-sasaki-earthsea-wizard/MintPy/issues/2) の follow-up #1 の事前ステップ)

計測スクリプト: [benchmark/run_chunk_sweep.sh](run_chunk_sweep.sh)
ログ一式: [logs_chunk_sweep/](logs_chunk_sweep/)

---

## 1. 目的

[report_torch.md](report_torch.md) の SSD-Torch ベンチで `invert_network` の internal time が **257.4 s (1.43× speedup vs CPU)** に留まった。期待ほど大きくない速度向上の上限要因として、prior session ([2026-04-26 ノート](../.claude-notes/2026-04-26_invert-network-torch.md) L101-105) で以下 3 つの仮説を立てた:

| 仮説 | 期待される現象 |
|---|---|
| (a) 小 chunk → kernel launch + host↔device コピーが支配 | 小 chunk で wall が大きく増加 |
| (b) 中 chunk → cuSOLVER workspace + メモリ局所性で sweet spot | auto (19,403) 付近が最速 |
| (c) 大 chunk → cache miss / memory bandwidth bound | VRAM 限界手前で性能劣化 |

本 sweep は **profiler 導入前** に、chunk_size を変えるだけで仮説 (a)〜(c) のどれが効いているかを切り分け、profiler で見るべき箇所 (kernel? memcpy? Python overhead?) の事前仮説を立てるのが狙い。

---

## 2. 計測条件

| 項目 | 値 |
|---|---|
| マシン | Intel Core Ultra 9 285H (16C) / 93 GiB RAM / NVIDIA RTX 5080 (16 GiB, Blackwell sm_120) |
| ストレージ | ローカル NVMe SSD (`~/MintPy_bench/...`) |
| Python / torch | 3.12.3 / 2.11.0+cu128 |
| dataset | FernandinaSenDT128 (288 ifgrams × 98 acquisitions, 269,999 反転対象 pixel) |
| バックエンド | `mintpy.networkInversion.backend = torch` |
| 単 step | `--dostep invert_network` のみ (full pipeline は不要; phase 1 で end-to-end は確認済) |
| 測定 | `/usr/bin/time -v` で wall + max RSS、ログから `Time used:` (internal) と resolved `chunk_size` / `num_chunks` を抽出 |
| 環境隔離 | sweep 中は他プロセス (CPU heavy Docker container) を一時停止 |

### Sweep matrix

| `gpuChunkSize` | 実 chunk_size | 実 num_chunks | 想定する regime |
|--:|--:|--:|---|
| 1000 | 1,000 | 270 | very-small (overhead 支配を期待) |
| 5000 | 5,000 | 54 | small |
| 10000 | 10,000 | 27 | medium |
| 0 (auto) | **19,403** | **14** | 自動算出 (`0.4 × free_VRAM / per_pixel_bytes`) |
| 30000 | 30,000 | 9 | large |
| 40000 | 40,000 | 7 | near-VRAM (`80,000` は workspace OOM するため除外) |

各 chunk_size を **2 round の round-robin** (Round1: 1k→5k→10k→auto→30k→40k、Round2 同順) で計 12 run。`--dostep invert_network` は output 既存時に update mode で skip するため、各 run の前に `timeseries.h5` / `temporalCoherence.h5` / `numInvIfgram.h5` を削除して強制再実行。

> **methodology bug 注記**: 初回 sweep は (1) update-mode skip と (2) 前回 run の `gpuChunkSize` が work_dir の `smallbaselineApp.cfg` に焼き付いて template merge で残る、という 2 点で全 run が無効化されていた。最終版スクリプトは output 削除と `gpuChunkSize` の毎回明示書き出しで両方解消。

---

## 3. 結果

### 3.1 集計

| `gpuChunkSize` | num_chunks | wall mean (s) | wall min (s) | internal mean (s) | internal min (s) | per-pixel (ms) | max RSS (GiB) |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 1,000 | 270 | 271.81 | 266.60 | 246.95 | 243.70 | 0.915 | 2.50 |
| 5,000 | 54 | 273.64 | 271.29 | 246.20 | 244.00 | 0.912 | 2.52 |
| 10,000 | 27 | 268.73 | 263.41 | 245.30 | 242.60 | 0.909 | 2.53 |
| **19,403 (auto)** | **14** | **266.86** | **265.40** | **242.95** | **242.80** | **0.900** | **2.57** |
| 30,000 | 9 | 277.49 | 276.05 | 250.75 | 248.60 | 0.929 | 2.61 |
| 40,000 | 7 | 268.23 | 263.35 | 244.80 | 242.10 | 0.907 | 2.63 |

per-pixel は internal mean / 269,999 pixel × 1000。生データは [logs_chunk_sweep/summary.tsv](logs_chunk_sweep/summary.tsv)。

### 3.2 round 別生データ

| `gpuChunkSize` | r1 wall | r2 wall | r1 internal | r2 internal |
|--:|--:|--:|--:|--:|
| 1,000   | 277.02 | 266.60 | 250.2 | 243.7 |
| 5,000   | 271.29 | 275.99 | 244.0 | 248.4 |
| 10,000  | 274.05 | 263.41 | 248.0 | 242.6 |
| 19,403  | 268.31 | 265.40 | 243.1 | 242.8 |
| 30,000  | 276.05 | 278.93 | 248.6 | 252.9 |
| 40,000  | 263.35 | 273.10 | 242.1 | 247.5 |

Round-to-round 揺らぎ: 概ね ±2〜5 秒 (機械学習系 single-shot bench としては標準的)。

---

## 4. 解析

### 4.1 sweep 曲線はほぼフラット

internal time は **全レンジで 242〜251 秒に収まる (幅 9 秒, ±1.8%)**。Round-to-round 揺らぎ (~5s) を考えると、**chunk_size を変えても性能はほぼ動かない**というのが第一の所見。

### 4.2 仮説 (a)〜(c) の検証

| 仮説 | 期待 | 実測 | 結論 |
|---|---|---|---|
| (a) 小 chunk overhead | cs=1k で大幅劣化 | cs=1k は auto より +4 秒 (+1.6%) のみ | **棄却**: kernel launch + memcpy overhead は無視できる |
| (b) auto sweet spot | auto (19,403) が最速 | 確かに最速 (242.95s) だが他との差は ≤3% | **弱く支持**: sweet spot は存在するが ROI は小さい |
| (c) 大 chunk degradation | cs=40k で劣化 | cs=40k = 244.8s (auto と同等)、cs=30k = 250.75s (+3.2%) | **観測されず**: VRAM 限界で degradation は出ていない |

特に (a) の棄却が重要。270 chunks (cs=1k) でも wall がほぼ変わらないことから、**chunk per-launch overhead を見積もれる**:

$$
\text{per-chunk overhead} \approx \frac{246.95 - 242.95}{270 - 14} = \frac{4.0\,\text{s}}{256\,\text{chunks}} \approx 16\,\text{ms/chunk}
$$

auto (14 chunks) における chunk launch の総コストは `14 × 16ms = 0.22 秒`、internal time の **0.09%** に過ぎない。

### 4.3 wall − internal は constant

| `gpuChunkSize` | wall − internal (s) |
|--:|--:|
| 1k | 24.86 |
| 5k | 27.44 |
| 10k | 23.43 |
| 19k | 23.91 |
| 30k | 26.74 |
| 40k | 23.43 |

差は **24〜27 秒で chunk_size に依存せず一定**。これは `smallbaselineApp.py --dostep invert_network` の周辺コスト (Python 起動 + template/cfg 解釈 + post-step `generate_mask.py` ~few sec + I/O finalization)。本 sweep の対象外。

### 4.4 cs=30,000 がやや遅い件

唯一 +6〜8 秒の劣化が見えるが、9 chunks のうち最後の 1 chunk が 29,999 pixel (full) で workload は均等に近い。原因仮説は cuSOLVER workspace の境界アライメント or VRAM allocator の挙動だが、**1 ポイントだけの劣化なので統計的に強い主張はできない** (round 内差 ±2.2s、隣接 cs=40k は +1.85s で安定範囲)。**profiler でも特に追わない**。

---

## 5. 結論

1. **chunk_size の sweet spot は auto (19,403) で確かに最速。ただし他の chunk_size との差は ≤3%、現実的には flat**
2. **chunk launch + host↔device memcpy overhead は ~16 ms/chunk で、auto では internal time の 0.1% 未満**
3. **したがって invert_network の高速化に「chunk_size をいじる」アプローチは不適**。internal time 243 秒の正体は別のところにある
4. 現状の `_auto_chunk_size` 関数は VRAM ベースで妥当な値を出している (チューニング不要)

---

## 6. 次ステップへの含意

本 sweep は profiler 導入前の null result として、次の profiling 計画にとって有意義な制約を与えた:

- chunk launch overhead は無視できる → **`torch.profiler` のタイムラインで kernel launch カウントを追う必要なし**
- 大 chunk で性能が伸びない (cs=40k > cs=30k なのに auto と同等) → cuSOLVER の `gels` driver は中サイズバッチで既に飽和している → **GPU 計算自体は支配項ではない可能性が高い**
- 24〜27 秒の固定 overhead は smallbaselineApp 側 → **invert_network 内部 (~243s) を分解する場合、smallbaselineApp wrapper レベルではなく `run_ifgram_inversion_patch` 以下で取る**

[Issue #2](https://github.com/s-sasaki-earthsea-wizard/MintPy/issues/2) follow-up #1 (profiling) で見るべき主要候補:

| 候補 | 期待される寄与 |
|---|---|
| **`read_stack_obs`** (h5py read 288 × 269,999 pixels) | 数十秒、I/O bound |
| **`calc_weight_sqrt`** (CPU, coherence → weight、3 chunk × 100k pixels) | 数十秒、CPU bound |
| `estimate_timeseries_batch` 内 GPU lstsq | 残り |
| host-device コピー (`y_dev`, `w_dev` per chunk) | 既に小と判明 |

→ **py-spy** で Python 関数の hot path top-down + **torch.profiler** で `read_stack_obs` / `calc_weight_sqrt` / GPU 領域の per-region 内訳 を組み合わせることで `read / weight / lstsq / memcpy` の比率を確定する。

---

## 7. 再現

```bash
# 前提: ~/MintPy_bench/FernandinaSenDT128/mintpy/ に dataset を SSD コピー済
WORK_DIR=$HOME/MintPy_bench/FernandinaSenDT128/mintpy \
    bash benchmark/run_chunk_sweep.sh
```

12 run × ~270 秒 ≈ **約 55 分**。CPU heavy な他プロセス (Docker など) は事前に停止する (本 sweep の `Time used:` は CPU contention に敏感)。
