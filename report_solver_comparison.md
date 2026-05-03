# `_SOLVER='cholesky'` vs `_SOLVER='lstsq'` 比較レポート

実施日: 2026-05-03 (JST)
対象: torch backend (`mintpy.networkInversion.backend = torch`) の chunk-内 solver、
[`mintpy.ifgram_inversion_gpu._SOLVER`](https://github.com/s-sasaki-earthsea-wizard/MintPy/blob/main/src/mintpy/ifgram_inversion_gpu.py#L50) 定数の二択比較
データセット: FernandinaSenDT128 (98 dates × 288 ifgrams、269,999 px に反転)

> **本レポートのスコープ**: 両 solver を完走させ、wall time / 数値同等性 / kernel 構造を並べる。
> **`cholesky` の採用判断・`lstsq` 経路の削除判断はスコープ外** ([Issue #4](https://github.com/s-sasaki-earthsea-wizard/MintPy/issues/4) 別タスク)。

---

## TL;DR

| 観点 | 結果 |
|---|---|
| **Wall time** (3-shot mean、`--dostep invert_network`) | **cholesky 61.34 s / lstsq 275.07 s ≈ 4.5×** |
| **Internal time** (`Time used:`、Python startup 除外) | **cholesky 13.83 s / lstsq 228.17 s ≈ 16.5×** |
| **数値一致** (per-pixel RMS、269,999 px、ts_lstsq の per-pixel std で正規化) | normalised max **1.19e-5**、p99 1.89e-6、median 5.41e-7 — float32 round-off レベル |
| **GPU kernel 起動回数** (1 chunk あたり) | **cholesky 57 vs lstsq 3,841,835 ≈ 67,400×** |
| **GPU kernel 時間** (1 chunk あたり) | **cholesky 24.2 ms vs lstsq 10.41 s ≈ 430×** |
| **NaN/Inf** | 両 solver とも 0/0 (270k pixel × 98 epoch = 26.46M 値中) |

`gels` driver の per-pixel iterative QR (~1.88 M kernel launches per chunk) が cholesky の batched normal equation + Cholesky (~5 launches per chunk) で完全に置き換わっている、という [report_profile.md](report_profile.md) の予測通りの結果。

---

## 1. 計測条件

| 項目 | 値 |
|---|---|
| マシン | Intel Core Ultra 9 285H (16C) / 93 GiB RAM / NVIDIA RTX 5080 (16 GiB, Blackwell sm_120) |
| OS / kernel | Ubuntu 24.04, kernel 6.17.0-22-generic |
| Python | 3.12.3, `.venv/` (uv 管理) |
| 主要ライブラリ | torch 2.11.0+cu128, numpy 2.4.4, h5py 3.16.0 |
| GPU driver / CUDA | CUDA 13.1 driver, nvcc 13.0 |
| 対象 commit | `eedfaf17` (PR #9 merged, `_SOLVER='cholesky'` がデフォルト) |
| Work dir | `~/MintPy_bench/FernandinaSenDT128/mintpy/` (ローカル NVMe SSD) |
| Template | [`FernandinaSenDT128_torch.txt`](FernandinaSenDT128_torch.txt) + `mintpy.networkInversion.gpuChunkSize = 0` (auto) |
| Solver 切替 | env var `MINTPY_SOLVER` から `_SOLVER` を runtime override (本 sibling repo の wrapper、MintPy 本体無変更) |
| 反転対象 pixel | 269,999 / 270,000 (avgSpatialCoh threshold 0.7、cold-start) |

ハーネス:
- bench: [`run_solver_comparison.sh`](run_solver_comparison.sh) (3-shot/solver、pre-clean h5、`/usr/bin/time -v`)
- wrapper: [`run_smallbaseline_with_solver.py`](run_smallbaseline_with_solver.py)
- profile: [`profile_torch_solver.py`](profile_torch_solver.py) (solver-aware monkey-patch)
- RMS: [`compare_solutions.py`](compare_solutions.py) (per-pixel std で正規化)

---

## 2. Wall time / Internal time (3-shot 計測)

`/usr/bin/time -v` 計測の壁時計時間と MintPy 自身の `Time used:` 出力 (Python startup を除いた純粋処理時間)。

| solver | shot 1 wall | shot 2 wall | shot 3 wall | mean wall | shot 1 internal | shot 2 internal | shot 3 internal | mean internal |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| cholesky | 60.75 | 58.92 | 64.35 | **61.34 s** | 14.0 | 13.7 | 13.8 | **13.83 s** |
| lstsq | 269.63 | 278.81 | 276.78 | **275.07 s** | 228.0 | 228.1 | 228.4 | **228.17 s** |

| 比較対象 | 比 |
|---|--:|
| Wall mean (cholesky / lstsq) | 0.223 → **4.49× 高速** |
| Internal mean (cholesky / lstsq) | 0.0606 → **16.50× 高速** |

raw artifacts: [bench/summary.tsv](logs_solver_run1/bench/summary.tsv)

### Wall と Internal の差について

cholesky の wall=61 s に対して internal=14 s。差 ~47 s の大半は **Python interpreter 起動 + torch / mintpy import + CUDA lazy init** で、`--dostep invert_network` 単発実行ごとに発生する一定コスト。lstsq では internal=228 s なのでこの 47 s は相対的に消える。

つまり「実用上の高速化」は **internal time 16.5× が最もフェアな比較**。`run_smallbaselineApp` 全体パイプライン (`load_data` 〜 `hdfeos5`) のように Python 起動が 1 回で済むケースでは internal 比に近づく。

---

## 3. 数値同等性 (RMS 比較)

両 solver で得られた `timeseries.h5` の per-pixel RMS を計算し、`ts_lstsq` の per-pixel 時系列 std で正規化したものを solver 間の許容差として報告。`std(ts_lstsq, axis=0) < 1e-6 m` の pixel (1/270,000) は「signal がほぼゼロで分母が小さい」ため正規化統計から除外。

| 統計 | 絶対 RMS (m) | 正規化 RMS = `RMS(ts_chol − ts_lstsq) / std(ts_lstsq)` |
|---|--:|--:|
| count | 270,000 | 269,999 |
| min | 0.0 | 4.63e-8 |
| median | 5.48e-9 | 5.41e-7 |
| p99 | 2.16e-8 | 1.89e-6 |
| **max** | **1.59e-7 m (約 0.16 µm)** | **1.19e-5** |
| mean | 6.59e-9 | 6.39e-7 |

参考: `ts_lstsq` の per-pixel 振幅 (signal std) は median 9.18 mm、p99 42.8 mm、max 113 mm。

raw artifacts: [compare/summary.txt](logs_solver_run1/compare/summary.txt) / [compare/rms_cholesky_vs_lstsq.json](logs_solver_run1/compare/rms_cholesky_vs_lstsq.json)

### 数値同等性の解釈

- **絶対 RMS の最大は 0.16 µm** = 1.59e-7 m。実データ振幅の最大 113 mm に対して 7 桁下。
- **正規化 RMS の最大は 1.19e-5**。同じレベルが [Issue #4](https://github.com/s-sasaki-earthsea-wizard/MintPy/issues/4) acceptance criteria の参考値 (`1e-4 × signal scale`) に対して 1/8 程度。
- float32 の machine epsilon は 1.2e-7。SBAS 設計行列の cond は実測 ≤ 10³、normal-equation 化で `cond(G^T G) ≤ 10⁶` と理論上 0.1 弱の relative error が出てもおかしくないが、実データでは **dominant な signal の上に round-off 級の差しか残らない**。
- NaN/Inf は cholesky / lstsq とも 0 個 (26.46M 値中)。rank-deficient pixel も両方とも 0 件。

> **どこまでが意味のある差か** はこのレポートでは判定しない。Issue #4 の acceptance criteria (RMS < 1e-4 × signal scale) と照合した数値は上記の通り。

---

## 4. GPU kernel breakdown (1 chunk あたり)

両 solver を `torch.profiler` で `schedule(wait=1, active=1)` の 1 chunk 分プロファイル。Per-chunk step boundary は dispatch 関数 (`_solve_cholesky` / `_solve_lstsq`) を monkey-patch し、`prof.step()` を 1 chunk 1 回挟む形 ([profile_torch_solver.py](profile_torch_solver.py))。

raw artifacts:
- cholesky: [profile/cholesky/parsed.md](logs_solver_run1/profile/cholesky/parsed.md) / [key_averages.txt](logs_solver_run1/profile/cholesky/tb_trace/) / [trace.json](logs_solver_run1/profile/cholesky/tb_trace/) (139 KB)
- lstsq: [profile/lstsq/parsed.md](logs_solver_run1/profile/lstsq/parsed.md) / [trace.json](logs_solver_run1/profile/lstsq/tb_trace/) (3.90 GiB、`key_averages` は OOM で `parse_trace.py` 経由で reduce、[report_profile.md](report_profile.md) 同手順)

### 4.1 Top-level (per chunk)

| 指標 | cholesky | lstsq | 比 (lstsq / cholesky) |
|---|--:|--:|--:|
| `ProfilerStep#1` (= 1 chunk wall) | 1.161 s | 19.869 s | **17.1×** |
| Total `kernel` time | 24.20 ms | 10.409 s | **430×** |
| Total `cuda_runtime` (host launch) time | 1.097 s | 18.210 s | **16.6×** |
| Number of `kernel` events | 57 | 3,841,835 | **67,400×** |
| Number of `cuda_runtime` events | 85 | 3,861,259 | **45,400×** |
| Number of `gpu_memcpy` events | 8 | 7 | ≈同 |
| Memcpy HtoD (合計) | 7.73 ms | 7.75 ms | ≈同 |

cholesky の kernel time (24 ms) と launch overhead (1.1 s) は逆転 (launch >> compute) しているが、**absolute time が極小**なので overhead 自体が無視できる。lstsq は **launch overhead 18.2 s + kernel time 10.4 s** が並走で重なって 19.9 s wall という、典型的な launch-bound 状態。

> **重要**: cholesky の `cuda_runtime=1.097 s` は ProfilerStep の overhead を含むので、純粋な launch 時間ではない (`overhead=1.060 s` がほぼ占める)。実際の launch overhead は 24 ms 級と読むべき。

### 4.2 Top GPU kernels — cholesky (1 chunk)

| Rank | Total | Calls | Kernel | 役割 |
|--:|--:|--:|---|---|
| 1 | 7.18 ms | 1 | `cutlass_80_simt_sgemm_128x32_8x5_nt_align1` | `N = G_w^T G_w` (batched GEMM) |
| 2 | 2.52 ms | 5 | `potrf_syrk_T16_nc_kernel<float, 5, 4, 4, 5, 4>` | Cholesky factor (panel syrk) |
| 3 | 2.51 ms | 3 | elementwise `Mul` | 重み broadcasting |
| 4 | 2.44 ms | 1 | `gemv2N_kernel<...>` | `G_w^T y` (batched GEMV) |
| 5 | 1.69 ms | 1 | `triu_tril_kernel` | 三角化 |
| 6 | 1.22 ms | 1 | `strsv_trans_kernel_outplace_batched` | back-solve (L^T) |
| 7 | 1.17 ms | 6 | `potrfBatch_trsm_lower<float, float, 16>` | Cholesky panel trsm |
| 8 | 1.12 ms | 7 | `potrf_cta_lower_batch<float, float, 16>` | Cholesky CTA-level |
| 9 | 1.07 ms | 1 | `strsv_notrans_kernel_outplace_batched` | back-solve (L) |
| 10 | 906 µs | 1 | `potrf_syrk_nc_kernel` | Cholesky panel syrk (closing) |

→ ほぼ「**1 GEMM + 1 batched-Cholesky factor + 2 batched-trsm + 1 batched-GEMV**」で完結。1 chunk = 1 batch の GPU work。

### 4.3 Top GPU kernels — lstsq (1 chunk)

| Rank | Total | Calls | Kernel | 役割 |
|--:|--:|--:|---|---|
| 2 | 3.995 s | **1,882,091** | `ormtr_gemv_c<float, 4>` | gels back-substitution (per-iteration support) |
| 3 | 3.669 s | **19,403** | `geqr2_smem_domino_fast<float, float, 8, 512>` | gels QR factorization |
| 4 | 2.691 s | **1,882,091** | `ormtr_gerc<float, 5, 3, 1>` | gels back-substitution (per-iteration apply) |
| 5 | 42.5 ms | 58,209 | `copy_info_kernel` | per-pixel info bookkeeping |
| 6 | 21.2 ms | 19,403 | `Memset (Device)` | per-pixel 初期化 |
| 11 | 974 µs | 2 | `batch_trsm_left_kernel` | back-solve |

per-pixel 構造:
- `1,882,091 / 19,403 = 97 = num_unknown` (= num_date − 1)
- gels driver は **「pixel ごとに独立な iterative QR」** を実行: 各 pixel に Householder reflection を 97 回、各 reflection に 2 つの support kernel
- `19,403 + 1,882,091 + 1,882,091 + 58,209 ≈ 3.84 M kernel launches`

cholesky の 57 launches と対比: **batched solver vs per-pixel iterative solver** の構造的な違いが、kernel launch overhead で 18 s vs 数十 ms (実質) の差として現れる。

---

## 5. なぜ cholesky が速いのか (kernel-level の解釈)

cholesky path の chunk-内 GPU work は概念的に:

```
N = (G * w[:, None])^T @ (G * w[:, None])     # 1 batched GEMM, (n, num_unknown, num_unknown)
r = (G * w[:, None])^T @ (y * w[:, None])     # 1 batched GEMV, (n, num_unknown)
L = cholesky(N)                                # 1 batched cholesky_ex (potrf)
X = cholesky_solve(r, L)                       # 1 batched 2x triangular solve
```

これら全部が cuSolver / cuBLAS の **batched API** で 1 launch に収まる (各 launch 内で n 個の pixel 並列)。実測では panel/CTA 分割で多少の launch 増 (potrf 系で 18 launches 程度) があるが、合計でも **57 launches / chunk**。

一方 lstsq path:
```
torch.linalg.lstsq(A_batch, b_batch)  # cuSolver gels driver (QR)
```

cuSolver `gels` は batched API ではあるが、内部で **pixel ごとに独立した iterative Householder QR** を実行。1 reflection に 2 サブカーネル × 97 reflections × 19,403 pixels = **3.86 M launches / chunk**。

> 1 launch ≈ 5 µs の host-side `cudaLaunchKernel` overhead (CUDA runtime 標準値)。3.86M × 5 µs = **18 s** が overhead。実測 `cuda_runtime=18.21 s` と一致。

これは [report_profile.md](report_profile.md) の Phase 2 予測と一致:

| | gels (lstsq) | normal-eq + Cholesky |
|---|---|---|
| Per-chunk launches | ~1.88 M | ~5 |
| Per-chunk GPU compute | 10.4 s | ~6 s (理論) |
| Per-chunk launch overhead | ~18 s | ≪ 1 s |
| Predicted chunk wall | 19.9 s | ≈ 6 s |

実測 (本レポート §2): cholesky internal/wall は 14 chunks 合算で 13.83 s / 61.34 s。1 chunk 平均 ≈ 1 s wall (ProfilerStep#1 の 1.161 s と整合)。**予測 6 s/chunk より更に速い** — `cuSolver/cuBLAS` の batched 実装が想定以上に効率的。

---

## 6. 計測上の制約 / メモ

### 6.1 lstsq profile での `key_averages()` OOM

lstsq path の `prof.key_averages().table()` は kineto event を Python オブジェクトとして全数 materialise しようとするため、本 dataset サイズ (active=1 でも 15.94 M events) で `std::bad_alloc` クラスの host RSS 増 (本実行では 45 GiB 到達後 SIGINT で停止) を起こす。これは [report_profile.md](report_profile.md) で 2026-05-03 incident として記録されている既知挙動 (`key_averages.txt` は best-effort、`tb_trace/*.json` が authoritative)。

本レポートでは [`parse_trace.py`](parse_trace.py) で trace JSON をオフライン reduction し kernel breakdown を取得。`key_averages()` が「途中で interrupt されている」のは異常ではなく、設計上の挙動 (cholesky 側は events が 561 個のみで一瞬で完了)。

### 6.2 Python 起動オーバーヘッド

cholesky の wall=61 s に対して internal=14 s。差 47 s = (Python interpreter init + torch import + mintpy import + CUDA lazy init) の固定コスト。これは `--dostep invert_network` 単独実行の構造によるもので、`smallbaselineApp.py` を 18 step 全部 1 プロセスで回す通常のパイプラインでは 1 回しか発生しない。

### 6.3 SSD vs NAS

[`feedback_bench_io_isolation.md`](https://github.com/s-sasaki-earthsea-wizard/MintPy/issues) に従い wall time は SSD で計測。RMS 検証は I/O 非依存なので SSD で兼ねる。

### 6.4 update-mode skip 罠

各 shot 開始前に `timeseries.h5 / temporalCoherence.h5 / numInvIfgram.h5` を `rm -f` ([`run_solver_comparison.sh`](run_solver_comparison.sh) の `run_one_shot`)。invert_network は MintPy の update-mode skip key list に入っており、leftover h5 が残ると silent に no-op される。

---

## 7. 結論 (本レポートの観察事実、判断はスコープ外)

1. `_SOLVER='cholesky'` は `_SOLVER='lstsq'` に対し、本 dataset で **internal time 16.5×、wall time 4.5×** 高速。
2. 数値同等性は **正規化 max RMS 1.19e-5** = float32 round-off レベル。NaN/Inf 0 件、rank-deficient 0 件。
3. 高速化の機構は **kernel launch 構造の違い** (batched GEMM/Cholesky 5 launches vs gels per-pixel iterative QR 3.86 M launches per chunk)。compute 自体の差より launch overhead 削減が dominant。
4. メモリ traffic (HtoD/DtoH) は両 solver で同等 (HtoD 7.7 ms / chunk)、よってこれらの最適化は当該 dataset では未踏。

**スコープ外**: 本結果から `_solve_lstsq` 経路を残すべきか、`_SOLVER` を template flag に昇格するか、cholesky を upstream PR 化するか — これらの判断は [Issue #4](https://github.com/s-sasaki-earthsea-wizard/MintPy/issues/4) で別タスクとして扱う。
