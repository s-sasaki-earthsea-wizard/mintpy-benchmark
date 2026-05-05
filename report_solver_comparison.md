# `_SOLVER='cholesky'` vs `_SOLVER='lstsq'` comparison report

Date: 2026-05-03 (JST)
Subject: Two-way comparison of the chunk-internal solver in the torch backend (`mintpy.networkInversion.backend = torch`), via the [`mintpy.ifgram_inversion_gpu._SOLVER`](https://github.com/s-sasaki-earthsea-wizard/MintPy/blob/eedfaf17/src/mintpy/ifgram_inversion_gpu.py#L50) constant
(the `_SOLVER` constant has been removed by a cleanup PR landed after this report — referenced via permalink).
Dataset: FernandinaSenDT128 (98 dates × 288 ifgrams, inverted on 269,999 px)

> **Scope of this report**: run both solvers to completion and present wall time / numerical equivalence / kernel structure side-by-side.
> **Decisions about adopting `cholesky` or removing the `lstsq` path are out of scope** (handled in [Issue #4](https://github.com/s-sasaki-earthsea-wizard/MintPy/issues/4) as separate tasks. After this report was completed, the same issue's removal decision dropped `_SOLVER` / `_solve_lstsq` and the three comparison harness scripts in this sibling repo (`run_solver_comparison.sh` / `run_smallbaseline_with_solver.py` / `profile_torch_solver.py`) in a cleanup PR. All in-text references therefore use permalinks pinned to the comparison-era commit `0682c4c`).

---

## TL;DR

| Aspect | Result |
|---|---|
| **Wall time** (3-shot mean, `--dostep invert_network`) | **cholesky 61.34 s / lstsq 275.07 s ≈ 4.5×** |
| **Internal time** (`Time used:`, Python startup excluded) | **cholesky 13.83 s / lstsq 228.17 s ≈ 16.5×** |
| **Numerical equivalence** (per-pixel RMS over 269,999 px, normalised by `ts_lstsq` per-pixel std) | normalised max **1.19e-5**, p99 1.89e-6, median 5.41e-7 — at float32 round-off level |
| **GPU kernel launches** (per chunk) | **cholesky 57 vs lstsq 3,841,835 ≈ 67,400×** |
| **GPU kernel time** (per chunk) | **cholesky 24.2 ms vs lstsq 10.41 s ≈ 430×** |
| **NaN/Inf** | 0/0 in both solvers (out of 270k pixels × 98 epochs = 26.46M values) |

The result lines up with the prediction in [report_profile.md](report_profile.md): the per-pixel iterative QR of the `gels` driver (~1.88 M kernel launches per chunk) is wholly replaced by cholesky's batched normal equation + Cholesky (~5 launches per chunk).

---

## 1. Measurement conditions

| Item | Value |
|---|---|
| Machine | Intel Core Ultra 9 285H (16C) / 93 GiB RAM / NVIDIA RTX 5080 (16 GiB, Blackwell sm_120) |
| OS / kernel | Ubuntu 24.04, kernel 6.17.0-22-generic |
| Python | 3.12.3, `.venv/` (managed by uv) |
| Key libraries | torch 2.11.0+cu128, numpy 2.4.4, h5py 3.16.0 |
| GPU driver / CUDA | CUDA 13.1 driver, nvcc 13.0 |
| Target commit | `eedfaf17` (PR #9 merged, `_SOLVER='cholesky'` is the default) |
| Work dir | `~/MintPy_bench/FernandinaSenDT128/mintpy/` (local NVMe SSD) |
| Template | [`FernandinaSenDT128_torch.txt`](FernandinaSenDT128_torch.txt) + `mintpy.networkInversion.gpuChunkSize = 0` (auto) |
| Solver switch | `_SOLVER` runtime override via env var `MINTPY_SOLVER` (this sibling repo's wrapper; MintPy itself untouched) |
| Inverted pixels | 269,999 / 270,000 (avgSpatialCoh threshold 0.7, cold start) |

Harness (after the lstsq path was removed, retained for this report only as historical permalinks):
- bench: [`run_solver_comparison.sh`](https://github.com/s-sasaki-earthsea-wizard/mintpy-benchmark/blob/0682c4c/run_solver_comparison.sh) (3 shots/solver, pre-clean of h5, `/usr/bin/time -v`)
- wrapper: [`run_smallbaseline_with_solver.py`](https://github.com/s-sasaki-earthsea-wizard/mintpy-benchmark/blob/0682c4c/run_smallbaseline_with_solver.py)
- profile: [`profile_torch_solver.py`](https://github.com/s-sasaki-earthsea-wizard/mintpy-benchmark/blob/0682c4c/profile_torch_solver.py) (solver-aware monkey-patch)
- RMS: [`compare_solutions.py`](compare_solutions.py) (normalised by per-pixel std; kept as a generic h5 comparison tool)

---

## 2. Wall time / internal time (3-shot measurement)

Wall clock from `/usr/bin/time -v` and the `Time used:` value MintPy itself reports (pure processing time with Python startup excluded).

| solver | shot 1 wall | shot 2 wall | shot 3 wall | mean wall | shot 1 internal | shot 2 internal | shot 3 internal | mean internal |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| cholesky | 60.75 | 58.92 | 64.35 | **61.34 s** | 14.0 | 13.7 | 13.8 | **13.83 s** |
| lstsq | 269.63 | 278.81 | 276.78 | **275.07 s** | 228.0 | 228.1 | 228.4 | **228.17 s** |

| Comparison | Ratio |
|---|--:|
| Wall mean (cholesky / lstsq) | 0.223 → **4.49× faster** |
| Internal mean (cholesky / lstsq) | 0.0606 → **16.50× faster** |

Raw artifacts: [bench/summary.tsv](logs_solver_run1/bench/summary.tsv)

### On the wall vs internal gap

cholesky's wall=61 s contrasts with internal=14 s. Most of the ~47 s gap is **Python interpreter startup + torch / mintpy import + CUDA lazy init**, a fixed cost paid once per `--dostep invert_network` invocation. On lstsq the internal=228 s makes that 47 s relatively invisible.

In other words, the **fairest comparison of practical speedup is internal time at 16.5×**. Pipelines such as the full `run_smallbaselineApp` (`load_data` … `hdfeos5`) that pay Python startup once converge towards the internal ratio.

---

## 3. Numerical equivalence (RMS comparison)

We compute the per-pixel RMS of `timeseries.h5` from each solver and normalise it by the `ts_lstsq` per-pixel temporal std as the inter-solver tolerance metric. Pixels where `std(ts_lstsq, axis=0) < 1e-6 m` (1 of 270,000) are excluded from normalised statistics — the signal is essentially zero so the divisor is too small to be meaningful.

| Statistic | Absolute RMS (m) | Normalised RMS = `RMS(ts_chol − ts_lstsq) / std(ts_lstsq)` |
|---|--:|--:|
| count | 270,000 | 269,999 |
| min | 0.0 | 4.63e-8 |
| median | 5.48e-9 | 5.41e-7 |
| p99 | 2.16e-8 | 1.89e-6 |
| **max** | **1.59e-7 m (~0.16 µm)** | **1.19e-5** |
| mean | 6.59e-9 | 6.39e-7 |

Reference: per-pixel signal std of `ts_lstsq` is median 9.18 mm, p99 42.8 mm, max 113 mm.

Raw artifacts: [compare/summary.txt](logs_solver_run1/compare/summary.txt) / [compare/rms_cholesky_vs_lstsq.json](logs_solver_run1/compare/rms_cholesky_vs_lstsq.json)

### Interpreting the numerical equivalence

- **Maximum absolute RMS is 0.16 µm** = 1.59e-7 m. Seven orders of magnitude below the maximum real-data signal (113 mm).
- **Maximum normalised RMS is 1.19e-5**. About 1/8 of the reference value (`1e-4 × signal scale`) used as acceptance criteria in [Issue #4](https://github.com/s-sasaki-earthsea-wizard/MintPy/issues/4).
- float32 machine epsilon is 1.2e-7. The SBAS design matrix has measured cond ≤ 10³, so going through normal equations gives `cond(G^T G) ≤ 10⁶` — relative error of ~0.1 would be theoretically possible, yet on real data the dominant signal **leaves only round-off-level differences on top of itself**.
- NaN/Inf is 0 for both cholesky and lstsq (out of 26.46M values). Rank-deficient pixels: 0 for both.

> **What level of difference is meaningful** is not a question this report decides. The numbers above are reported against the Issue #4 acceptance criteria (RMS < 1e-4 × signal scale).

---

## 4. GPU kernel breakdown (per chunk)

Both solvers profiled with `torch.profiler` under `schedule(wait=1, active=1)` for one chunk. The per-chunk step boundary is enforced by monkey-patching the dispatch function (`_solve_cholesky` / `_solve_lstsq`) and inserting a single `prof.step()` per chunk ([profile_torch_solver.py @ 0682c4c](https://github.com/s-sasaki-earthsea-wizard/mintpy-benchmark/blob/0682c4c/profile_torch_solver.py)).

Raw artifacts:
- cholesky: [profile/cholesky/parsed.md](logs_solver_run1/profile/cholesky/parsed.md) / [key_averages.txt](logs_solver_run1/profile/cholesky/tb_trace/) / [trace.json](logs_solver_run1/profile/cholesky/tb_trace/) (139 KB)
- lstsq: [profile/lstsq/parsed.md](logs_solver_run1/profile/lstsq/parsed.md) / [trace.json](logs_solver_run1/profile/lstsq/tb_trace/) (3.90 GiB; `key_averages` OOMs and is reduced offline by `parse_trace.py`, same procedure as [report_profile.md](report_profile.md))

### 4.1 Top-level (per chunk)

| Metric | cholesky | lstsq | Ratio (lstsq / cholesky) |
|---|--:|--:|--:|
| `ProfilerStep#1` (= 1 chunk wall) | 1.161 s | 19.869 s | **17.1×** |
| Total `kernel` time | 24.20 ms | 10.409 s | **430×** |
| Total `cuda_runtime` (host launch) time | 1.097 s | 18.210 s | **16.6×** |
| Number of `kernel` events | 57 | 3,841,835 | **67,400×** |
| Number of `cuda_runtime` events | 85 | 3,861,259 | **45,400×** |
| Number of `gpu_memcpy` events | 8 | 7 | ≈ same |
| Memcpy HtoD (total) | 7.73 ms | 7.75 ms | ≈ same |

For cholesky the kernel time (24 ms) and launch overhead (1.1 s) are inverted (launch >> compute), but the **absolute time is so small** that the overhead is itself negligible. For lstsq the **launch overhead 18.2 s and kernel time 10.4 s overlap in parallel** to produce a 19.9 s wall — a textbook launch-bound profile.

> **Important**: cholesky's `cuda_runtime=1.097 s` includes the ProfilerStep overhead, so it is not pure launch time (`overhead=1.060 s` accounts for almost all of it). Real launch overhead is on the order of 24 ms.

### 4.2 Top GPU kernels — cholesky (per chunk)

| Rank | Total | Calls | Kernel | Role |
|--:|--:|--:|---|---|
| 1 | 7.18 ms | 1 | `cutlass_80_simt_sgemm_128x32_8x5_nt_align1` | `N = G_w^T G_w` (batched GEMM) |
| 2 | 2.52 ms | 5 | `potrf_syrk_T16_nc_kernel<float, 5, 4, 4, 5, 4>` | Cholesky factor (panel syrk) |
| 3 | 2.51 ms | 3 | elementwise `Mul` | weight broadcasting |
| 4 | 2.44 ms | 1 | `gemv2N_kernel<...>` | `G_w^T y` (batched GEMV) |
| 5 | 1.69 ms | 1 | `triu_tril_kernel` | triangularisation |
| 6 | 1.22 ms | 1 | `strsv_trans_kernel_outplace_batched` | back-solve (L^T) |
| 7 | 1.17 ms | 6 | `potrfBatch_trsm_lower<float, float, 16>` | Cholesky panel trsm |
| 8 | 1.12 ms | 7 | `potrf_cta_lower_batch<float, float, 16>` | Cholesky CTA-level |
| 9 | 1.07 ms | 1 | `strsv_notrans_kernel_outplace_batched` | back-solve (L) |
| 10 | 906 µs | 1 | `potrf_syrk_nc_kernel` | Cholesky panel syrk (closing) |

→ Essentially "**1 GEMM + 1 batched Cholesky factor + 2 batched trsm + 1 batched GEMV**". Per chunk, GPU work is one batch.

### 4.3 Top GPU kernels — lstsq (per chunk)

| Rank | Total | Calls | Kernel | Role |
|--:|--:|--:|---|---|
| 2 | 3.995 s | **1,882,091** | `ormtr_gemv_c<float, 4>` | gels back-substitution (per-iteration support) |
| 3 | 3.669 s | **19,403** | `geqr2_smem_domino_fast<float, float, 8, 512>` | gels QR factorization |
| 4 | 2.691 s | **1,882,091** | `ormtr_gerc<float, 5, 3, 1>` | gels back-substitution (per-iteration apply) |
| 5 | 42.5 ms | 58,209 | `copy_info_kernel` | per-pixel info bookkeeping |
| 6 | 21.2 ms | 19,403 | `Memset (Device)` | per-pixel initialisation |
| 11 | 974 µs | 2 | `batch_trsm_left_kernel` | back-solve |

Per-pixel structure:
- `1,882,091 / 19,403 = 97 = num_unknown` (= num_date − 1)
- The gels driver runs **an independent iterative QR per pixel**: 97 Householder reflections per pixel, with two support kernels per reflection
- `19,403 + 1,882,091 + 1,882,091 + 58,209 ≈ 3.84 M kernel launches`

Compared with cholesky's 57 launches: the structural gulf between **batched solver vs per-pixel iterative solver** translates into the kernel-launch overhead difference of 18 s vs tens of ms (effective).

---

## 5. Why cholesky is faster (kernel-level interpretation)

Conceptually the chunk-internal GPU work for the cholesky path is:

```
N = (G * w[:, None])^T @ (G * w[:, None])     # 1 batched GEMM, (n, num_unknown, num_unknown)
r = (G * w[:, None])^T @ (y * w[:, None])     # 1 batched GEMV, (n, num_unknown)
L = cholesky(N)                                # 1 batched cholesky_ex (potrf)
X = cholesky_solve(r, L)                       # 1 batched 2x triangular solve
```

All of this fits within cuSolver / cuBLAS's **batched APIs**, one launch each (with n pixels in parallel inside the launch). Some panel/CTA splits add a few launches (potrf-related ~18 launches), but the total per chunk is still **57 launches**.

The lstsq path:
```
torch.linalg.lstsq(A_batch, b_batch)  # cuSolver gels driver (QR)
```

cuSolver `gels` exposes a batched API, but internally runs **an independent iterative Householder QR per pixel**. With 2 sub-kernels per reflection × 97 reflections × 19,403 pixels = **3.86 M launches per chunk**.

> Each launch carries a host-side `cudaLaunchKernel` overhead of ~5 µs (the standard CUDA runtime figure). 3.86M × 5 µs = **18 s** of overhead — matching the measured `cuda_runtime=18.21 s`.

This is consistent with the Phase 2 prediction in [report_profile.md](report_profile.md):

| | gels (lstsq) | normal-eq + Cholesky |
|---|---|---|
| Per-chunk launches | ~1.88 M | ~5 |
| Per-chunk GPU compute | 10.4 s | ~6 s (theoretical) |
| Per-chunk launch overhead | ~18 s | ≪ 1 s |
| Predicted chunk wall | 19.9 s | ≈ 6 s |

Measured (this report §2): cholesky internal/wall is 13.83 s / 61.34 s aggregated over 14 chunks — about 1 s wall per chunk on average (consistent with ProfilerStep#1's 1.161 s). **Faster than the predicted 6 s/chunk** — `cuSolver / cuBLAS` batched implementations are more efficient than expected.

---

## 6. Measurement caveats / notes

### 6.1 `key_averages()` OOM on the lstsq profile

`prof.key_averages().table()` on the lstsq path tries to materialise every kineto event as a Python object. At this dataset size (15.94 M events even with `active=1`) it triggers a `std::bad_alloc`-class host-RSS spike (this run reached 45 GiB RSS before SIGINT). This is the known behaviour recorded as the 2026-05-03 incident in [report_profile.md](report_profile.md) (`key_averages.txt` is best-effort; `tb_trace/*.json` is authoritative).

For this report, [`parse_trace.py`](parse_trace.py) reduces the trace JSON offline and obtains the kernel breakdown. The fact that `key_averages()` is interrupted partway is not an anomaly but the designed behaviour (the cholesky side has only 561 events and finishes instantly).

### 6.2 Python startup overhead

cholesky's wall=61 s vs internal=14 s: the 47 s gap is the fixed cost of (Python interpreter init + torch import + mintpy import + CUDA lazy init). It is structural to running `--dostep invert_network` standalone, and is paid once when running the full 18-step pipeline through `smallbaselineApp.py`.

### 6.3 SSD vs NAS

Per [`feedback_bench_io_isolation.md`](https://github.com/s-sasaki-earthsea-wizard/MintPy/issues), wall time is measured on SSD. RMS validation is I/O-independent so it shares the SSD run.

### 6.4 update-mode skip pitfall

Before each shot, `timeseries.h5 / temporalCoherence.h5 / numInvIfgram.h5` are removed via `rm -f` (`run_one_shot` in [`run_solver_comparison.sh @ 0682c4c`](https://github.com/s-sasaki-earthsea-wizard/mintpy-benchmark/blob/0682c4c/run_solver_comparison.sh)). `invert_network` is in MintPy's update-mode skip list, so leftover h5 files cause a silent no-op.

---

## 7. Conclusions (observations only — judgements out of scope)

1. `_SOLVER='cholesky'` is **16.5× faster on internal time and 4.5× faster on wall** versus `_SOLVER='lstsq'` on this dataset.
2. Numerical equivalence: **normalised max RMS 1.19e-5** at float32 round-off level. Zero NaN/Inf, zero rank-deficient pixels.
3. The mechanism behind the speedup is the **difference in kernel-launch structure** (batched GEMM/Cholesky 5 launches vs gels per-pixel iterative QR 3.86 M launches per chunk). Launch-overhead reduction dominates over the difference in compute itself.
4. Memory traffic (HtoD/DtoH) is comparable between the two (HtoD 7.7 ms / chunk), so optimising that axis remains untouched on this dataset.

**Out of scope**: whether to retain the `_solve_lstsq` path, whether to promote `_SOLVER` to a template flag, whether to upstream cholesky — handled separately in [Issue #4](https://github.com/s-sasaki-earthsea-wizard/MintPy/issues/4).
