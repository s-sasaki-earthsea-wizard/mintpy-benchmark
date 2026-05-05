# FernandinaSenDT128 GPU acceleration report (torch backend)

Date: 2026-04-26 (JST)
Subject: 18-step run of `smallbaselineApp.py` with `mintpy.networkInversion.backend = torch` enabled
Harness: [benchmark/run_bench.sh](../scripts/run_bench.sh)
GPU template: [FernandinaSenDT128_torch.txt](../fixtures/FernandinaSenDT128_torch.txt)

Log set:

| Label | Storage | backend | Log directory | Inverted pixels |
|---|---|---|---|--:|
| **NAS-CPU** (reference) | NAS (CIFS) | cpu | [logs_baseline/](../logs_baseline/) | 157,667 |
| **SSD-CPU** | Local NVMe | cpu | [logs_cpu_local/](../logs_cpu_local/) | 269,999 |
| **SSD-Torch** | Local NVMe | torch | [logs_torch/](../logs_torch/) | 269,999 |

Implementation: [`src/mintpy/ifgram_inversion_gpu.py`](../../src/mintpy/ifgram_inversion_gpu.py) + dispatch in [`src/mintpy/ifgram_inversion.py`](../../src/mintpy/ifgram_inversion.py)
Commit: [8ab560fb](https://github.com/s-sasaki-earthsea-wizard/MintPy/commit/8ab560fb)

---

## 1. Measurement conditions

| Item | Value |
|---|---|
| Machine | Intel Core Ultra 9 285H (16C) / 93 GiB RAM / NVIDIA RTX 5080 (16 GiB, Blackwell sm_120) |
| OS | Ubuntu 24.04.3 LTS, kernel 6.17.0-20-generic |
| Python | 3.12.3, `.venv/` (managed by uv) |
| Key libraries | torch 2.11.0+cu128, numpy 2.4.4, scipy 1.17.1, h5py 3.16.0 |
| GPU driver / CUDA | 590.48.01 / CUDA 13.1 driver, nvcc 13.0 |
| Execution | All 18 steps run with `--dostep`, one process per step (cold start) |

See each `logs_*/machine_info.txt` for full details.

---

## 2. Dataset scale — `invert_network` workload differs between NAS and SSD

The NAS baseline (`logs_baseline/`) and the SSD benches (`logs_cpu_local/`, `logs_torch/`) see different contents in `avgSpatialCoh.h5` (part of the quality mask), so the number of pixels selected for inversion in `invert_network` differs:

| Variant | Inverted pixels | Ratio |
|---|--:|--:|
| NAS baseline | 157,667 | 58.4% |
| SSD CPU / Torch | 269,999 | 100.0% |

When the NAS baseline ran, `avgSpatialCoh.h5` already existed from a prior run and zero-marked roughly 110k pixels, leaving fewer valid pixels. The SSD benches ran cold-start, so `quick_overview` regenerated the file and almost every pixel was deemed valid.

> **It is therefore unfair to directly compare the `invert_network` time of NAS-CPU and SSD-CPU.** This report uses **SSD-CPU vs SSD-Torch as the apples-to-apples comparison**, with NAS-CPU listed only for reference.

---

## 3. Per-step wall time

Wall-clock time as measured by `/usr/bin/time -v` (Python startup included). The steps with `exit=1` are cascade failures caused by a missing ERA5 grib cache ([§7](#7-bench-environment-constraints)) and are unrelated to the bench's main subject.

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
| | **total wall** | **991** | **808** | **720** | |

⚠ The NAS-CPU `invert_network` runs a different workload (157k vs 269k pixels) and is excluded from the comparison.

---

## 4. `invert_network` detail (apples-to-apples)

`internal` is the `Time used:` value MintPy itself reports — pure processing time with Python startup excluded.

| Variant | pixels | wall (s) | internal (s) | per-pixel internal (ms) | speedup vs SSD-CPU |
|---|--:|--:|--:|--:|--:|
| SSD-CPU (baseline) | 269,999 | 386.09 | 367.0 | 1.359 | 1.00× |
| **SSD-Torch** | 269,999 | 280.39 | **257.4** | **0.953** | **1.43×** |
| NAS-CPU (different workload) | 157,667 | 237.85 | 218.0 | 1.382 | — |

**Result**: `invert_network` is **1.43×** faster on internal time and **1.38×** faster on wall.

### Sanity check on the GPU path

From [logs_torch/invert_network.log](../logs_torch/invert_network.log):

```
estimating time-series via torch backend (batched, GPU)
GPU auto chunk_size = 19403 pixels (free VRAM 15.1 GiB)
estimating time-series via torch batched WLS in 14 chunk(s) of up to 19403 pixels ...
```

`mintpy.networkInversion.gpuChunkSize = 0` (auto) was used; from 15.1 GiB of free VRAM the auto-sizer picked 19,403 pixels per chunk and finished in 14 chunks.

---

## 5. Numerical equivalence

### Unit tests ([tests/test_ifgram_inversion_gpu.py](../../tests/test_ifgram_inversion_gpu.py))

On synthetic data (98 dates × 288 pairs, same shape as FernandinaSenDT128):

| Case | Expected accuracy | Result |
|---|---|:-:|
| WLS, no NaN | rms < 1e-5 × signal | OK |
| WLS, 3% NaN (redundant network) | rms < 1e-4 × signal | OK |
| OLS, no NaN | rms < 1e-5 × signal | OK |
| min_norm_phase | rms < 1e-5 × signal | OK |
| chunk size invariance | rtol 1e-6 | OK |
| Unsupported backend → error | ValueError | OK |

All cases match the per-pixel CPU `scipy.lstsq` reference at the float32 round-off level.

### Real-data smoke test (157,667 pixels, FernandinaSenDT128)

| Item | Value |
|---|---|
| NaN/Inf in `ts` output | **0 / 0** (out of 15.4M values) |
| NaN/Inf in `tcoh` output | **0 / 0** |
| RMS difference of `tcoh` vs CPU | **2.33e-7** (max 1.87e-5) |

No rank-deficient pixels were observed on real data, and the result agrees with CPU to the float32 limit. **The full-rank assumption of the `gels` driver** poses no issue on this dataset.

---

## 6. Why only 1.4×

The `Time used: 257.4 s` figure (SSD-Torch internal) is not pure lstsq time — it covers all of `run_ifgram_inversion_patch`:

```
read_stack_obs + ref_phase backfill + mask construction   ┐
                                                          ├─ remains CPU numpy / h5py
calc_weight_sqrt (coherence → weight, chunked)            ┘

estimate_timeseries_batch (the hot loop GPU-ised by this commit) ── primary hot loop

phase → meter unit conversion + reshape                   ── ~0 s
```

Inside `estimate_timeseries_batch`:
- 14 chunks, with each chunk doing host→device copies (`y`, `weight_sqrt`, `valid` mask)
- `torch.linalg.lstsq` (CUDA `gels` driver, batched)
- tcoh computation + cumsum (GPU)
- device→host copies (`ts`, `tcoh`, `nobs`)

The `gels` driver is QR factorization on a fast full-rank path, but even batched, on cuSOLVER it solves 19,403 matrices of shape `(num_pair, num_unknown) = (288, 97)` in parallel — not as efficient as a pure GEMM (workspace allocation overhead, plus wasted FLOPs on weight-zero rows used to mask NaNs).

**In short, the lstsq loop itself is GPU-ised, but the surrounding CPU work (read, weight) and the GPU-side overhead inside the kernel are non-trivial, capping the current speedup at 1.4×.** Profiling will break this down into components (§8 follow-up).

---

## 7. Bench environment constraints

A subset of steps starting at `correct_troposphere` reported `exit=1` (cascade failure) on both `SSD-CPU` and `SSD-Torch`. The cause is **a missing ERA5 grib cache that PyAPS requires**: it was not present in the SSD copy, so `number of grib files used: 0` led `progressBar` to raise `ZeroDivisionError` (`src/mintpy/objects/progress.py:109`). On the NAS baseline some prior cache path resolved the grib successfully.

The bench's main subject (`invert_network`) is unaffected: GPU and CPU benches fail the same steps for the same reason, so comparability is preserved.

---

## 8. Follow-ups

| # | Task | Expected impact |
|--:|---|---|
| 1 | Profile `invert_network` with py-spy / cProfile and break out read / weight / lstsq / copy | Quantify the speedup ceiling |
| 2 | GPU-ise `calc_weight_sqrt` (chunked numpy → torch) | Cut weight build (~25 s/run estimated) |
| 3 | Add `--backend cupy` and bench it | Compare CuPy vs PyTorch characteristics |
| 4 | CPU fallback for rank-deficient pixels | Insurance for datasets that need it |
| 5 | Stand up an ERA5 grib cache for benches | Enable full-pipeline benches |
| 6 | Promote to an upstream PR (per the Wiki [Upstream-PR-Checklist](https://github.com/s-sasaki-earthsea-wizard/MintPy/wiki/Upstream-PR-Checklist)) | Contribute back to insarlab/MintPy |

---

## 9. Conclusions

- `mintpy.networkInversion.backend = torch` produces **no NaN/Inf on real data and matches CPU at the float32 round-off level**.
- `invert_network` is **1.43× faster on internal time and 1.38× faster on wall**.
- The lstsq portion is essentially GPU-ised; the remaining headroom lies in the **CPU-bound weight construction** and **host↔device copies at chunk boundaries**.
- Defaults stay upstream-compatible (`mintpy.networkInversion.backend = cpu`); users opt in via `--backend torch` or the template flag.
