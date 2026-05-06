# FernandinaSenDT128 baseline runtime report

Run date: 2026-04-26 (JST)
Target: full 18-step `smallbaselineApp.py FernandinaSenDT128.txt`
Harness: [scripts/run_bench.sh](../scripts/run_bench.sh)
Logs: [logs_baseline/](../logs_baseline/)

---

## 1. Machine

| Item | Value |
|---|---|
| OS | Ubuntu 24.04.3 LTS (kernel 6.17.0-20-generic) |
| CPU | Intel Core Ultra 9 285H — 16 cores / 16 threads, max 5.4 GHz |
| L2 / L3 cache | 28 MiB / 24 MiB |
| Memory | 93 GiB (Swap 8 GiB) |
| GPU | NVIDIA GeForce RTX 5080 (16 GiB, Blackwell sm_120) |
| GPU driver / CUDA | 590.48.01 / CUDA 13.1 driver, nvcc 13.0 |
| Data location | local CIFS share over 1 GbE |
| Python | 3.12.3 (`.venv/`, managed by uv) |
| Key libraries | numpy 2.4.4, scipy 1.17.1, h5py 3.16.0, dask 2026.3.0, torch 2.11.0+cu128 (unused), cupy-cuda12x 14.0.1 (unused) |

See [logs_baseline/machine_info.txt](../logs_baseline/machine_info.txt) for raw output.

> **Note**: The ~1 GB input dataset lives on a CIFS share reached over 1 GbE, so I/O-bound steps (`load_data`, etc.) are slower than they would be on local NVMe.

---

## 2. Dataset size

Main dimensions of [FernandinaSenDT128/mintpy/inputs/ifgramStack.h5](../../FernandinaSenDT128/mintpy/inputs/):

| Item | Value |
|---|---|
| Interferogram pairs | 288 |
| Acquisition dates | 98 |
| Pixels | 270,000 (likely 450 × 600) |
| Valid pixels for inversion | 157,667 (58.4%) |

---

## 3. Per-step runtime

`wall` is wall-clock time from `/usr/bin/time` (includes Python process startup); `internal` is the `Time used:` value MintPy itself reports (pure compute time). `startup` ≈ wall − internal is the per-step CLI boot + import cost.

| # | step | wall (s) | internal (s) | startup (s) | max RSS (MB) | share (internal) |
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
| | **total** | **991.0** | **500.8** | **489.5** | — | 100% |

### Caveat on the wall column

Steps were run one-per-process via `--dostep`, so each row's `wall` includes the Python interpreter + MintPy import cost (~20–30 s / step, totalling 489.5 s). A normal end-to-end invocation (`smallbaselineApp.py FernandinaSenDT128.txt`) pays that cost **once**, so the realistic wall time is roughly `internal total + 30 s ≈ 8.8 min`.

---

## 4. Bottleneck analysis (GPU acceleration priority)

The top three steps by `internal` time account for **70%** of the total.

| Rank | step | internal | share | Workload (GPU suitability) |
|--:|---|--:|--:|---|
| 1 | **invert_network** | 218.0 s | 43.5% | Per-pixel SBAS least-squares inversion (massively parallel — strong GPU fit) |
| 2 | **load_data** | 88.6 s | 17.7% | I/O + format conversion (NAS-bound; little GPU upside) |
| 3 | modify_network | 43.9 s | 8.8% | Pair filtering and coherence calculation |
| 4 | residual_RMS | 25.8 s | 5.1% | Residual RMS aggregation |
| 5 | quick_overview | 25.1 s | 5.0% | Mean coherence / velocity overview |
| 6 | reference_date | 17.7 s | 3.5% | Time-series reference-date subtraction |
| 7 | deramp | 16.4 s | 3.3% | Plane / quadratic fit removal |
| 8 | reference_point | 15.4 s | 3.1% | Reference-point pick |
| 9 | correct_topography | 14.9 s | 3.0% | DEM residual correction |
| 10 | google_earth | 9.1 s | 1.8% | Output formatting |
| 11 | geocode | 8.8 s | 1.8% | Geocoding (resampling) |
| 12 | correct_troposphere | 7.3 s | 1.5% | ERA5 correction (cached) |
| 13 | velocity | 5.1 s | 1.0% | Linear-trend regression |
| - | correct_SET | 2.1 s | 0.4% | Solid-earth tides |
| - | correct_ionosphere | 1.5 s | 0.3% | (disabled) |
| - | correct_LOD | 0.9 s | 0.2% | (Sentinel: not needed) |
| - | correct_unwrap_error | 0.1 s | ≈0 | (disabled) |
| - | hdfeos5 | 0.1 s | ≈0 | (disabled) |

### First GPU target: `invert_network`

- Single largest hotspot — **43.5%** of total compute on its own.
- Algorithmic shape: 157,667 pixels × (288 × 98) least-squares — embarrassingly parallel, maps directly to batched linear algebra (`torch.linalg.lstsq`, `cp.linalg.lstsq`).
- Implementation lives in `src/mintpy/ifgram_inversion.py`.

### Secondary candidates

- **load_data** is I/O-bound, so the right first move is to copy the dataset onto local SSD and re-measure to isolate the share-network effect. GPU has limited headroom here.
- **modify_network / residual_RMS / quick_overview** are each in the 25–45 s range. Re-evaluate after `invert_network` is GPU-accelerated to see which is the next mountain.
- **deramp / correct_topography** are least-squares fits and would map well to GPU, but the absolute time saved (~15 s each) is small.

---

## 5. Suggested next actions

1. Read `src/mintpy/ifgram_inversion.py` to understand the LSQ inversion loop and its I/O pattern.
2. Prototype a CuPy / PyTorch batched inversion in a separate module (e.g. `mintpy/ifgram_inversion_gpu.py`).
3. Add a switch in the template (e.g. `mintpy.networkInversion.backend = cpu|cupy|torch`).
4. Re-run the same harness ([run_bench.sh](../scripts/run_bench.sh)), save under `benchmark/logs_gpu/`, and write a comparison report (`report_gpu.md`).
