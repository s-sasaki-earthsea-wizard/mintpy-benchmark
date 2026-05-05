# `invert_network` chunk_size sweep (torch backend)

Date: 2026-04-26 (JST)
Subject: Validate the sweet spot of `mintpy.networkInversion.gpuChunkSize`, and pre-separate the GPU-lstsq intrinsic time from surrounding overhead (a preparatory step for follow-up #1 of [Issue #2](https://github.com/s-sasaki-earthsea-wizard/MintPy/issues/2))

Harness: [benchmark/run_chunk_sweep.sh](run_chunk_sweep.sh)
Log set: [logs_chunk_sweep/](logs_chunk_sweep/)

---

## 1. Goal

In the SSD-Torch bench from [report_torch.md](report_torch.md), the internal time of `invert_network` plateaued at **257.4 s (1.43× speedup over CPU)**. The expected ceiling factors for that smaller-than-hoped speedup were the three hypotheses raised in the prior session ([2026-04-26 notes](../.claude-notes/2026-04-26_invert-network-torch.md) L101–105):

| Hypothesis | Expected behaviour |
|---|---|
| (a) Small chunks → kernel launch + host↔device copies dominate | Wall increases sharply at small chunks |
| (b) Mid chunks → cuSOLVER workspace + memory locality form a sweet spot | The auto value (19,403) is fastest |
| (c) Large chunks → cache-miss / memory-bandwidth bound | Performance degrades near the VRAM ceiling |

This sweep is run **before introducing a profiler**: by varying chunk_size alone we determine which of (a)–(c) is in play, which in turn lets us form a prior on what the profiler should focus on (kernels? memcpy? Python overhead?).

---

## 2. Measurement conditions

| Item | Value |
|---|---|
| Machine | Intel Core Ultra 9 285H (16C) / 93 GiB RAM / NVIDIA RTX 5080 (16 GiB, Blackwell sm_120) |
| Storage | Local NVMe SSD (`~/MintPy_bench/...`) |
| Python / torch | 3.12.3 / 2.11.0+cu128 |
| Dataset | FernandinaSenDT128 (288 ifgrams × 98 acquisitions, 269,999 inverted pixels) |
| Backend | `mintpy.networkInversion.backend = torch` |
| Single step | `--dostep invert_network` only (full pipeline not needed; phase 1 already validated end-to-end) |
| Measurement | `/usr/bin/time -v` for wall + max RSS; logs are scraped for `Time used:` (internal) and the resolved `chunk_size` / `num_chunks` |
| Environment isolation | Other heavy processes (CPU-heavy Docker container) were paused during the sweep |

### Sweep matrix

| `gpuChunkSize` | Resolved chunk_size | num_chunks | Intended regime |
|--:|--:|--:|---|
| 1000 | 1,000 | 270 | very-small (overhead-dominated regime) |
| 5000 | 5,000 | 54 | small |
| 10000 | 10,000 | 27 | medium |
| 0 (auto) | **19,403** | **14** | auto-derived (`0.4 × free_VRAM / per_pixel_bytes`) |
| 30000 | 30,000 | 9 | large |
| 40000 | 40,000 | 7 | near-VRAM (`80,000` was excluded as it OOMs the workspace) |

Each chunk_size was run in **2 round-robin rounds** (Round 1: 1k → 5k → 10k → auto → 30k → 40k; Round 2 same order), for 12 runs total. `--dostep invert_network` skips when its outputs already exist (update mode), so before each run `timeseries.h5` / `temporalCoherence.h5` / `numInvIfgram.h5` were deleted to force re-execution.

> **Methodology bug, recorded for posterity**: the initial sweep was invalidated end-to-end by two issues — (1) update-mode skip, and (2) the previous run's `gpuChunkSize` was baked into `smallbaselineApp.cfg` in `work_dir` and survived template merging. The final harness fixes both: it deletes the outputs and explicitly writes `gpuChunkSize` on every run.

---

## 3. Results

### 3.1 Aggregate

| `gpuChunkSize` | num_chunks | wall mean (s) | wall min (s) | internal mean (s) | internal min (s) | per-pixel (ms) | max RSS (GiB) |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 1,000 | 270 | 271.81 | 266.60 | 246.95 | 243.70 | 0.915 | 2.50 |
| 5,000 | 54 | 273.64 | 271.29 | 246.20 | 244.00 | 0.912 | 2.52 |
| 10,000 | 27 | 268.73 | 263.41 | 245.30 | 242.60 | 0.909 | 2.53 |
| **19,403 (auto)** | **14** | **266.86** | **265.40** | **242.95** | **242.80** | **0.900** | **2.57** |
| 30,000 | 9 | 277.49 | 276.05 | 250.75 | 248.60 | 0.929 | 2.61 |
| 40,000 | 7 | 268.23 | 263.35 | 244.80 | 242.10 | 0.907 | 2.63 |

per-pixel = internal mean / 269,999 pixels × 1000. Raw data: [logs_chunk_sweep/summary.tsv](logs_chunk_sweep/summary.tsv).

### 3.2 Round-by-round raw data

| `gpuChunkSize` | r1 wall | r2 wall | r1 internal | r2 internal |
|--:|--:|--:|--:|--:|
| 1,000   | 277.02 | 266.60 | 250.2 | 243.7 |
| 5,000   | 271.29 | 275.99 | 244.0 | 248.4 |
| 10,000  | 274.05 | 263.41 | 248.0 | 242.6 |
| 19,403  | 268.31 | 265.40 | 243.1 | 242.8 |
| 30,000  | 276.05 | 278.93 | 248.6 | 252.9 |
| 40,000  | 263.35 | 273.10 | 242.1 | 247.5 |

Round-to-round noise: roughly ±2–5 s (typical for ML-style single-shot benches).

---

## 4. Analysis

### 4.1 The sweep curve is essentially flat

Internal time stays **within 242–251 seconds across the whole range (a 9-second / ±1.8% spread)**. With round-to-round noise around ±5 s, the headline observation is that **changing chunk_size barely moves performance**.

### 4.2 Verifying hypotheses (a)–(c)

| Hypothesis | Expectation | Observation | Verdict |
|---|---|---|---|
| (a) Small-chunk overhead | cs=1k drops sharply | cs=1k is +4 s (+1.6%) over auto only | **Rejected**: kernel launch + memcpy overhead is negligible |
| (b) Auto sweet spot | auto (19,403) is fastest | Auto is indeed fastest (242.95 s), but the gap to others is ≤ 3% | **Weakly supported**: a sweet spot exists, but the ROI is small |
| (c) Large-chunk degradation | cs=40k regresses | cs=40k = 244.8 s (≈ auto), cs=30k = 250.75 s (+3.2%) | **Not observed**: no degradation at the VRAM ceiling |

The rejection of (a) is the important one. With 270 chunks (cs=1k) the wall barely moves, which lets us **bound the per-chunk overhead**:

$$
\text{per-chunk overhead} \approx \frac{246.95 - 242.95}{270 - 14} = \frac{4.0\,\text{s}}{256\,\text{chunks}} \approx 16\,\text{ms/chunk}
$$

At auto (14 chunks) the total chunk-launch cost is `14 × 16 ms = 0.22 s` — a mere **0.09%** of internal time.

### 4.3 wall − internal is constant

| `gpuChunkSize` | wall − internal (s) |
|--:|--:|
| 1k | 24.86 |
| 5k | 27.44 |
| 10k | 23.43 |
| 19k | 23.91 |
| 30k | 26.74 |
| 40k | 23.43 |

The gap is **24–27 s, independent of chunk_size**. This is the cost surrounding `smallbaselineApp.py --dostep invert_network` (Python startup + template/cfg parsing + post-step `generate_mask.py` ~few seconds + I/O finalization). Out of scope for this sweep.

### 4.4 The mild slowdown at cs=30,000

Only this point shows a +6–8 s regression. With 9 chunks the last chunk holds 29,999 pixels (full), so the workload is nearly even. Plausible causes are cuSOLVER workspace boundary alignment or VRAM allocator behaviour, but **a single-point regression cannot support a strong claim** (intra-round spread is ±2.2 s and the neighbouring cs=40k point at +1.85 s is in the stable range). **The profiler will not pursue this further.**

---

## 5. Conclusions

1. **The chunk_size sweet spot is auto (19,403) and is indeed fastest, but the gap to other sizes is ≤ 3% — practically flat.**
2. **Chunk launch + host↔device memcpy overhead is ~16 ms/chunk, well under 0.1% of internal time at auto.**
3. **Therefore tuning `chunk_size` is not a productive route for accelerating invert_network.** The 243-second internal time has its origin elsewhere.
4. The current `_auto_chunk_size` function picks a reasonable VRAM-based value; no tuning needed.

---

## 6. Implications for the next step

As a null result, this sweep gives a useful constraint for the upcoming profiling plan:

- Chunk launch overhead is negligible → **no need to track kernel launch counts on the `torch.profiler` timeline**.
- Performance does not improve at large chunks (cs=40k > cs=30k yet ≈ auto) → cuSOLVER's `gels` driver is already saturated at mid-size batches → **GPU compute itself is unlikely to be the dominant term**.
- The 24–27 s fixed overhead lives on the smallbaselineApp side → **when decomposing inside invert_network (~243 s), measure below `run_ifgram_inversion_patch`, not at the smallbaselineApp wrapper level**.

Primary candidates for [Issue #2](https://github.com/s-sasaki-earthsea-wizard/MintPy/issues/2) follow-up #1 (profiling):

| Candidate | Expected contribution |
|---|---|
| **`read_stack_obs`** (h5py read of 288 × 269,999 pixels) | Tens of seconds, I/O-bound |
| **`calc_weight_sqrt`** (CPU, coherence → weight, 3 chunks × 100k pixels) | Tens of seconds, CPU-bound |
| GPU lstsq inside `estimate_timeseries_batch` | The remainder |
| Host-device copies (`y_dev`, `w_dev` per chunk) | Already shown to be small |

→ Combining **py-spy** (top-down hot path of Python functions) with **torch.profiler** (per-region breakdown of `read_stack_obs` / `calc_weight_sqrt` / GPU regions) will pin down the read / weight / lstsq / memcpy ratio.

---

## 7. Reproduction

```bash
# Prerequisite: dataset SSD-copied to ~/MintPy_bench/FernandinaSenDT128/mintpy/
WORK_DIR=$HOME/MintPy_bench/FernandinaSenDT128/mintpy \
    bash benchmark/run_chunk_sweep.sh
```

12 runs × ~270 s ≈ **about 55 minutes**. Pause CPU-heavy processes (Docker etc.) beforehand — `Time used:` in this sweep is sensitive to CPU contention.
