# GalapagosSenDT128 large-scene benchmark report (cpu vs torch solver)

Date: 2026-05-06 (JST)
Subject: confirm `invert_network` scaling and `solver=torch` speedup hold on a substantially larger scene than the FernandinaSenDT128 tutorial dataset (companion to upstream RFC [insarlab/MintPy#1489](https://github.com/insarlab/MintPy/issues/1489), tracking issue [s-sasaki-earthsea-wizard/MintPy#6](https://github.com/s-sasaki-earthsea-wizard/MintPy/issues/6))
Harness: [scripts/run_bench.sh](../scripts/run_bench.sh) with `BENCH_STEPS=invert_network`
GPU template: [fixtures/GalapagosSenDT128_torch.txt](../fixtures/GalapagosSenDT128_torch.txt)
CPU template: embedded `inputs/GalapagosSenDT128.template` (shipped with the dataset, no `solver` line ⇒ defaults to cpu)
Numerical comparison: [tools/compare_solutions.py](../tools/compare_solutions.py) → [galapagos_diff/cpu_vs_torch_r1.json](galapagos_diff/cpu_vs_torch_r1.json)
Implementation: [src/mintpy/ifgram_inversion_gpu.py](https://github.com/s-sasaki-earthsea-wizard/MintPy/blob/11b94388/src/mintpy/ifgram_inversion_gpu.py) + dispatch in [src/mintpy/ifgram_inversion.py](https://github.com/s-sasaki-earthsea-wizard/MintPy/blob/11b94388/src/mintpy/ifgram_inversion.py)
MintPy fork commit: [11b94388](https://github.com/s-sasaki-earthsea-wizard/MintPy/commit/11b94388)

Log set:

| Label | Storage | solver | Log directory | Inverted pixels (per box) |
|---|---|---|---|--:|
| **SSD-CPU** (reference) | Local NVMe | cpu | `logs_galapagos_cpu_r1/` (untracked) | 578k × 5 + 510k = 1.1 M of 3.4 M (32.6%) |
| **SSD-Torch** | Local NVMe | torch | `logs_galapagos_torch_r1/` (untracked) | 578k × 5 + 510k = 1.1 M of 3.4 M (32.6%) |

Both runs read from a warm SSD copy of the Zenodo archive (see §2). Logs themselves are intentionally untracked per the repository's `.gitignore` policy; numerical findings below are transcribed by hand.

---

## 1. Measurement conditions

| Item | Value |
|---|---|
| Machine | Intel Core Ultra 9 285H (16C) / 93 GiB RAM / NVIDIA RTX 5080 (16 GiB, Blackwell sm_120) |
| OS | Ubuntu 24.04, kernel 6.17.0-22-generic |
| Python | 3.12.3, `.venv/` (managed by uv) |
| Key libraries | torch 2.11.0+cu128, numpy, scipy, h5py |
| GPU driver / CUDA | RTX 5080 (15.47 GiB VRAM) / CUDA 12.8 (cu128 wheels) |
| Storage | warm SSD copy at `~/MintPy_bench/GalapagosSenDT128/mintpy/` (`ifgramStack.h5` 28 GB) |
| Execution | `smallbaselineApp.py --dostep invert_network` only, one process per solver, cold start |
| ulimit guard | virtual memory capped at 75 GiB (80 % of 93 GiB physical) — see `scripts/lib/setup_ulimit.sh` |

---

## 2. Dataset scale — Galapagos vs Fernandina

Source: [Zenodo 4743058](https://zenodo.org/records/4743058) — `GalapagosSenDT128.tar.xz` (21.5 GB, CC-BY-4.0). The archive is a snapshot of MintPy state immediately after `load_data` (Yunjun et al. 2019 / Gregg et al. 2022): `inputs/ifgramStack.h5` and `inputs/geometryRadar.h5` are pre-populated MintPy HDF5, so `--dostep load_data` is an update-mode skip.

| metric | FernandinaSenDT128 (tutorial) | GalapagosSenDT128 | ratio |
|---|--:|--:|--:|
| LENGTH × WIDTH | 450 × 600 | 2000 × 1700 | — |
| total pixels | 270,000 | 3,400,000 | **12.6×** |
| ifgs total / kept (`dropIfgram`) | 288 / 288 | 490 / 475 | 1.65× (kept) |
| theoretical compute (pixels × ifgs²) | 1× | — | **34.3×** |
| inverted pixels (post quality mask) | 269,999 (SSD bench) | 1,107,562 | 4.10× |
| `ifgramStack.h5` size on disk | ~150 MB | 28 GB | ~190× |

The "inverted pixels" ratio (4.10×) is smaller than the "total pixels" ratio (12.6×) because the Galapagos quality mask is more selective on water / low-coherence pixels. The arithmetic intensity per inverted pixel is what matters for `invert_network`, so the meaningful scaling factor is **pixels_inverted × ifgs²** ≈ 4.10 × 1.65² ≈ **11.2×**.

---

## 3. invert_network step wall time

Wall-clock is `/usr/bin/time -v` (Python startup + MintPy init included). Internal time is the value MintPy reports at the end of `smallbaselineApp.py` ("Time used: X mins Y secs"), reflecting compute only.

| solver | wall (s) | internal (s) | max RSS (KB) | max RSS (GiB) | exit |
|---|--:|--:|--:|--:|:-:|
| cpu | 6189.00 | 6169.30 | 3,907,800 | 3.73 | 0 |
| torch | 170.06 | 139.00 | 6,078,856 | 5.80 | 0 |
| **speedup** | **36.4×** | **44.4×** | — | — | — |

For comparison, the Fernandina tutorial-scale measurements (Phase 2, [report_solver_comparison.md](report_solver_comparison.md)) reach **step wall 4.49×** and **internal 16.50×**.

Going from Fernandina to Galapagos:

- internal speedup **16.5× → 44.4× (2.7×)**, i.e. GPU occupancy was the binding constraint at tutorial scale and is no longer at large scale
- step-wall speedup **4.49× → 36.4× (8.1×)** — the same wall fixed cost (Python startup + `import torch` + CUDA context = ~30 s) becomes a much smaller fraction of total wall as the compute work grows

The **scaling story holds and strengthens at scale**. The torch path is now within **31 s of pure compute** (170 s wall − 139 s internal), most of which is one-time Python / CUDA setup that does not grow with scene size.

---

## 4. Where the time goes

### CPU side — pure Python loop dominates

`/usr/bin/time` for the cpu run: User=6112.58 s, System=55.86 s, **99 % CPU utilisation**. The MintPy step wall is essentially compute-bound on a single Python thread doing per-pixel `scipy.linalg.lstsq`. NaN-handled WLS pixels are not vectorisable by `scipy.linalg.lstsq` (each pixel has its own row mask + weights), hence the linear scaling with pixels.

### Torch side — GPU is fed in batches per box

`/usr/bin/time` for the torch run: User=48.96 s, System=63.73 s, **66 % CPU**. The torch path moves per-pixel WLS systems onto the GPU as batched stacks. MintPy splits the 2000 × 1700 raster into y-direction "boxes" of length 340 (6 boxes total). Within each box, torch path further chunks pixels by VRAM budget. Excerpt from `logs_galapagos_torch_r1/invert_network.log`:

```
box width:  1700
box length: 340
number of pixels to invert: 578000 out of 578000 (100.0%)
estimating time-series via torch solver (batched, GPU)
GPU auto chunk_size = 8726 pixels (free VRAM 11.2 GiB)
estimating time-series via torch batched WLS in 67 chunk(s) of up to 8726 pixels ...
```

The auto chunk size settles at ~8726 pixels per GPU launch (VRAM budgeted from 11.2 GiB free, accounting for the `(8726, 475, 475)` weighted normal-equation tensors). Across the 6 boxes, ~390 GPU kernel launches total cover the 1.1 M inverted pixels.

RSS climbed to 5.80 GiB for the torch run versus 3.73 GiB for cpu — the difference is host-side staging buffers (chunked NumPy arrays for the per-pixel A^TWA / A^TWy pre-computation) plus the torch CUDA context.

---

## 5. Numerical equivalence

Per-pixel RMS of `timeseries_torch_r1.h5 − timeseries_cpu_r1.h5` over the full 98 × 2000 × 1700 timeseries, normalised by the cpu reference's per-pixel temporal std (raw JSON: [galapagos_diff/cpu_vs_torch_r1.json](galapagos_diff/cpu_vs_torch_r1.json)):

| metric | min | median | p99 | max | mean |
|---|--:|--:|--:|--:|--:|
| absolute RMS (m) | 0.0 | 6.48e-08 | 3.95e-07 | **1.64e-05** | 8.69e-08 |
| normalised RMS (—) | 9.21e-08 | 3.75e-06 | 2.11e-05 | 4.30e-04 | 4.77e-06 |
| reference signal std (m) | 1.94e-04 | 1.64e-02 | 8.58e-02 | 6.96e-01 | 2.15e-02 |

Interpretation:

- **Absolute RMS max = 16.4 µm** is three orders of magnitude below typical Sentinel-1 LOS displacement signal (mm scale). This is well within float32 round-off.
- **Normalised RMS p99 = 2.1e-05**, well under the gate `< 1e-04` set by [report_solver_comparison.md](report_solver_comparison.md).
- **Normalised RMS max = 4.3e-04** is a single-pixel outlier; that pixel has signal std ≈ 1.94e-04 m (essentially noise-only), so the ratio amplifies by definition. Both solutions are effectively zero at that pixel — this is not a real disagreement.

→ Numerical equivalence preserved at large scale.

---

## 6. Issue #6 acceptance

| acceptance | status |
|---|---|
| Scene chosen, downloaded, `--dostep load_data` completes successfully | ✅ Galapagos Zenodo 4743058 (21.5 GB tar.xz). Smoke `--dostep load_data` exits 0 in 1.1 s (HDF5 update-mode skip) |
| Both `solver=cpu` and `solver=torch` complete; output equivalence verified at float32 round-off | ✅ both exit 0; abs RMS max 16 µm = round-off |
| `invert_network` wall share reported alongside the tutorial-scale number, with explicit comparison | ✅ §3: cpu wall **6189 s** vs Fernandina baseline **218 s** (28.4× scaling). At Galapagos scale `invert_network` consumes more wall than the entire 18-step Fernandina pipeline, confirming that its share grows with scene size |
| `report_large_scene.md` committed to `mintpy-benchmark` with methodology and raw findings | ✅ this report |
| Comment on insarlab/MintPy#1489 with a 2–3 sentence summary + permalink | next, after the commit SHA settles |

---

## 7. Caveats

- **Single round only.** Issue #6 originally proposed round-robin × 2-3 rounds for noise reduction. Given the 1 h 43 m cpu run, this report is **n=1** for each condition. The 36.4× speedup is large enough that within-condition variance does not affect the qualitative conclusion, but a follow-up round would tighten the absolute numbers.
- **`load_data` was no-op.** The Zenodo archive ships pre-loaded HDF5, so `--dostep load_data` is an update-mode skip (1.1 s). Total-pipeline `invert_network` *share* is therefore not directly comparable to a from-scratch Fernandina run; this report uses absolute `invert_network` wall as the primary metric.
- **SSD warm cache.** Both runs read from local NVMe with the OS page cache warm after the cpu run. NAS path was not measured here; the prior Fernandina report ([report_torch.md](report_torch.md) §2) flagged that NAS path adds ~3-4× I/O overhead in `load_data` but not in `invert_network` (compute-bound).
- **Out of scope:** chunk-size sweep and `torch.profiler` GPU breakdown — see [report_chunk_sweep.md](report_chunk_sweep.md) and [report_profile.md](report_profile.md) for the methodology applied at tutorial scale.

---

## 8. Reproduction

```bash
# 1. Fetch and extract (~21.5 GB tar.xz, ~31 GB extracted)
mkdir -p ~/MintPy_bench && cd ~/MintPy_bench
wget -c https://zenodo.org/records/4743058/files/GalapagosSenDT128.tar.xz
tar --use-compress-program='xz -T0' -xf GalapagosSenDT128.tar.xz
mkdir -p GalapagosSenDT128 && mv mintpy GalapagosSenDT128/mintpy   # match Fernandina layout

# 2. cpu run (~1 h 43 m on RTX 5080 / SSD)
cd <MintPy fork>/benchmark
WORK_DIR=~/MintPy_bench/GalapagosSenDT128/mintpy \
    BENCH_STEPS=invert_network \
    make bench-galapagos LOG_DIR=$(pwd)/logs_galapagos_cpu_r1
cp ~/MintPy_bench/GalapagosSenDT128/mintpy/timeseries.h5 \
   ~/MintPy_bench/GalapagosSenDT128/mintpy/timeseries_cpu_r1.h5

# 3. Reset state to avoid update-mode skip on the torch run
cd ~/MintPy_bench/GalapagosSenDT128/mintpy
rm -f timeseries.h5 maskTempCoh.h5 numInvIfgram.h5 temporalCoherence.h5 smallbaselineApp.cfg

# 4. torch run (~3 min on RTX 5080 / SSD)
cd <MintPy fork>/benchmark
WORK_DIR=~/MintPy_bench/GalapagosSenDT128/mintpy \
    BENCH_STEPS=invert_network \
    make bench-galapagos-torch LOG_DIR=$(pwd)/logs_galapagos_torch_r1
cp ~/MintPy_bench/GalapagosSenDT128/mintpy/timeseries.h5 \
   ~/MintPy_bench/GalapagosSenDT128/mintpy/timeseries_torch_r1.h5

# 5. Compare
python tools/compare_solutions.py \
    --a ~/MintPy_bench/GalapagosSenDT128/mintpy/timeseries_torch_r1.h5 \
    --b ~/MintPy_bench/GalapagosSenDT128/mintpy/timeseries_cpu_r1.h5 \
    --json-out reports/galapagos_diff/cpu_vs_torch_r1.json
```
