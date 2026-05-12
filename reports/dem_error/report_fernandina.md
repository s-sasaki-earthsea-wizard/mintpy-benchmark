# `correct_topography` CPU vs GPU on FernandinaSenDT128

First numerical + timing measurement of the
[`mintpy.gpu.dem_error`](https://github.com/s-sasaki-earthsea-wizard/MintPy/blob/60c9e3ac/src/mintpy/gpu/dem_error.py)
batched-Cholesky solver against the production CPU per-pixel
`scipy.linalg.lstsq` loop on the tutorial scene FernandinaSenDT128
(450 × 600 = 270k pixels, 98 acquisitions, 157,667 valid pixels to
invert after masking).

Tracking issue: [s-sasaki-earthsea-wizard/MintPy#17](https://github.com/s-sasaki-earthsea-wizard/MintPy/issues/17).

Fork commit under test:
[`60c9e3ac`](https://github.com/s-sasaki-earthsea-wizard/MintPy/commit/60c9e3ac) (`perf/correct-topography-torch`).

## 1. Hardware / Environment

| | |
|---|---|
| CPU | Intel Core Ultra 9 285H, 16 cores |
| Host RAM | 93 GiB |
| GPU | NVIDIA GeForce RTX 5080 (Blackwell sm_120, 16 GiB) |
| CUDA driver / runtime | 13.1 / 13.0 (nvcc) |
| PyTorch | cu128 wheel |
| Python | 3.12.3 (uv venv) |
| Storage | NAS / CIFS (work dir, h5 inputs) |

`OMP_NUM_THREADS=1` is set by `correct_dem_error` for the CPU path (its
standard config); the GPU path does not change this.

## 2. Two measurement surfaces

The first iteration measures two complementary entry points to isolate
where the GPU win lands at this scene size:

- **(a) Direct Python call** — `scripts/run_correct_topography_bench.py`
  invokes [`mintpy.dem_error.correct_dem_error_patch`](https://github.com/s-sasaki-earthsea-wizard/MintPy/blob/60c9e3ac/src/mintpy/dem_error.py#L283)
  once with `box=None` and a `solver=` kwarg. Reads the time-series +
  geometry h5 inputs internally but performs no h5 writeback. Captures
  process wall + peak host RSS + peak CUDA VRAM + the three output
  arrays for diffing.
- **(b) Step wall via `smallbaselineApp.py`** — `scripts/run_correct_topography_step_bench.py`
  monkeypatches the dispatch (the CLI doesn't yet expose `--solver` on
  `dem_error.py`) and drives the full `--dostep correct_topography`
  step. Wall is directly comparable to the `invert_network` step-wall
  numbers in [`report_torch.md`](../report_torch.md) /
  [`report_large_scene.md`](../report_large_scene.md).

Both harnesses use `--poly-order 2` and `--step-date 20170910,20180613`,
matching the FernandinaSenDT128 production config
(`smallbaselineApp.cfg`). `mintpy.topographicResidual.pixelwiseGeometry
= auto` resolves to `yes` so we exercise the pixel-wise design-matrix
branch, which is the GPU target.

Page cache: each table reports the second consecutive run (`logs_*_r2`)
so both solvers see warm CIFS caches. Run #1 numbers are quoted only
where they expose process-startup vs MintPy-internal divergence.

## 3. (a) Direct-call wall — warm cache

`bash scripts/run_correct_topography_bench.sh
benchmark/logs_correct_topo_fernandina_r2`

| solver | wall (s) | max RSS (MiB) | peak VRAM alloc (MiB) | peak VRAM reserved (MiB) |
|---|--:|--:|--:|--:|
| cpu   | **7.45** | 657   | — | — |
| torch | **7.64** | 1,576 | 1,074.6 | 1,156.0 |

**Speedup (cpu / torch wall) = 0.97×.** At Fernandina scale the direct
call is dominated by h5 read (NAS), masking, and Python-side array
prep — the CPU per-pixel scipy loop is ~5 s of those 7.45 s and the
GPU compute itself is sub-second, but CUDA context init + cuSolver
warm-up adds back what the loop saves. This is the expected outcome
for a 158k-pixel × 98-date workload on a 16 GiB Blackwell card: peak
VRAM is 1.05 GiB out of 16 GiB and the entire fit completes in one
chunk (`auto chunk_size = 917,448 pixels`, see [`gpu/dem_error.py`](https://github.com/s-sasaki-earthsea-wizard/MintPy/blob/60c9e3ac/src/mintpy/gpu/dem_error.py#L170)).

## 4. (b) Step wall via `smallbaselineApp.py --dostep correct_topography`

`bash scripts/run_correct_topography_step_bench.sh
benchmark/logs_correct_topo_step_fernandina_r2`

Three timers nest, each measuring a different surface:

| timer | what it measures | cpu (s) | torch (s) | ratio |
|---|---|--:|--:|--:|
| process wall (`time.perf_counter` around `smallbaselineApp.main`) | full Python invocation including module-import phase | **30.70** | **11.17** | 2.75× |
| `smallbaselineApp` internal `Time used` | from `run_smallbaselineApp` entry through exit | 11.9 | 11.0 | 1.08× |
| `correct_dem_error` internal `time used` | the step body itself (template merge, box loop, patch call, h5 writeback) | 10.4 | 9.1 | 1.14× |

Process wall is dominated by Python-import cost on the CPU side
because we measure two consecutive cold-process invocations (one per
solver). MintPy fork carries enough lazy imports that the CPU process
ships `~18 s` of import + parse before `smallbaselineApp.main` even
starts; the GPU process's wall is mostly within `Time used` because
its torch imports are themselves fast once the Python module cache is
warm from the prior run. The 2.75× process-wall ratio is therefore an
artifact of the harness, not the solver: the **MintPy internal `Time
used` (1.08×)** is the apples-to-apples step number.

Where did the GPU win go? The same place as in (a): at this scale the
scipy per-pixel loop is only ~5–7 s out of the step's ~10–11 s,
because h5 read + box split + h5 writeback + smallbaselineApp
orchestration account for the rest. Replacing the loop with sub-second
GPU work shaves ~1 s off `time used`, exactly what we see.

This contrasts with `invert_network`, where the SBAS solver dominates
the step (CPU 218 s of a 230 s step on the same scene; see
[`report_torch.md`](../report_torch.md)) and the equivalent dispatch
delivered a 4.49× step-wall win.

## 5. Numerical equivalence (CPU vs GPU, real Fernandina data)

Outputs from the run #2 (a) harness. Three arrays produced by
`correct_dem_error_patch`:

| output | shape | rms | max abs diff | \|cpu\|.max | rms / \|cpu\|.max | nonzero overlap |
|---|---|--:|--:|--:|--:|--:|
| `delta_z` | (450, 600)         | 1.67e-4 m  | 5.46e-3 m  | 40.69 m | **4.10e-6** | 58.4% (≈ mask) |
| `ts_cor`  | (98, 450, 600)     | 2.52e-8 rad | 1.91e-6 rad | 0.446 rad | **5.65e-8** | 57.8% |
| `ts_res`  | (98, 450, 600)     | 5.78e-8 rad | 4.32e-6 rad | 0.111 rad | **5.23e-7** | 58.1% (+ 0.23% CPU-only nonzero) |

All deviations are at float32 round-off scale. `delta_z` rms / scale
of 4e-6 is consistent with the same metric on the synthetic test
fixture (3e-6, see Issue #17 implementation status comment) and with
`invert_network` Phase 2's `<1e-4` upstream gate.

The 0.23% of pixels where CPU's `ts_res` is non-zero but GPU's is zero
corresponds to pixels the rank-deficient detection in
[`solve_normal_equations_batched`](https://github.com/s-sasaki-earthsea-wizard/MintPy/blob/60c9e3ac/src/mintpy/gpu/_common.py)
zeroes out via the `cholesky_ex` `info != 0` mask — the CPU loop runs
scipy's `lstsq` on those pixels and gets back a noisy residual rather
than zero. The 0.015% reverse case (`gpu`-only nonzero) is consistent
with float32-comparison noise around the zero boundary on near-rank-
deficient pixels.

We deliberately do not declare a hard numerical gate here — see Issue
#17 §3 for the rationale on letting real-data diff distributions
inform the threshold rather than transplanting `<1e-4` from
`invert_network`. The above numbers comfortably fit any plausible
choice of gate.

## 6. VRAM peak

`torch.cuda.max_memory_allocated()` = **1,074.6 MiB** (~1.05 GiB) on
the GPU run.
`torch.cuda.max_memory_reserved()` = 1,156.0 MiB.

The `(K, D, P)` `G_batch` tensor with K=157,667, D=98, P=6 alone is
157,667 × 98 × 6 × 4 = ~370 MiB; the rest is the per-chunk
intermediates (G_fit, y_fit, X, ts_cor_chunk, ts_res_chunk, ts_pred,
G0_col0) — consistent with what `auto_chunk_size` budgets at
`VRAM_SAFETY=0.4`. The card holds 16 GiB so headroom is enormous at
this scale. Scale-up to Galapagos (3.4 M pixels, 475 ifgs) is where
the chunk-size auto-tuning earns its keep.

## 7. Reproducibility

```bash
# from MintPy fork root, on perf/correct-topography-torch (or later)
cd benchmark
bash scripts/run_correct_topography_bench.sh logs_correct_topo_fernandina_r2
bash scripts/run_correct_topography_step_bench.sh logs_correct_topo_step_fernandina_r2

# diff the (a) outputs (already produced by the first harness)
.venv/bin/python tools/compare_dem_error_outputs.py \
    --cpu-dir logs_correct_topo_fernandina_r2/cpu \
    --gpu-dir logs_correct_topo_fernandina_r2/torch
```

Logs (`logs_*/`) are machine-dependent and not tracked in git (see
`.gitignore`); the metrics and diff numbers above are the authoritative
record. The harness intentionally measures the second run (`_r2`) of
each script so both solvers share warm-cache state.

## 8. Conclusions and next step

1. **Numerical equivalence holds on real data.** Deviations are
   float32 round-off (rms / \|cpu\|.max ≤ 6e-6 for `delta_z` and ≤
   1e-6 for the two time-series outputs); rank-deficient pixels are
   handled consistently with `invert_network`'s policy.
2. **Wall speedup at Fernandina scale is modest** — 1.08× on
   `smallbaselineApp` internal `Time used`, 1.14× on `correct_dem_error`
   internal `time used`, 0.97× on direct-call wall. Process wall reports
   a larger ratio (2.75×) but that is an import-cost artifact and not
   informative about the solver itself. Fernandina's compute share is
   already small (~5–7 s out of ~11 s); GPU has nothing to amortise
   against beyond what already fits in a sub-second cuSolver call.
3. **VRAM is comfortable.** 1.05 GiB / 16 GiB at full Fernandina; one
   chunk handles all 158k pixels. The auto-chunk path is exercised but
   not stressed.
4. **Next bench target: GalapagosSenDT128** (3.4 M pixels, 475 ifgs;
   ~22× more pixels and a wider `R × sin θ` swath). The
   `invert_network` precedent on the same scene jumped from 4.49× to
   36.4× step-wall going Fernandina → Galapagos because the solver
   share dominates at scale. The same harness scripts parameterise
   the work dir + ts file via env vars (`WORK_DIR=…`,
   `TS_FILE=…` etc.) so the Galapagos run can drop straight in.
5. **Phase 2 structure exploit (`N_shared` row/col-scaling instead of
   the full `(K, D, P)` allocation)** remains deferred. Fernandina
   does not stress VRAM or kernel time enough to justify it; the
   decision lives or dies on Galapagos numbers and kernel-time
   breakdown.
