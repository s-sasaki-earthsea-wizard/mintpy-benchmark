# Profile of invert_network step (torch backend)

## TL;DR

py-spy profile of `--dostep invert_network` on the torch backend confirms
that the 1.43× speedup ceiling reported in [report_torch.md](report_torch.md)
is set by `torch.linalg.lstsq` itself: it accounts for ~82% of wall time.
The remaining hot regions (`read_stack_obs`, `calc_weight_sqrt`) total
< 1% on warm SSD cache and offer no useful host-side optimisation surface.

A follow-up torch.profiler run (2026-05-03, see *GPU kernel breakdown*
below) further reveals that `torch.linalg.lstsq` itself is **host
launch-overhead bound, not compute bound**: the cusolver `gels` driver
launches ~1.88 million micro-kernels per chunk (≈97 Householder
iterations × 2 support kernels × 19,403 pixels), aggregating to 18.1 s
of `cudaLaunchKernel` overhead vs. 10.4 s of actual GPU compute on a
19.9 s chunk wall. This strengthens the case for the Phase 2
normal-equation + batched-Cholesky design (issue
[#4](https://github.com/s-sasaki-earthsea-wizard/MintPy/issues/4)),
which collapses the per-chunk launch count from ~1.88 M to ~5.

## Methodology

- Harness: [`benchmark/run_profile_pyspy.sh`](run_profile_pyspy.sh)
- Backend: `mintpy.networkInversion.backend = torch`,
  `mintpy.networkInversion.gpuChunkSize = 0` (auto-resolved to 19,388 px
  per chunk → 14 chunks)
- Dataset: SSD copy of FernandinaSenDT128, 269,999 px to invert
- Sampler: py-spy 0.4.2, `--rate 100 --idle --subprocesses`, flamegraph
  format
- Launch: `python -m mintpy.cli.smallbaselineApp <template> --dostep
  invert_network` directly under py-spy (no driver script)

A complementary torch.profiler run for kernel-level breakdown is
documented separately below (*GPU kernel breakdown*); the methodology
in this section covers only the py-spy results.

## Results

### Top-level timings

| | Value |
|---|---:|
| Wall (incl. py-spy overhead, Python startup, smallbaselineApp wrapper) | 274.5 s |
| Internal (`Time used:` reported by MintPy) | 226.4 s |
| Samples collected (py-spy) | 26,491 |
| Sampler errors | 0 |

### Region breakdown (inclusive samples)

Inclusive sample counts come from the py-spy SVG `<title>` annotations.
Approx. time = `samples / total × wall`.

| Region (Python frame) | Samples | % of wall | Approx. time |
|---|--:|--:|--:|
| `run_ifgram_inversion_patch` L817 (GPU-dispatch branch) | 21,941 | 82.8% | ~227 s |
| └ `estimate_timeseries_batch` L203 (`torch.linalg.lstsq`) | 21,704 | 81.9% | ~225 s |
| `calc_weight_sqrt` (L517 + L543, inclusive) | 231 | 0.87% | ~2.4 s |
| `read_stack_obs` L405 | 43 | 0.16% | ~0.4 s |
| `mask_stack_obs` | 0 | bypass | — |
| Driver init (torch import + CUDA preload) | ~1,000 | ~3.8% | ~10 s |

`mask_stack_obs` is bypassed because the GPU-dispatch branch in
[`ifgram_inversion.py`](../src/mintpy/ifgram_inversion.py#L809-L833)
returns before the legacy per-pixel CPU loop that would call it.

## Findings

1. **GPU lstsq dominates wall time.** `torch.linalg.lstsq` runs over
   batched chunks (14 chunks of up to ~19,400 px) and accounts for ~82%
   of wall time. This is consistent with the chunk_size sweep null
   result ([report_chunk_sweep.md](report_chunk_sweep.md)): per-chunk
   launch overhead is < 0.1%, leaving the kernel itself as the binding
   factor.

2. **Pre-processing is not a bottleneck on warm SSD.** `read_stack_obs`
   (h5py I/O) and `calc_weight_sqrt` (CPU coherence → weight) together
   account for < 1% of wall time. Earlier hypotheses based on the
   cold-NAS baseline overestimated their contribution; on local SSD
   with a warm page cache neither offers a useful host-side
   optimisation surface.

3. **Driver init is one-time and small.** torch import and CUDA preload
   together total ~3.8% of wall (≈10 s), all inside Python startup.
   Not amortisable within a single `--dostep invert_network` call.

## GPU kernel breakdown (torch.profiler, 2026-05-03)

### Methodology

- Harness: [`run_profile_torch.sh`](run_profile_torch.sh) +
  [`profile_torch.py`](profile_torch.py)
- Profiler config: `schedule(wait=1, warmup=0, active=1, repeat=1)` —
  records exactly one chunk after CUDA lazy init has stabilised, with
  `tensorboard_trace_handler` flushing the active window to disk
- Flags: `with_stack=False`, `record_shapes=False`,
  `profile_memory=False` (these are the dominant host-RSS
  contributors and were the trigger of two prior OOM attempts)
- Per-chunk step boundary: `torch.linalg.lstsq` is monkey-patched so
  `prof.step()` advances 1:1 with chunks (`estimate_timeseries_batch`
  calls it exactly once per chunk)
- `ulimit -v`: 80% of physical RAM (75 GiB on this 93 GiB host) via
  [`lib/setup_ulimit.sh`](lib/setup_ulimit.sh)

`prof.key_averages().table()` aggregation overflowed Python heap
(`std::bad_alloc` at 50.7 GiB RSS) even with `active=1`: kineto
recorded 15.7 M events for one chunk (3.86 M `kernel` + 3.86 M
`cuda_runtime` + 6.1 M `ac2g` correlation links + …), and
materialising them as Python objects exceeded the C++ allocator's
reach. The harness catches the `MemoryError` and falls through; the
on-disk Chrome trace JSON (3.87 GiB) remains the authoritative
artifact and is reduced offline by
[`parse_trace.py`](parse_trace.py).

### Results — one chunk (19,403 pixels, 19.887 s wall)

#### Top-level

| | Value |
|---|--:|
| Chunk wall (`ProfilerStep#1` annotation) | 19.887 s |
| Total `kernel` events GPU time | 10.406 s |
| Total `cuda_runtime` (host-side launch) time | 18.120 s |
| Total `gpu_memcpy` time | 13.4 ms |
| Total `gpu_memset` time | 21.3 ms |
| Number of `kernel` events | 3,841,835 |
| Number of `cuda_runtime` events | 3,861,259 |

The 18.1 s of host-side launch overhead exceeds the 10.4 s of GPU
compute on a 19.9 s wall — the two overlap, but the difference
(≈9.5 s) is exactly the wall-time gap during which the GPU sits idle
waiting for the next launch. **The chunk is host-launch bound, not
compute bound.**

#### GPU kernels

| Kernel | Calls | Total | % of kernel time |
|---|--:|--:|--:|
| `ormtr_gemv_c<float, 4>` (gels back-substitution) | 1,882,091 | 3.996 s | 38.4% |
| `geqr2_smem_domino_fast<float, …, 8, 512>` (gels QR factor) | 19,403 | 3.669 s | 35.3% |
| `ormtr_gerc<float, 5, 3, 1>` (gels back-substitution) | 1,882,091 | 2.686 s | 25.8% |
| **gels QR sub-total** | | **10.351 s** | **99.4%** |
| `copy_info_kernel` | 58,209 | 42.5 ms | 0.4% |
| Memset (per-pixel zero) | 19,403 | 21.3 ms | 0.2% |
| `Memcpy HtoD` | 3 | 7.7 ms | 0.07% |
| `Memcpy DtoH` | 3 | 5.7 ms | 0.05% |
| (everything else combined) | | < 0.1 ms | < 0.01% |

Per-pixel structure: 1,882,091 ÷ 19,403 = **97.0 = `num_unknown`**
(num_date − 1). gels is processing each pixel as an *independent*
iterative QR with one Householder reflection per unknown; the two
support kernels per reflection × 97 reflections × 19,403 pixels
account for the 1.88 M-call total, plus one `geqr2_smem_domino_fast`
call per pixel for the QR factorisation itself.

#### Memory traffic

H2D + D2H + D2D combined total **13.4 ms (0.07% of chunk wall)**.
The `estimate_timeseries_batch` design that uploads `G` once and reuses
it across chunks is already saturated on this axis; no further
optimisation surface here.

#### CPU-side aten ops

`aten::linalg_lstsq` accounts for 37.87 s of inclusive CPU op time
(2 calls × 18.94 s — the warmup chunk and the active chunk). This is
inclusive of the synchronous wait for GPU completion, so it tracks
the chunk wall directly. All non-`lstsq` aten ops total < 0.7 s.

### Findings

1. **gels is launch-overhead bound, not compute bound.** Per-pixel
   iterative QR launches ~97 Householder iterations × 2 support
   kernels = ~194 micro-kernel launches per pixel. For 19,403 pixels
   this is **1.88 million launches per chunk**, each with ~5 µs of
   `cudaLaunchKernel` overhead. Aggregate: 18.1 s of `cuda_runtime`
   events vs. 10.4 s of actual GPU compute on a 19.9 s chunk.

2. **Memory traffic is not the bottleneck.** 13.4 ms of memcpy in a
   19.9 s chunk = 0.07%. The single-upload design for `G` plus
   per-chunk pageable transfers for `y` / `weight_sqrt` is already
   well-shaped for this workload.

3. **Per-pixel iteration is the cost driver.** The 1.88M-call
   structure means *any* batched solver that handles all pixels
   inside one cusolver / cuBLAS launch — not 97 launches per pixel —
   delivers a step-change in wall time, independent of the
   asymptotic flop count.

## Implications for Phase 2

- Porting `read_stack_obs` and `calc_weight_sqrt` to GPU is **not
  justified** (py-spy: < 1% of wall on warm SSD).
- The original 1.5–1.7× hypothesis in [#4](https://github.com/s-sasaki-earthsea-wizard/MintPy/issues/4)
  was based on **compute reduction** (~5.4 → ~3 Mflops/px). The
  kernel breakdown reveals a much larger opportunity in the
  **launch-overhead** dimension:

  | | gels (current) | normal-eq + Cholesky (proposed) |
  |---|---|---|
  | Per-chunk kernel launches | ~1.88 M | ~5 (1 batched GEMM + 1 batched Cholesky + 2–3 batched trsm) |
  | Per-chunk GPU compute | 10.4 s | ~6 s (theoretical, 0.6× compute) |
  | Per-chunk launch overhead | ~10 s | ≪ 1 s |
  | **Predicted chunk wall** | 19.9 s | ≈ 6 s (≈3× speedup) |

  cusolver's batched Cholesky (`cusolverDnSpotrfBatched`, exposed by
  `torch.linalg.cholesky` on a `(n, m, m)` tensor) issues *one* kernel
  for the entire batch of `n` pixels, eliminating the 1.88 M-launch
  structure entirely. Same for `torch.cholesky_solve`.

- **Recommendation:** proceed directly to issue
  [#4 step 2](https://github.com/s-sasaki-earthsea-wizard/MintPy/issues/4)
  (normal-equation PoC). Both axes (compute and launch count) point
  the same way; the original 2.2–2.5× CPU-vs-GPU target may be
  conservative.

## Limitations

- **`prof.key_averages()` does not work at this scale.** Even with
  `active=1`, kineto records ~15.7 M events for one chunk and the
  Python materialisation in `_parse_kineto_results` overflows the C++
  allocator (`std::bad_alloc` at 50.7 GiB RSS, well below the 75 GiB
  ulimit). Two earlier attempts without `schedule()` (one with
  `with_stack=True`) hit the same wall at higher RSS (94.9 GiB,
  required a hard reboot). The current harness writes the on-disk
  Chrome trace via `tensorboard_trace_handler` and reduces it offline
  with [`parse_trace.py`](parse_trace.py); `key_averages.txt` is best-
  effort and routinely empty.
- **Per-chunk profile only.** `schedule(active=1)` records exactly
  one chunk; cross-chunk variance is not measured. The chunk-size
  sweep ([report_chunk_sweep.md](report_chunk_sweep.md)) already
  showed per-chunk launch overhead is < 0.1%, so cross-chunk
  variance is expected to be small.
- **Numerical figures are machine-dependent.** Wall, RSS, the auto
  chunk size, and SSD page cache state all depend on this host.
  Reproduction requires the same harness, dataset and warm-cache
  conditions.

## Files

- py-spy harness: [`run_profile_pyspy.sh`](run_profile_pyspy.sh)
- torch.profiler harness:
  [`run_profile_torch.sh`](run_profile_torch.sh) +
  [`profile_torch.py`](profile_torch.py)
- Trace post-processor: [`parse_trace.py`](parse_trace.py)
- Shared OOM safety net: [`lib/setup_ulimit.sh`](lib/setup_ulimit.sh)
- Bench-only deps: [`requirements.txt`](requirements.txt)
- Test fixture template:
  [`FernandinaSenDT128_torch.txt`](FernandinaSenDT128_torch.txt)

Raw artifacts (`pyspy.svg`, `*.pt.trace.json`, `run.log`, `run.time`,
env snapshots) are not committed (see `.gitignore`).
