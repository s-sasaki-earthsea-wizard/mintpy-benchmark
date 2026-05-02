# Profile of invert_network step (torch backend)

## TL;DR

py-spy profile of `--dostep invert_network` on the torch backend confirms
that the 1.43× speedup ceiling reported in [report_torch.md](report_torch.md)
is set by `torch.linalg.lstsq` itself: it accounts for ~82% of wall time.
The remaining hot regions (`read_stack_obs`, `calc_weight_sqrt`) total
< 1% on warm SSD cache and offer no useful host-side optimisation surface.
GPU-internal breakdown is left as a follow-up (issue
[#2](https://github.com/s-sasaki-earthsea-wizard/MintPy/issues/2)).

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

torch.profiler-based instrumentation was attempted twice and dropped;
see Limitations.

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

## Implications for Phase 2

- Porting `read_stack_obs` and `calc_weight_sqrt` to GPU is **not
  justified** by this profile.
- The next optimisation surface is inside `torch.linalg.lstsq` itself
  (cuSolver `gels` driver). py-spy cannot see GPU kernels, so the next
  step is either:
  - host-side `torch.cuda.Event` instrumentation around one chunk's
    H2D copy / kernel launch / sync, or
  - an external profiler such as Nsight Systems (out-of-process so
    the in-process memory failure mode below cannot recur).

## Limitations

- **No GPU-internal breakdown.** py-spy samples Python frames only; CUDA
  kernel timings are not visible. torch.profiler with
  `activities=[CUDA, CPU]` was attempted and accumulated ~95 GiB of
  in-memory profiler state across the full patch (14 chunks × thousands
  of cuSolver events) and was killed by the OOM-killer; a second
  attempt with `activities=[CPU]` only reproduced the same failure mode
  through ATen op events. Both runs reached the post-compute teardown
  before dying, indicating profiler context cleanup — not the bench
  workload — was the OOM driver. The harness now sets
  `ulimit -v 80 GiB` as a guard against runaway profile sessions.
- **Numerical figures are machine-dependent.** Wall, RSS, the auto
  chunk size, and SSD page cache state all depend on this host.
  Reproduction requires the same harness, dataset and warm-cache
  conditions.

## Files

- Harness: [`benchmark/run_profile_pyspy.sh`](run_profile_pyspy.sh)
- Bench-only deps: [`benchmark/requirements.txt`](requirements.txt)
- Test fixture template:
  [`benchmark/FernandinaSenDT128_torch.txt`](FernandinaSenDT128_torch.txt)

Raw artifacts (`pyspy.svg`, `run.log`, `run.time`, env snapshots) are
not committed (see `benchmark/.gitignore`).
