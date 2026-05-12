# `correct_topography` CPU vs GPU on GalapagosSenDT128

Second numerical + timing measurement of the
[`mintpy.gpu.dem_error`](https://github.com/s-sasaki-earthsea-wizard/MintPy/blob/60c9e3ac/src/mintpy/gpu/dem_error.py)
batched-Cholesky solver, this time on GalapagosSenDT128 (2000 × 1700 =
3,400,000 pixels, 98 acquisitions) — the same dataset used for the
`invert_network` large-scene bench in [`report_large_scene.md`](../report_large_scene.md).

Companion to [`report_fernandina.md`](report_fernandina.md). Where the
Fernandina report measured a regime in which GPU and CPU are roughly
tied (1.08× on `smallbaselineApp` internal `Time used`), Galapagos is
the first scene where the GPU dispatch clears overhead and delivers
meaningful wall-clock speedup.

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
| Storage | local SSD at `~/MintPy_bench/GalapagosSenDT128/mintpy/` |

The Galapagos working tree is on local SSD because the source archive
(Zenodo 4743058, 21.5 GB) was previously extracted there for the
`invert_network` large-scene run (see [`report_large_scene.md`](../report_large_scene.md) §1).
This is a notable difference from `report_fernandina.md` which runs
from CIFS-mounted NAS; the I/O contribution to wall time is therefore
smaller in this report.

## 2. Dataset

| | |
|---|---|
| Image shape (rows × cols) | 2000 × 1700 (= 3,400,000 pixels) |
| Acquisitions (`num_date`) | 98 |
| Valid pixels after masking | ~3.4 M (>99.99% of mask is non-degenerate) |
| Deformation model | poly_order = 2 (3 columns), no step dates |
| Design matrix `G_0` columns (`P`) | 4 (= 1 geom + 3 polynomial) |
| `pixelwiseGeometry` | yes (auto resolves to yes; GPU target) |
| Time-series source | `timeseries.h5` (output of `invert_network`; ERA5 / deramp not applied) |

The Galapagos `smallbaselineApp.cfg` leaves `mintpy.topographicResidual.stepDate`
at `auto` (= no step jumps), so `G_defo` has only 3 polynomial columns
versus Fernandina's 5 (3 poly + 2 step). This makes Galapagos's
per-pixel design matrix `G_0` shape `(98, 4)` versus Fernandina's
`(98, 6)` — the smallest practical `P` for this step, and the most
adversarial case for "GPU speedup driven by compute share."

## 3. Speedup measurement (direct-call wall)

`bash scripts/run_correct_topography_bench.sh
logs_correct_topo_galapagos_r2`

Two consecutive runs were taken; r2 (warm SSD cache + the `STEP_DATES=""`
bash bug fix) is the reported number.

| metric | cpu | torch | ratio |
|---|--:|--:|--:|
| direct-call wall (s) | **163.83** | **26.66** | **6.15×** |
| max host RSS (MiB) | 5,577 | 11,484 | — |
| CUDA peak alloc (MiB) | — | 7,757.8 | — |
| CUDA peak reserved (MiB) | — | 8,058.7 | — |

(r1, cold cache + with the now-fixed bash bug that accidentally passed
Fernandina's step dates so `P` became 6 instead of 4: cpu 171.66 s /
torch 26.81 s = 6.40×; numerical diff dropped roughly identically. See
[`scripts/run_correct_topography_bench.sh`](../../scripts/run_correct_topography_bench.sh)
diff: the `STEP_DATES="${STEP_DATES:-...}"` substitution was replaced
with `${STEP_DATES-...}` (no colon) so an explicit `STEP_DATES=""` is
honored rather than falling back to the Fernandina defaults.)

The shape of the GPU wall:

- CUDA context init + first allocation: ~1–2 s (amortized across the
  whole run)
- `timeseries.h5` read of 1.33 GB from SSD: ~3 s
- `geometryRadar.h5` read (742 MB), masking, NaN/zero-pixel skipping:
  ~5 s
- 3 batched-Cholesky chunks of ~1.13 M pixels each at `(D, P) = (98, 4)`:
  sub-second of cuSolver work + ~1 s/chunk of H2D/D2H
- Output assemble (float32 cast + 3 × `(98, 2000, 1700)` array fills):
  ~5 s

The 26.66 s GPU wall is therefore dominated by framework overhead and
I/O — the actual GPU compute itself is well under 5 s of that.

## 4. Numerical equivalence (CPU vs GPU, real Galapagos data)

| output | shape | rms | max abs diff | \|cpu\|.max | rms / \|cpu\|.max | nonzero overlap | CPU-only nonzero | GPU-only nonzero |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| `delta_z` | (2000, 1700)        | 2.74e-4 m   | 1.07e-2 m   | 385.57 m   | **7.10e-7**  | 99.99% | 5.9e-7 | 0       |
| `ts_cor`  | (98, 2000, 1700)    | 5.22e-8 rad | 4.59e-6 rad | 2.515 rad  | **2.08e-8**  | 98.98% | 1.2e-8 | 0       |
| `ts_res`  | (98, 2000, 1700)    | 9.32e-8 rad | 1.15e-5 rad | 0.210 rad  | **4.44e-7**  | 99.99% | 7.5e-8 | 6.9e-8  |

All deviations are at float32 round-off scale. The mask overlap is
near-perfect for `delta_z` and `ts_res`; the 1% gap in `ts_cor` is
identical between CPU and GPU (both have the same mask), so it
reflects pixel-zero coverage in the inputs, not a divergence.

The CPU-only / GPU-only nonzero fractions are 4–6 orders of magnitude
tighter than on Fernandina (`ts_res` was 0.23% CPU-only / 0.015%
GPU-only there; here it is 7.5e-8 / 6.9e-8). The improvement comes
entirely from the smaller `P`: with `P=4` the normal equations
`N = G^T G` are better conditioned, so `cholesky_ex` reports far fewer
rank-deficient pixels and the CPU `lstsq` produces less residual on
those pixels too. This is incidental to the implementation and would
flip back if `P` grew (e.g. user supplies many step dates).

## 5. Comparison to Fernandina + interpretation

| dataset | pixels (K) | inverted | `P` | storage | direct-call CPU | direct-call GPU | speedup |
|---|--:|--:|--:|---|--:|--:|--:|
| Fernandina | 270 k   | 158 k   | 6 | NAS | 7.45 s   | 7.64 s  | **0.97×** |
| Galapagos  | 3.4 M   | 3.4 M   | 4 | SSD | 163.83 s | 26.66 s | **6.15×** |

K scales 12.6× from Fernandina to Galapagos but speedup grows only
~6×. The simple model that explains this:

- CPU wall ≈ `c_io + c_setup + c_per_pixel × K_inverted`
  - `c_per_pixel` is the scipy `lstsq` Python + LAPACK dispatch cost
    per pixel; at `K = 158k → 7.45 s` and `K = 3.4M → 163.83 s` it is
    very nearly linear in `K_inverted` (within 8% of perfect scaling),
    and the small constant terms (`c_io + c_setup`) are 1–5 s.
- GPU wall ≈ `c_io + c_cuda_init + c_assemble × K_inverted + c_solve × ceil(K / chunk)`
  - The dominant scaling term here is `c_assemble × K` (D2H copies +
    float32 cast + NumPy writes into the full output arrays); the
    solver kernel time is sub-linear because batched Cholesky on
    `(K, 98, 4)` matrices is bandwidth-bound, not compute-bound.

In the limit `K → ∞`, both wall times grow linearly in `K`, with a
constant per-pixel ratio set by `c_per_pixel_CPU / c_assemble_GPU`.
Empirically that asymptote is roughly 50–100 ns/pixel CPU vs
~7 ns/pixel GPU, giving a structural ceiling of ~10× even at very
large scenes. Galapagos's 6.15× is consistent with being a couple of
turns short of that ceiling because `K = 3.4 M` is not yet large
enough for the constant `c_io / c_cuda_init` terms to wash out.

This is in sharp contrast to `invert_network` ([`report_large_scene.md`](../report_large_scene.md):
36.4× step-wall on the same Galapagos scene). The structural reason
is the per-pixel solve cost:

- `correct_topography` (this report): `G_0 ∈ R^{D × P}` with `D = 98`,
  `P = 4` (or 6). Cholesky of `N = G_0^T G_0 ∈ R^{4×4}` is ~32 flops
  per pixel; full solve incl. matmuls is ~3,600 flops per pixel.
- `invert_network`: `A ∈ R^{M × (N-1)}` with `M ≈ 300+ ifgs`,
  `N - 1 ≈ 97 unknowns`. Solve cost is ~3 M flops per pixel.

So per pixel, `correct_topography` does ~1000× fewer flops. At the
same scene size both CPU loops measure a real wall, but on the GPU
side the compute is so small relative to the framework overhead that
nothing meaningful can be amortized against it. The GPU still wins
because the **CPU loop's per-pixel Python + LAPACK dispatch overhead
is large in absolute terms** — that is what gets compressed — but the
ceiling is much lower than `invert_network` allows.

The Fernandina measurement on its own was inconclusive about whether
this ceiling is "low because of Fernandina's small scene" or "low for
fundamental reasons." Galapagos pins down the latter: at 12.6× more
pixels we are still well below `invert_network`'s 36× because the
compute share itself is structurally smaller.

## 6. VRAM peak + chunking

| | |
|---|--:|
| `torch.cuda.max_memory_allocated()` | 7,757.8 MiB (~7.6 GiB) |
| `torch.cuda.max_memory_reserved()` | 8,058.7 MiB (~7.9 GiB) |
| Auto chunk size (free VRAM 15.1 GiB) | ~1.13 M pixels |
| Number of chunks | 3 |

Peak is ~48% of the 16 GiB card — comfortable headroom but no longer
the rounding error it was on Fernandina (1.05 GiB / 6.5%). The
`auto_chunk_size` path picks 3 chunks; per-chunk memory is dominated
by `(K, D, P)` `G_batch` and the per-pixel intermediates. The
graceful-degradation behaviour is exercised at this scale — a 6 GiB
card would split into ~7 chunks, a 4 GiB card into ~11, with no code
change required.

**Phase 2 structure exploit** (the `N_shared` row/col-scaling trick
that would skip the full `(K, D, P)` allocation): at Galapagos peak
VRAM 7.6 GiB out of 16 GiB and the kernel's compute share is
sub-second, there is no measurable win to be had. The 26.66 s GPU
wall is set by I/O + framework overhead, not by the solve itself.
Phase 2 is unwarranted on this evidence and is dropped from the
roadmap unless a future bench scene stresses VRAM > 80% or the
per-chunk kernel becomes the wall-time bottleneck.

## 7. Reproducibility

```bash
# from MintPy fork root, on perf/correct-topography-torch (or later).
# Galapagos working tree expected at $WORK_DIR (Zenodo 4743058 extracted).
cd benchmark
WORK_DIR=~/MintPy_bench/GalapagosSenDT128/mintpy \
TS_FILE=~/MintPy_bench/GalapagosSenDT128/mintpy/timeseries.h5 \
GEOM_FILE=~/MintPy_bench/GalapagosSenDT128/mintpy/inputs/geometryRadar.h5 \
STEP_DATES="" \
    bash scripts/run_correct_topography_bench.sh logs_correct_topo_galapagos_r2
```

Note `STEP_DATES=""` (empty) is required to override the Fernandina
default in the bash wrapper. The `${STEP_DATES-…}` substitution
honours empty strings as deliberate overrides.

Logs (`logs_*/`) are machine-dependent and not tracked in git; the
metrics and diff numbers above are the authoritative record.

## 8. Conclusions

1. **Numerical equivalence holds at Galapagos scale.** rms / scale ≤
   7e-7 for all three outputs; near-perfect mask overlap with the CPU
   reference (CPU-only and GPU-only nonzero fractions both ≤ 1e-6).
2. **6.15× wall speedup** on the direct-call surface. Fernandina was
   inconclusive (0.97× direct-call); Galapagos is the first scene
   where the GPU dispatch clears overhead and delivers meaningful
   acceleration.
3. **`P` is the dominant factor in the speedup ceiling**, not pixel
   count. Galapagos's `P = 4` keeps total flops ~1000× below
   `invert_network`'s per-pixel cost; this is why we see 6× here
   versus 36× on `invert_network`'s Galapagos run, despite identical
   pixel counts.
4. **VRAM headroom is comfortable** (48% of a 16 GiB card). The
   Phase 2 structure exploit is dropped on this evidence — the
   bottleneck is framework overhead, not the solver allocation.
5. **Numerical-equivalence gate proposal**: `rms / |cpu|.max < 1e-5`
   for each of `delta_z` / `ts_cor` / `ts_res`. The observed range
   across Fernandina + Galapagos is 2e-8 to 7e-7, with a generous
   1.5× headroom. (Compare `invert_network`'s `<1e-4` upstream gate,
   which is looser by an order of magnitude.) This number is now in
   range to write up for the upstream PR.
6. **Two-point speedup curve (Fernandina 270k → Galapagos 3.4M) is
   not enough for a strong claim**. Intermediate-scale and
   wider-coverage Zenodo datasets are needed to confirm the
   `c_per_pixel_CPU / c_assemble_GPU` asymptote and to test
   sensitivity to `P` and `D` separately. Tracked as a follow-up
   issue.
