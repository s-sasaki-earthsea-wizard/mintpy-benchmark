# `correct_topography` CPU vs GPU on SanFranBaySenD42 (GMTSAR)

Third numerical + timing measurement of the
[`mintpy.gpu.dem_error`](https://github.com/s-sasaki-earthsea-wizard/MintPy/blob/60c9e3ac/src/mintpy/gpu/dem_error.py)
batched-Cholesky solver, this time on SanFranBaySenD42 — a Sentinel-1
descending track 42 stack of the San Francisco Bay area processed by
**GMTSAR** (Zenodo [15814132](https://zenodo.org/records/15814132)). The
goal is to probe (a) processor independence by running the same dispatch
on a non-ISCE2 input, and (b) the **D (num_date) axis** with D = 333 vs
the previous D = 98 of Fernandina / Galapagos.

Companion to [`report_fernandina.md`](report_fernandina.md) and
[`report_galapagos.md`](report_galapagos.md). Where Fernandina sits in
the overhead-dominated regime (0.97× tie) and Galapagos rides the
compute-dominated regime (6.15×), SanFranBay falls between the two: K
is small (similar to Fernandina) but D = 333 raises per-pixel CPU cost
~3.4× and lifts the speedup off the overhead floor.

Tracking issue: [s-sasaki-earthsea-wizard/MintPy#19](https://github.com/s-sasaki-earthsea-wizard/MintPy/issues/19)
(Tier 1 scene 1 of 3 — Zenodo bench survey extending #17's Fernandina +
Galapagos pair).

Fork commit under test:
[`60c9e3ac`](https://github.com/s-sasaki-earthsea-wizard/MintPy/commit/60c9e3ac) (`perf/correct-topography-torch`).

## 1. Hardware / Environment

| | |
|---|---|
| CPU | Intel Core Ultra 9 285H, 16 cores |
| Host RAM | 93 GiB |
| GPU | NVIDIA GeForce RTX 5080 (Blackwell sm_120, 16 GiB) |
| CUDA driver / runtime | 13.1 / 13.0 (nvcc) |
| PyTorch | 2.11.0+cu128 wheel |
| Python | 3.12.3 |
| Storage | local SSD at `~/MintPy_bench/SanFranBaySenD42/` |

Two distinct venvs are used for this scene, and they are not the same as
the one used in `report_fernandina.md` / `report_galapagos.md`:

- **Preprocessing venv** (`~/MintPy_bench/.venv`, `--system-site-packages`
  + apt `python3-gdal` + `numpy<2`): runs `smallbaselineApp.py --end
  invert_network` to build `timeseries.h5` + `geometryGeo.h5` from the
  GMTSAR `.grd` stack. GDAL is required because `mintpy.prep_gmtsar`
  imports `osgeo.gdal`; the fork's main `.venv` deliberately omits GDAL
  to keep the `[gpu]` install procedure minimal (see
  [`benchmark/requirements.txt`](../../requirements.txt) header for the
  full setup steps).
- **Bench venv** (fork `.venv`, numpy 2.x + torch cu128, no GDAL): the
  `run_correct_topography_bench.sh` harness runs here. It only needs
  to read the already-built `timeseries.h5` + `geometryGeo.h5` via
  `h5py`, so GDAL is not on the critical path of the measurement.

This is the first scene in this report series that exercises the
**non-ISCE2-topsStack ingest path**. Fernandina and Galapagos were both
processed by ISCE2/topsStack and shipped phase-domain HDF5 directly; the
SanFranBay stack ships raw GMTSAR `.grd` files and required a full
`load_data → invert_network` preprocessing pass to produce a
`timeseries.h5` suitable for the bench. Preprocessing wall on this
machine was **7 min 9 s** (most of it CPU-bound `.grd → h5` conversion;
`invert_network` itself took 4 min 50 s with the [PR #8](https://github.com/s-sasaki-earthsea-wizard/MintPy/pull/8)
GPU torch solver on D = 333 / 1297 ifgs).

## 2. Dataset

| | |
|---|---|
| Image shape (rows × cols) | 413 × 790 (= 326,270 pixels) |
| Acquisitions (`num_date`) | **333** |
| Interferograms in `ifgramStack.h5` | 1,297 |
| Time-series source | `timeseries.h5` (output of `invert_network`; no tropo / ramp / DEM-err corrections applied) |
| Geometry file | `inputs/geometryGeo.h5` (per-pixel; GMTSAR ingest path) |
| Coordinate system | geo (lat/lon) |
| `pixelwiseGeometry` (tutorial cfg) | `no` — but bench overrides at patch level (see §3) |
| Deformation model | `poly_order = 2` (3 columns), no step dates |
| Design matrix `G_0` columns (`P`) | 4 (= 1 geom + 3 polynomial) |
| Wavelength | 0.0555 m (Sentinel-1 C-band) |

D = 333 is **3.4× larger** than Fernandina / Galapagos's D = 98 and is
the largest D in this report series. The per-pixel design matrix is
`(333, 4)` so the per-pixel CPU loop solves a `333 × 4` lstsq via
`(N + D × P²)` style flop count — ~5,300 flops/pixel vs Galapagos's
~1,600. The GPU path does the same work as a batched `(K, 4, 4)`
Cholesky after a `(K, 333, 4)` matmul, where the additional flops
disappear into the GPU's compute throughput.

The production cfg for this dataset (MintPy upstream
[`tests/configs/SanFranBaySenD42.txt`](https://github.com/insarlab/MintPy/blob/main/tests/configs/SanFranBaySenD42.txt))
declares `mintpy.topographicResidual.pixelwiseGeometry = no` with a
comment "fast but not the best." That tutorial choice runs the
mean-geometry branch, which is already batched (`range_dist.size == 1`)
and is **not** the GPU dispatch target. The bench harness bypasses the
cfg by calling `correct_dem_error_patch` directly with the full
per-pixel `geometryGeo.h5`, forcing the pixelwise path
(`range_dist.size > 1`) — the path GPU dispatch exists to accelerate.
The measurement therefore answers "what speedup do users get when they
opt into the pixelwise (more accurate) path on this scene?", not "what
does the SanFranBay tutorial production run actually use."

## 3. Speedup measurement (direct-call wall)

`bash scripts/run_correct_topography_bench.sh logs_correct_topo_sanfranbay`
(env: `WORK_DIR`, `TS_FILE`, `GEOM_FILE` overrides for this scene;
`STEP_DATES=""`, `POLY_ORDER=2`, no excludes — all matching the auto
defaults).

Single warm-cache run (no cold/warm split since the SSD-resident inputs
are small enough that the first read is already fast):

| metric | cpu | torch | ratio |
|---|--:|--:|--:|
| direct-call wall (s) | **14.53** | **8.23** | **1.76×** |
| max host RSS (MiB) | 1,587 | 3,231 | — |
| CUDA peak alloc (MiB) | — | 4,381.0 | — |
| CUDA peak reserved (MiB) | — | 4,239.7 | — |

GPU wall 8.23 s is dominated by framework overhead (CUDA init +
`timeseries.h5` read + per-chunk assemble), the same overhead floor seen
on Fernandina and Galapagos. CPU wall 14.53 s reflects the per-pixel
scipy `lstsq` loop on `(333, 4)` matrices × 326k pixels.

## 4. Numerical diff (CPU vs GPU outputs)

`tools/compare_dem_error_outputs.py` against the saved `.npy` arrays:

| output | rms / \|cpu\|.max | max abs diff | \|cpu\|.max | nonzero overlap | nan |
|---|--:|--:|--:|--:|--:|
| `delta_z`  | **4.33e-6** | 1.03e-3 | 52.40 m | 71.77% | 0 / 0 |
| `ts_cor`   | **6.68e-8** | 4.84e-7 | 0.318 m | 71.55% | 0 / 0 |
| `ts_res`   | **1.28e-6** | 2.25e-6 | 0.142 m | 71.77% | 0 / 0 |

The 71-72% nonzero-overlap reflects the GMTSAR water mask + invalid
pixels (28% of the scene is San Francisco Bay water + edge pixels;
both CPU and GPU paths short-circuit those to zero identically).
CPU-only / GPU-only nonzero fractions are ≤ 4e-8, well below the
rank-deficient idiom threshold.

All three outputs comfortably pass the **`rms / |cpu|.max < 1e-5`
numeric gate** proposed in
[`report_galapagos.md`](report_galapagos.md) §8, with the tightest
margin being `delta_z` at 4.33e-6 (~2.3× headroom). The gate continues
to look well-calibrated: tight enough to catch numerical regressions,
loose enough to absorb float32 round-off across all three scenes
measured so far.

## 5. VRAM + chunking

CUDA peak alloc 4,381 MiB = **27% of the 16 GiB RTX 5080**. The
harness's auto chunk size selected a single chunk (no chunking applied)
since 326k pixels × (333, 4) intermediate tensors easily fit.

The VRAM peak is dominated by:
- `(K, D, P) = (326k, 333, 4)` `G_batch` build: ~1.65 GiB at float32
- `(K, P, P) = (326k, 4, 4)` Cholesky factor: ~21 MiB
- `(D, K, P)` style intermediates for `ts_cor` / `ts_res` reconstruction

D = 333 makes `G_batch` itself ~3.4× larger than Fernandina's
`(157k, 98, 6)` (~370 MiB → 1.65 GiB), which is the headline VRAM
consequence of the longer time series. The 27% utilisation still leaves
~12 GiB headroom on a 16 GiB card.

## 6. Speedup curve so far (three scenes)

| scene | K (pixels) | D | P | CPU wall (s) | GPU wall (s) | speedup | source |
|---|--:|--:|--:|--:|--:|--:|---|
| FernandinaSenDT128 | 270,000 | 98 | 6 | 7.45 | 7.64 | **0.97×** | [`report_fernandina.md`](report_fernandina.md) |
| **SanFranBaySenD42** | **326,270** | **333** | **4** | **14.53** | **8.23** | **1.76×** | this report |
| GalapagosSenDT128 | 3,400,000 | 98 | 4 | 163.83 | 26.66 | **6.15×** | [`report_galapagos.md`](report_galapagos.md) |

K dominates the speedup ranking but is not the only factor: SanFranBay
sits at only 1.2× Fernandina's K and yet reaches 1.8× the speedup,
purely because D is 3.4× longer. This is consistent with the asymptote
model in [`report_galapagos.md`](report_galapagos.md) §2 — CPU per-pixel
cost scales as `D × P²` while GPU overhead is K-bounded and largely
D-independent, so longer D lifts CPU wall faster than GPU wall.

## 7. Processor independence

This is the first scene in the series that uses a **non-ISCE2 ingest
path**. Fernandina and Galapagos were both processed by ISCE2/topsStack
and shipped phase-domain HDF5 directly. SanFranBay starts from GMTSAR
`.grd` files and required:

1. `mintpy.prep_gmtsar.run()` → builds `ifgramStack.h5` + `geometryGeo.h5`
   from `.grd` via `osgeo.gdal` (apt `python3-gdal` 3.8.4)
2. `mintpy.invert_network` with `solver = torch` → builds `timeseries.h5`

The bench harness then reads the resulting HDF5 inputs and runs the
identical patch-level dispatch as in the previous two reports. Output
shape, units (m), bperp / slantRangeDistance / incidenceAngle semantics
are all standard MintPy regardless of upstream processor. The 1.76×
speedup observed here is therefore **GMTSAR-pipeline independent of any
ISCE2-specific quirk in Fernandina / Galapagos**.

## 8. Reproduction

```bash
# (1) Sibling preprocessing venv setup (one-off; see
#     benchmark/requirements.txt header for details).
sudo apt install python3-gdal
uv venv ~/MintPy_bench/.venv --python 3.12 --system-site-packages
uv pip install -p ~/MintPy_bench/.venv/bin/python \
    -e .[gpu] -r benchmark/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --index-strategy unsafe-best-match
uv pip install -p ~/MintPy_bench/.venv/bin/python "numpy<2"

# (2) Get the dataset
mkdir -p ~/MintPy_bench/SanFranBaySenD42
cd ~/MintPy_bench/SanFranBaySenD42
wget -c https://zenodo.org/api/records/15814132/files/SanFranBaySenD42.tar.xz/content -O SanFranBaySenD42.tar.xz
tar xJf SanFranBaySenD42.tar.xz

# (3) Preprocess (load_data → invert_network on GPU)
cp <fork>/benchmark/fixtures/SanFranBaySenD42_torch.txt SanFranBaySenD42/mintpy/SanFranBaySenD42.cfg
cd SanFranBaySenD42/mintpy
~/MintPy_bench/.venv/bin/smallbaselineApp.py SanFranBaySenD42.cfg --end invert_network

# (4) Bench correct_topography (fork .venv, harness)
cd <fork>
WORK_DIR=~/MintPy_bench/SanFranBaySenD42/SanFranBaySenD42/mintpy \
TS_FILE=$WORK_DIR/timeseries.h5 \
GEOM_FILE=$WORK_DIR/inputs/geometryGeo.h5 \
STEP_DATES="" POLY_ORDER=2 \
bash benchmark/scripts/run_correct_topography_bench.sh \
    benchmark/logs_correct_topo_sanfranbay
```

## 9. Conclusions

- ✅ **Numeric gate `< 1e-5` holds** on a third independent scene + a
  third processor pipeline (ISCE2 + ISCE2 + GMTSAR). `delta_z` at
  4.33e-6 has 2.3× headroom; tightening to `< 1e-6` would still pass
  for `ts_cor` (6.68e-8) but cut Fernandina + SanFranBay's `delta_z`
  margins to <1×. **Keep gate at `< 1e-5`** for now.
- ✅ **Processor independence confirmed** — no ISCE2-specific assumption
  in either CPU or GPU code path. Same dispatch works against GMTSAR
  output with identical numerical agreement.
- ✅ **D-axis sensitivity confirmed** — 3.4× longer D lifts speedup
  meaningfully (0.97× → 1.76×) at near-Fernandina K, validating the
  per-pixel-cost-scales-with-D part of
  [`report_galapagos.md`](report_galapagos.md) §2's asymptote model.
- 🟨 **K still dominates** — SanFranBay's K = 326k keeps total speedup
  well below the ~10× asymptote even with D = 333. The intermediate K
  axis (500k–1M) called out in [Issue #19](https://github.com/s-sasaki-earthsea-wizard/MintPy/issues/19)
  §3.1 remains effectively un-sampled by Tier 1; SanFranSF and Kuju are
  expected at <500k each.

Next: SanFranSenDT42 (ARIA, D = 114) is the second Tier 1 scene to add
a fourth ingest path; Kuju (ALOS / ROI_PAC, L-band) closes the Tier 1
set with wavelength + sensor diversity.
