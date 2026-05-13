# `correct_topography` CPU vs GPU on KujuAlosAT422F650 (ALOS / ROI_PAC, L-band)

Fifth and final scene in the Issue #19 Tier 1 bench survey, completing
the matrix across:

- **Sensor / wavelength**: ALOS PALSAR L-band (λ = 0.236 m) vs the four
  C-band Sentinel-1 scenes (λ = 0.0555 m) in the rest of the series.
- **Processor**: ROI_PAC (4th and final ingest path; the prior four
  scenes covered ISCE2/topsStack × 2, GMTSAR, and ARIA).
- **Coordinate frame**: radar coordinates (Kuju is the only scene in the
  series that ships `geometryRadar.h5` instead of `geometryGeo.h5`).
- **D axis low end**: D = 24 is the smallest in the series (Fernandina /
  Galapagos = 98, SanFranSF = 114, SanFranBay = 333).

Tracking issue: [s-sasaki-earthsea-wizard/MintPy#19](https://github.com/s-sasaki-earthsea-wizard/MintPy/issues/19)
(Tier 1 scene 3 of 3).

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
| Storage | local SSD at `~/MintPy_bench/KujuAlosAT422F650/` |

ROI_PAC inputs (`.unw` + `.rsc` header pairs, plain binary) do **not**
require GDAL, so preprocessing for Kuju can technically run from the
fork `.venv` alone. For consistency with SanFranBay + SanFranSF (which
do need GDAL), this report uses the same sibling preprocessing venv
(`~/MintPy_bench/.venv`, `--system-site-packages` + `numpy<2`); see
[`benchmark/requirements.txt`](../../requirements.txt) for setup.

Preprocessing wall: **~1 min** (smallest scene in the series; 167 ifgs
× 24 acquisitions × 226k pixels finish quickly even with the GPU torch
solver enabled in [`fixtures/KujuAlosAT422F650_torch.txt`](../../fixtures/KujuAlosAT422F650_torch.txt)).

## 2. Dataset

| | |
|---|---|
| Image shape (rows × cols) | 509 × 444 (= 225,996 pixels) |
| Acquisitions (`num_date`) | **24** |
| Interferograms in `ifgramStack.h5` | 167 |
| Time-series source | `timeseries.h5` (post-`invert_network`, no corrections) |
| Geometry file | `inputs/geometryRadar.h5` (per-pixel, radar coords) |
| Coordinate system | **radar** (lookup-table `geometryGeo.h5` exists but unused for bench) |
| Deformation model | `poly_order = 2` (3 columns), no step dates |
| Design matrix `G_0` columns (`P`) | 4 (= 1 geom + 3 polynomial) |
| Wavelength | **0.2361 m** (ALOS PALSAR L-band) |

Kuju's K is the smallest in the report series (~84% of Fernandina's
270k), and D = 24 is by far the shortest time series (Fernandina /
Galapagos D = 98 is ~4× longer; SanFranBay D = 333 is ~14×). Per-pixel
design-matrix solve cost is therefore the lowest of any scene measured.

The 4.25× longer wavelength has **no** direct effect on the GPU
dispatch arithmetic: the solver works in radians-scaled-to-metres on
both CPU and GPU, and the per-pixel `G_0 = (b_perp/(R sin θ), G_defo)`
is built the same way regardless of `λ`. Wavelength enters only at the
final `phase2range = -λ / 4π` multiplier, which is a constant across
both paths. This scene therefore tests "does the GPU code path stay
numerically clean when fed L-band-scale Δz values?" rather than
"does longer wavelength improve speedup?".

## 3. Speedup measurement (direct-call wall)

`bash scripts/run_correct_topography_bench.sh logs_correct_topo_kuju`

Single warm-cache run:

| metric | cpu | torch | ratio |
|---|--:|--:|--:|
| direct-call wall (s) | **9.14** | **3.61** | **2.53×** |
| max host RSS (MiB) | 380 | 1,200 | — |
| CUDA peak alloc (MiB) | — | 302.3 | — |
| CUDA peak reserved (MiB) | — | 351.0 | — |

Kuju delivers the highest speedup at near-Fernandina K in the series
despite D = 24 producing the smallest per-pixel design matrix. The 2.53×
matches SanFranSF (2.43×, D = 114) almost exactly — strong evidence
that, in this K regime, speedup is **K-dominated** and weakly sensitive
to D (the D-axis effect only emerges at large K where per-pixel cost
dominates over fixed framework overhead).

## 4. Numerical diff (CPU vs GPU outputs)

| output | rms / \|cpu\|.max | abs rms | \|cpu\|.max | nonzero overlap | nan |
|---|--:|--:|--:|--:|--:|
| `delta_z`  | **4.67e-8** | 5.9e-6 m | 126.2 m | 99.78% | 0 / 0 |
| `ts_cor`   | **1.41e-8** | 3.4e-8 m | 2.42 m | 95.62% | 0 / 0 |
| `ts_res`   | **2.36e-8** | 6.3e-8 m | 2.68 m | 99.78% | 0 / 0 |

All three outputs comfortably pass the **`rms / |cpu|.max < 1e-5`
numeric gate**, with `delta_z` at 4.67e-8 ≈ 200× inside the gate. The
99.78% overlap is the highest in the series — Kuju's scene mask is
clean, no large water bodies or aggressive border masking, so essentially
every pixel sees the same (CPU, GPU) treatment.

`|cpu|.max` for `delta_z` (126 m) is the cleanest in the series:
SanFranSF's 3,980 m delta_z outlier and SanFranBay's 52 m do not appear
here. ALOS L-band's longer wavelength (4.25× Sentinel-1) makes phase
unwrapping more robust per acquisition, and Kuju's modest deformation
signal stays well within DEM-error magnitudes that physical sources can
realistically produce.

## 5. VRAM + chunking

CUDA peak alloc 302.3 MiB = **1.8% of the 16 GiB RTX 5080**. The
smallest VRAM footprint of the series:

- `(K, D, P) = (226k, 24, 4)` G_batch: ~83 MiB at float32 (compare:
  SanFranBay's `(326k, 333, 4)` = 1,650 MiB, ~20× larger)
- `(K, P, P) = (226k, 4, 4)` Cholesky factor: ~14 MiB
- Intermediates for `ts_cor` / `ts_res` reconstruction: order of MB

D = 24 makes the dominant tensor (G_batch) trivially fit in any modern
GPU. The 1.8% utilisation suggests the dispatch would work even on
older / lower-VRAM cards for this scene shape.

## 6. Wavelength independence verified

CPU vs GPU `delta_z` agree to abs rms 5.9 µm = micrometre-scale, with no
NaN, no rank-deficient cpu-only / gpu-only outliers. Switching from C-
band to L-band changes nothing in the solver code path — only the
`phase2range = -λ / 4π` post-multiplier — so this confirms the GPU
dispatch is wavelength-independent within the float32 round-off budget.

## 7. Reproduction

```bash
# Sibling preprocessing venv: see benchmark/requirements.txt header.

# Dataset
mkdir -p ~/MintPy_bench/KujuAlosAT422F650
cd ~/MintPy_bench/KujuAlosAT422F650
wget -c https://zenodo.org/api/records/3952917/files/KujuAlosAT422F650.tar.xz/content -O KujuAlosAT422F650.tar.xz
tar xJf KujuAlosAT422F650.tar.xz

# Preprocess (load_data → invert_network on GPU)
cp <fork>/benchmark/fixtures/KujuAlosAT422F650_torch.txt KujuAlosAT422F650/mintpy/KujuAlosAT422F650.cfg
cd KujuAlosAT422F650/mintpy
~/MintPy_bench/.venv/bin/smallbaselineApp.py KujuAlosAT422F650.cfg --end invert_network

# Bench correct_topography (fork .venv, harness — note GEOM_FILE override
# to geometryRadar.h5 because Kuju is in radar coords, not geo)
cd <fork>
WORK_DIR=~/MintPy_bench/KujuAlosAT422F650/KujuAlosAT422F650/mintpy \
TS_FILE=$WORK_DIR/timeseries.h5 \
GEOM_FILE=$WORK_DIR/inputs/geometryRadar.h5 \
STEP_DATES="" POLY_ORDER=2 \
bash benchmark/scripts/run_correct_topography_bench.sh \
    benchmark/logs_correct_topo_kuju
```

## 8. Conclusions

- ✅ **Numeric gate `< 1e-5` holds** with ~200× margin on `delta_z`.
- ✅ **Processor independence confirmed on 4th ingest path** (ROI_PAC).
- ✅ **Wavelength independence verified** — C-band ⇄ L-band swap
  produces no numerical divergence beyond float32 round-off.
- ✅ **Speedup 2.53× at K = 226k** with D = 24 confirms the K-dominated
  speedup regime: speedup is barely changed (vs SanFranSF's 2.43× at
  K = 260k, D = 114) despite a 4.75× D ratio. D's contribution to
  speedup becomes visible only at large K where per-pixel solve cost is
  the dominant CPU term (e.g. Galapagos K = 3.4M).

Completes Tier 1. See [`report_bench_survey.md`](report_bench_survey.md)
for the combined 5-scene curve + gate validation across the series.
