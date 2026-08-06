# SBAS design-matrix conditioning survey (fp32 normal-equation headroom)

Date: 2026-08-06 (JST)
Subject: Measured 2-norm condition numbers of the weighted SBAS design matrix
`diag(w) B` that the torch solver of
[insarlab/MintPy#1490](https://github.com/insarlab/MintPy/pull/1490) solves via
float32 normal equations + batched Cholesky — across the same 5 scenes as
[report_end_to_end_bench.md](report_end_to_end_bench.md).
Harness: [`tools/measure_design_matrix_cond.py`](../tools/measure_design_matrix_cond.py)

> **Motivation**: a plausible review objection to PR #1490 is "normal equations
> square the condition number; why not a QR-based scheme such as CholeskyQR2
> (Fukaya et al. 2014, METR 2014-37; Yamamoto et al. 2015, ETNA 44)?".
> Worst-case bounds only bite if the actual κ is large, so this report measures
> κ on real scenes instead of arguing from bounds. Summary of the algorithmic
> comparison is in §4.

---

## TL;DR

fp32 unit roundoff u = 2⁻²⁴ ≈ 5.96e-8. Cholesky on the Gram matrix breaks
down around κ₂ ≈ u^(-1/2) ≈ **4.1e3**; normal-equation forward error scales
like κ₂²·u.

| Scene | kept ifgs × dates | κ₂(B) | κ₂(diag(w)B) median / p99 / max (sampled) | analytic bound (worst px) | breakdown headroom (4.1e3 / κ_max) |
|---|---|--:|---|--:|--:|
| FernandinaSenDT128 | 288 × 98 | 11.0 | 13.5 / 194 / **323** | 894 | **12.7×** |
| GalapagosSenDT128 | 475 × 98 | 12.7 | 13.6 / 148 / **242** | 1075 | **16.9×** |
| KujuAlosAT422F650 | 167 × 24 | 28.6 | 23.4 / 38.7 / **81.1** | 3061 | **50.5×** |
| SanFranBaySenD42 | 1297 × 333 | 55.1 | 55.1 / 200 / **721** | 1.62e4 | **5.7×** |
| SanFranSenDT42 † | 505 × 114 | 51.1 | 51.1 / 1078 / **3764** | 1.05e4 | **1.1×** (†) |

† SanFranSenDT42 is run with `weightFunc = no` (OLS) in the E2E bench
fixtures, so the system it *actually* solves is the unweighted `B` with
κ₂ = 51.1 (headroom ≈ 80×). Its diag(w)B column is a **what-if stress case**:
the κ the scene would exhibit if var-weighted. The other 4 scenes use the
MintPy default var weighting, matching their bench runs.

- **All benchmarked configurations sit comfortably inside the fp32 breakdown
  threshold**: worst sampled pixel among as-run systems is κ = 721
  (SanFranBay, headroom 5.7×); the unweighted κ₂(B) is O(10) everywhere.
- **Var weighting on a dense urban long stack is the identified stress
  direction**: the SanFranSenDT42 what-if tail reaches κ = 3764 (headroom
  1.1×), and the analytic per-pixel upper bound κ₂(diag(w))·κ₂(B) exceeds the
  threshold for both SanFran scenes. The near-threshold population is a thin
  tail of extreme weight-dynamic-range (mixed near-floor / high coherence)
  pixels — structurally the same low-temporal-coherence pixels that
  `maskTempCoh.h5` excludes downstream
  (cf. [report_end_to_end_bench.md §4.2](report_end_to_end_bench.md)).
- **Switching to CholeskyQR2 would not rescue the tail**: its applicability
  domain is the same κ₂ ≲ u^(-1/2) (the first Cholesky of the Gram matrix must
  succeed — Yamamoto et al. 2015, p.308), so it breaks down on exactly the
  same pixels. See §4.

---

## 1. Setup

| Item | Value |
|---|---|
| Machine | Intel Core Ultra 9 285H (16C) / 93 GiB RAM |
| MintPy code | fork worktree @ `297aa160` (v1.6.4); the measured code paths (`ifgramStack.get_design_matrix4timeseries`, `calc_weight_sqrt`) are upstream-identical |
| Python | 3.12.3, fork `.venv/` (uv) |
| Scenes | same work dirs as [report_end_to_end_bench.md](report_end_to_end_bench.md) (`mintpy_e2e_torch/inputs/ifgramStack.h5`; Fernandina/Galapagos from their standing work dirs) |
| Weighting | `weight_func='var'` (MintPy default `mintpy.networkInversion.weightFunc = auto`), i.e. inverse phase-variance from spatial coherence, epsilon floor 5e-2 — the exact `calc_weight_sqrt` call used by `run_ifgram_inversion_patch`. Applied to all 5 scenes for survey uniformity; note the SanFranSenDT42 E2E fixture itself sets `weightFunc = no`, so its weighted numbers are a what-if (see TL;DR †) |
| Matrix | `B` (min-norm velocity), the matrix the default `minNormVelocity = yes` path actually solves; unweighted κ₂(A) also recorded in the JSON outputs |
| Sampling | 2000 uniform-random pixels + the 100 pixels with the largest per-pixel weight dynamic range max(w)/min(w) (tail probe), SVD in float64 |
| Outputs | `logs_condition_number/<scene>.json` (untracked, machine-local) |

### Method notes

- κ₂ is computed by full SVD in float64; the fp32 solver's behaviour is then
  assessed against κ₂ via standard bounds (breakdown at κ₂ ≈ u^(-1/2),
  forward error ∝ κ₂²u), not simulated.
- The tail probe exploits κ₂(diag(w)B) ≤ κ₂(diag(w))·κ₂(B): pixels with
  extreme weight ratios are where the weighted κ can peak, so the sampled max
  is close to the scene-wide max, and the analytic bound column caps *all*
  pixels of the scene.

## 2. Results

Full numbers (min / median / p90 / p99 / max of sampled κ₂(diag(w)B), weight
ratios, κ₂(A)) are in the per-scene JSONs; headline values in the TL;DR table.
Additional observations:

- Weight dynamic range, not the network geometry, is what drives the κ tail:
  unweighted κ₂(B) is 11–55 across all five scenes, while per-pixel weight
  ratios max(w)/min(w) reach 6.6e3 (Fernandina) to 8.6e4 (SanFranBay). A pixel
  mixing near-floor coherence (epsilon 5e-2) with very high coherence rows is
  what produces κ ~ 10³.
- The p99-vs-max gap is large on the SanFran pair (p99 = 200 vs max = 721;
  p99 = 1078 vs max = 3764 in the what-if weighting): the near-threshold
  population is a thin tail, not a broad shift of the distribution.
- The Kuju / SanFranSF radar-coord divergence in the E2E report is *not*
  explained by the full-network κ measured here (Kuju max 81, headroom 50×;
  SanFranSF ran OLS at κ = 51) — see the caveat in §5: per-pixel dropping of
  invalid ifgrams shrinks the effective system and can make it rank-deficient
  outright, which is the `cholesky_ex` info-flag population, a different
  mechanism from smooth κ growth.

## 3. Error-scale sanity check vs observed diffs

Pessimistic per-pixel forward-error scale κ²u at the *sampled worst* pixel:

| Scene | κ_max² · u | Observed CPU-vs-torch product diff (rms/scale) |
|---|--:|---|
| FernandinaSenDT128 | 6.2e-3 | max 1.19e-5 (normalised, [report_solver_comparison.md](report_solver_comparison.md)); 7/7 E2E gates < 1e-5 |
| GalapagosSenDT128 | 3.5e-3 | 7/7 E2E gates < 1e-5 (max 3.48e-6) |
| SanFranBaySenD42 | 3.1e-2 | 6/6 E2E gates < 1e-5 (max 5.25e-6) |

Observed diffs sit 2–4 orders of magnitude below the worst-pixel κ²u scale:
the κ² amplification of the fp32 normal equations is nowhere near the binding
error source on the surviving-pixel population. The operative quantity is the
breakdown margin, and that is what the TL;DR table reports.

## 4. Relation to the CholeskyQR2 objection

What the two papers establish:

- CholeskyQR2 = Cholesky QR applied twice. If κ₂(X) ≲ u^(-1/2), both the
  orthogonality ‖QᵀQ − I‖ and residual are O(u) (ETNA 44, main theorem under
  assumption (2.1)).
- If κ₂(X) ≫ u^(-1/2), *"the Cholesky factorization of XᵀX can break down,
  and so does CholeskyQR2"* (ETNA 44, p.308; METR 2014-37 Fig. 4 shows the
  same breakdown wall experimentally).

Consequences for PR #1490, combined with the measurements above:

1. **Identical applicability domain.** CholeskyQR2 requires the same Gram
   Cholesky to succeed as the normal-equation solver. At the measured κ, both
   methods succeed on the same pixels and break down on the same (masked-
   downstream) tail. CholeskyQR2 widens nothing.
2. **Its benefit is orthogonality of Q, which a one-shot LS solve never
   uses.** CholeskyQR2 upgrades ‖QᵀQ − I‖ from O(κ²u) to O(u) — relevant for
   orthogonalisation workloads (block Krylov, eigensolvers), not for solving
   `min‖diag(w)(Bx − y)‖` once per pixel. For LS forward error the QR-vs-
   normal-equations gap is κu vs κ²u, and §3 shows κ²u is already slack here.
3. **Cost side**: CholeskyQR2 doubles the Gram/trsm work (2× flops of
   Cholesky QR; equal to Householder QR per METR Table 1) and its design
   motivation — communication avoidance across 16k nodes — does not apply to
   batched tiny matrices on a single GPU. The QR-based path was also already
   measured end-to-end in this repo: `torch.linalg.lstsq` vs batched Cholesky
   = 16.5× internal slowdown
   ([report_solver_comparison.md](report_solver_comparison.md)).
4. **If tail hardening is ever requested**, the targeted fixes are cheaper
   and more effective than CholeskyQR2: (a) re-solve only `cholesky_ex`-
   flagged / high-κ pixels in fp64 (doubling precision moves the breakdown
   wall from 4.1e3 to ~6.7e7), or (b) apply an rcond-truncated SVD on flagged
   pixels to match the CPU `lstsq` minimum-norm convention (also addresses
   the E2E §4.2.1 fill-convention mismatch).

## 5. Caveats

- Sampled κ uses the **full kept-ifgram network** per scene. The actual
  per-pixel system drops ifgrams with invalid phase for that pixel, so the
  effective matrix can be smaller and, for disconnected sub-networks, rank
  deficient (κ = ∞). Those pixels are the `cholesky_ex` info-flag population
  and are outside any κ-based argument — for *any* Gram-based method,
  CholeskyQR2 included.
- 2000 + 100-tail pixels per scene; the analytic bound column is the only
  claim covering all pixels.
- Weights depend on the coherence stack and the `NCORRLOOKS`/looks metadata
  of each scene; scenes reprocessed with different looks will shift the tail.

## 6. Reproduction

```bash
# fork .venv, any scene work dir containing inputs/ifgramStack.h5
../.venv/bin/python tools/measure_design_matrix_cond.py \
    --workdir <scene>/mintpy_e2e_torch \
    --out logs_condition_number/<scene>.json
```
