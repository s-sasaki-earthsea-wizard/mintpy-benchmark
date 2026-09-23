#!/usr/bin/env python3
"""Measure condition numbers of the SBAS network-inversion design matrices.

Context: the torch solver (insarlab/MintPy#1490) solves the weighted normal
equations N = Gw^T Gw in float32 via batched Cholesky. The forward error of
a normal-equation solve grows like kappa(Gw)^2 * u, and Cholesky breaks down
around kappa(Gw) ~ u^(-1/2) (~4e3 in float32). This script measures the
actual kappa values on real scenes so the fp32 headroom can be quantified,
instead of argued from worst-case bounds.

Measured quantities per scene (kept ifgrams only, dropIfgram=True):
  - kappa2(A), kappa2(B): unweighted design matrices from
    ifgramStack.get_design_matrix4timeseries (A = min-norm displacement,
    B = min-norm velocity; the MintPy default path solves B).
  - kappa2(diag(w) B) over a pixel sample, where w = weight_sqrt from
    calc_weight_sqrt(weight_func='var') -- exactly what estimate_timeseries
    multiplies into the system. Sample = random pixels + the pixels with the
    largest per-pixel weight ratio max(w)/min(w) (worst-case tail).
  - Analytic upper bound kappa2(diag(w)) * kappa2(B) using the global
    worst per-pixel weight ratio.

Usage:
  python measure_design_matrix_cond.py --workdir <mintpy_dir> \
      [--samples 2000] [--tail 100] [--seed 0] [--out result.json]

<mintpy_dir> must contain inputs/ifgramStack.h5. Requires the MintPy fork
venv (imports mintpy).
"""

import argparse
import json
import os
import time

import numpy as np

from mintpy.ifgram_inversion import calc_weight_sqrt
from mintpy.objects import ifgramStack


FP32_UNIT_ROUNDOFF = 2.0 ** -24  # u = eps/2 = 5.96e-8
FP32_BREAKDOWN_KAPPA = 1.0 / np.sqrt(FP32_UNIT_ROUNDOFF)  # ~4.1e3


def kappa2(mat):
    """2-norm condition number via SVD in float64."""
    sv = np.linalg.svd(np.asarray(mat, dtype=np.float64), compute_uv=False)
    return float(sv[0] / sv[-1])


def summarize(values):
    """Return distribution summary of an array of kappa values."""
    values = np.asarray(values)
    return {
        'min': float(values.min()),
        'median': float(np.median(values)),
        'p90': float(np.percentile(values, 90)),
        'p99': float(np.percentile(values, 99)),
        'max': float(values.max()),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--workdir', required=True,
                        help='MintPy work dir containing inputs/ifgramStack.h5')
    parser.add_argument('--samples', type=int, default=2000,
                        help='number of random pixels for per-pixel kappa (default: 2000)')
    parser.add_argument('--tail', type=int, default=100,
                        help='number of extra worst weight-ratio pixels (default: 100)')
    parser.add_argument('--seed', type=int, default=0, help='RNG seed (default: 0)')
    parser.add_argument('--out', default=None, help='write JSON result to this path')
    args = parser.parse_args()

    stack_file = os.path.join(args.workdir, 'inputs', 'ifgramStack.h5')
    stack_obj = ifgramStack(stack_file)
    stack_obj.open(print_msg=False)

    date12_list = stack_obj.get_date12_list(dropIfgram=True)
    A, B = stack_obj.get_design_matrix4timeseries(date12_list=date12_list)[0:2]
    num_pair, num_col = B.shape
    length, width = stack_obj.length, stack_obj.width
    num_pixel = length * width

    result = {
        'workdir': os.path.abspath(args.workdir),
        'num_pair_kept': int(num_pair),
        'num_date': int(num_col + 1),
        'shape': [int(length), int(width)],
        'fp32_unit_roundoff': FP32_UNIT_ROUNDOFF,
        'fp32_breakdown_kappa': float(FP32_BREAKDOWN_KAPPA),
    }

    # --- unweighted matrices ---
    result['kappa_A'] = kappa2(A)
    result['kappa_B'] = kappa2(B)
    print(f'scene: {result["workdir"]}')
    print(f'kept ifgrams: {num_pair}, dates: {num_col + 1}, pixels: {num_pixel}')
    print(f'kappa2(A) = {result["kappa_A"]:.4g}')
    print(f'kappa2(B) = {result["kappa_B"]:.4g}')

    # --- weights: same call as run_ifgram_inversion_patch (weight_func='var') ---
    t0 = time.time()
    box = (0, 0, width, length)
    weight_sqrt = calc_weight_sqrt(stack_obj, box, weight_func='var', dropIfgram=True)
    print(f'weight_sqrt computed in {time.time() - t0:.1f} s, '
          f'shape {weight_sqrt.shape}')

    # pixels with any observed coherence (proxy for invertible pixels)
    valid = weight_sqrt.max(axis=0) > 0
    num_valid = int(valid.sum())
    result['num_valid_pixel'] = num_valid
    print(f'valid pixels: {num_valid} / {num_pixel}')

    # per-pixel weight dynamic range r = max(w)/min(w); w = weight_sqrt**2
    w_sqrt_valid = weight_sqrt[:, valid]
    ratio_sqrt = w_sqrt_valid.max(axis=0) / w_sqrt_valid.min(axis=0)
    ratio = ratio_sqrt.astype(np.float64) ** 2
    result['weight_ratio'] = summarize(ratio)
    print(f'per-pixel weight ratio max(w)/min(w): {result["weight_ratio"]}')

    # analytic upper bound: kappa(diag(w_sqrt)) * kappa(B), worst pixel
    result['kappa_bound_worst'] = float(np.sqrt(ratio.max())) * result['kappa_B']
    print(f'upper bound kappa2(diag(w))*kappa2(B), worst pixel = '
          f'{result["kappa_bound_worst"]:.4g}')

    # --- sampled per-pixel kappa2(diag(w) B) ---
    rng = np.random.default_rng(args.seed)
    n_rand = min(args.samples, num_valid)
    idx_rand = rng.choice(num_valid, size=n_rand, replace=False)
    idx_tail = np.argsort(ratio)[-args.tail:]
    idx = np.unique(np.concatenate([idx_rand, idx_tail]))

    t0 = time.time()
    kappas = np.empty(idx.size)
    B64 = np.asarray(B, dtype=np.float64)
    for k, j in enumerate(idx):
        Gw = w_sqrt_valid[:, j].astype(np.float64)[:, None] * B64
        sv = np.linalg.svd(Gw, compute_uv=False)
        kappas[k] = sv[0] / sv[-1]
    print(f'{idx.size} per-pixel SVDs in {time.time() - t0:.1f} s')

    result['kappa_wB_sampled'] = summarize(kappas)
    result['num_sampled'] = int(idx.size)
    kmax = kappas.max()
    result['fp32_headroom_breakdown'] = float(FP32_BREAKDOWN_KAPPA / kmax)
    result['kappa2_u_worst'] = float(kmax ** 2 * FP32_UNIT_ROUNDOFF)
    print(f'kappa2(diag(w)B) sampled: {result["kappa_wB_sampled"]}')
    print(f'fp32 breakdown headroom (4.1e3 / kappa_max): '
          f'{result["fp32_headroom_breakdown"]:.1f}x')
    print(f'kappa_max^2 * u (fp32 normal-eq error scale): '
          f'{result["kappa2_u_worst"]:.3g}')

    if args.out:
        with open(args.out, 'w') as f:
            json.dump(result, f, indent=2)
        print(f'result written to {args.out}')


if __name__ == '__main__':
    main()
