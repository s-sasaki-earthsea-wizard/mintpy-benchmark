#!/usr/bin/env python3
"""Diff CPU vs GPU outputs from run_correct_topography_bench.py.

Loads ``delta_z.npy`` / ``ts_cor.npy`` / ``ts_res.npy`` from two log
directories (CPU baseline and GPU candidate) and prints per-output
deviation metrics. No assertion is made — the numerical-equivalence
threshold for ``correct_topography`` is intentionally not pre-specified
(see Issue #17, planning ノート §3); this tool dumps the distribution
for human inspection so the gate can be decided post-hoc on real data.

Metrics per output:
    rms                  sqrt(mean((cpu - gpu)**2))
    max_abs_diff         max(|cpu - gpu|)
    cpu_abs_max          max(|cpu|)
    rms_over_scale       rms / cpu_abs_max  (relative to CPU's peak)
    nan_count_cpu / gpu  number of NaNs in each output
    masked_overlap       fraction of pixels where both CPU and GPU are
                         non-zero (so the diff is over the inverted set,
                         not the zero-padded mask)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _diff_metrics(cpu: np.ndarray, gpu: np.ndarray) -> dict:
    """Compute deviation metrics between two arrays of identical shape."""
    if cpu.shape != gpu.shape:
        raise ValueError(f'shape mismatch: cpu={cpu.shape} gpu={gpu.shape}')

    cpu64 = cpu.astype(np.float64, copy=False)
    gpu64 = gpu.astype(np.float64, copy=False)
    diff = cpu64 - gpu64

    finite = np.isfinite(diff)
    diff_finite = diff[finite]
    cpu_finite = cpu64[finite]

    abs_max = float(np.abs(cpu_finite).max()) if cpu_finite.size else float('nan')
    rms = float(np.sqrt(np.mean(diff_finite**2))) if diff_finite.size else float('nan')
    max_abs = float(np.abs(diff_finite).max()) if diff_finite.size else float('nan')
    rms_rel = rms / abs_max if abs_max > 0 else float('nan')

    # Pixel-level overlap: where both are non-zero, the GPU produced a
    # solution for a pixel the CPU also solved. A divergence here would
    # indicate the mask differs, which would invalidate the diff.
    nonzero_overlap = float(np.mean((cpu64 != 0) & (gpu64 != 0)))
    nonzero_cpu_only = float(np.mean((cpu64 != 0) & (gpu64 == 0)))
    nonzero_gpu_only = float(np.mean((cpu64 == 0) & (gpu64 != 0)))

    return {
        'shape': list(cpu.shape),
        'dtype_cpu': str(cpu.dtype),
        'dtype_gpu': str(gpu.dtype),
        'rms': rms,
        'max_abs_diff': max_abs,
        'cpu_abs_max': abs_max,
        'rms_over_scale': rms_rel,
        'nan_count_cpu': int(np.isnan(cpu64).sum()),
        'nan_count_gpu': int(np.isnan(gpu64).sum()),
        'nonzero_overlap_frac': nonzero_overlap,
        'cpu_only_nonzero_frac': nonzero_cpu_only,
        'gpu_only_nonzero_frac': nonzero_gpu_only,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cpu-dir', required=True, help='Log dir of the CPU bench run')
    p.add_argument('--gpu-dir', required=True, help='Log dir of the GPU bench run')
    p.add_argument('--out', default=None,
                   help='Optional path to write the diff metrics as JSON')
    args = p.parse_args()

    cpu_dir = Path(args.cpu_dir).resolve()
    gpu_dir = Path(args.gpu_dir).resolve()

    out: dict = {
        'cpu_dir': str(cpu_dir),
        'gpu_dir': str(gpu_dir),
    }
    for name in ('delta_z', 'ts_cor', 'ts_res'):
        cpu = np.load(cpu_dir / f'{name}.npy')
        gpu = np.load(gpu_dir / f'{name}.npy')
        out[name] = _diff_metrics(cpu, gpu)

    # Wall-time + VRAM headline so the diff report stays self-contained.
    for tag, d in (('cpu', cpu_dir), ('gpu', gpu_dir)):
        m_file = d / 'metrics.json'
        if m_file.is_file():
            out[f'{tag}_metrics'] = json.loads(m_file.read_text())

    print(json.dumps(out, indent=2))

    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))

    return 0


if __name__ == '__main__':
    sys.exit(main())
