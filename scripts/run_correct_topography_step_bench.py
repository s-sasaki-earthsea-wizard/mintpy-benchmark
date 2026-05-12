#!/usr/bin/env python3
"""Step-wall bench for ``correct_topography`` via ``smallbaselineApp.py``.

Companion to ``run_correct_topography_bench.py``: where the latter
calls ``correct_dem_error_patch`` directly (= solver compute + h5 read
only), this driver runs the full ``smallbaselineApp.py --dostep
correct_topography`` orchestration including update-mode reset, box
split, and h5 writeback. The resulting wall is directly comparable to
the ``invert_network`` step-wall numbers from ``report_torch.md`` /
``report_large_scene.md``.

The MintPy CLI does not yet expose ``--solver`` on ``dem_error``
(Issue #17 defers CLI wiring until bench informs the default). To pick
the GPU path here we monkeypatch ``mintpy.dem_error.correct_dem_error_patch``
to forward a fixed ``solver`` kwarg before invoking the CLI's
``smallbaselineApp.main``. The patch is sibling-only and never leaves
this script.

Update-mode handling: leftover ``demErr.h5`` / ``timeseries_*_demErr.h5``
files cause MintPy's update mode to skip the step. We pre-clean those
to force re-run, mirroring how ``run_chunk_sweep.sh`` clears
``timeseries.h5`` before each invocation.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import resource
import sys
import time
from pathlib import Path

# Outputs the step writes; remove before re-run so update_mode doesn't skip.
# Names match mintpy.dem_error.correct_dem_error's `inps.dem_err_file` /
# `inps.ts_cor_file` defaults (demErr.h5, <ts_basename>_demErr.h5) plus the
# residual time-series (timeseriesResidual.h5) and configuration metadata
# tracking file. We only clean what production produces.
_STEP_OUTPUTS = [
    'demErr.h5',
    'timeseriesResidual.h5',
]


def _clean_step_outputs(work_dir: str, ts_basename: str) -> None:
    """Delete prior demErr outputs in ``work_dir`` so the step re-runs."""
    targets = list(_STEP_OUTPUTS)
    # Corrected time-series is named after the input ts file with `_demErr`
    # suffix, e.g. timeseries_ERA5_ramp_demErr.h5.
    targets.append(f'{ts_basename}_demErr.h5')
    for name in targets:
        path = os.path.join(work_dir, name)
        if os.path.isfile(path):
            os.remove(path)


def _patch_solver(solver: str) -> None:
    """Wrap ``correct_dem_error_patch`` so every call forwards ``solver=``.

    We rebind the symbol in the module namespace so ``correct_dem_error``
    (which references it via module-global lookup) picks up the wrapped
    version. Reverting is not needed: the driver exits at the end of
    ``smallbaselineApp.main`` so the process dies anyway.
    """
    from mintpy import dem_error as _de

    original = _de.correct_dem_error_patch

    @functools.wraps(original)
    def _wrapped(*args, **kwargs):
        kwargs['solver'] = solver
        return original(*args, **kwargs)

    _de.correct_dem_error_patch = _wrapped


def _drive_step(template_path: str, work_dir: str) -> int:
    """Invoke ``smallbaselineApp.py <template> --dostep correct_topography``.

    Mirrors what ``run_bench.sh`` would do via the CLI shim, but in-
    process so we keep the monkeypatch visible to the spawned step.
    """
    os.chdir(work_dir)
    sys.argv = ['smallbaselineApp.py', template_path,
                '--dostep', 'correct_topography']

    from mintpy.cli import smallbaselineApp
    try:
        smallbaselineApp.main()
    except SystemExit as e:
        return int(getattr(e, 'code', 0) or 0)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--template', required=True,
                   help='MintPy template path (passed to smallbaselineApp)')
    p.add_argument('--work-dir', required=True,
                   help='MintPy work dir (chdir target; outputs land here)')
    p.add_argument('--ts-basename', default='timeseries_ERA5_ramp',
                   help='Basename of the input time-series h5 (without .h5). '
                        'Used to identify the *_demErr.h5 to clean before run. '
                        'Default matches FernandinaSenDT128 standard config.')
    p.add_argument('--solver', required=True, choices=['cpu', 'torch'],
                   help="cpu = MintPy's per-pixel scipy loop. "
                        "torch = mintpy.gpu.dem_error batched Cholesky "
                        "(via monkeypatch since the CLI lacks --solver).")
    p.add_argument('--log-dir', required=True,
                   help='Where to write metrics.json. The step itself writes '
                        'demErr.h5 etc. into --work-dir as usual.')
    args = p.parse_args()

    log_dir = Path(args.log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    _clean_step_outputs(args.work_dir, args.ts_basename)

    if args.solver == 'torch':
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('--solver=torch requires CUDA but '
                               'torch.cuda.is_available() is False')
        torch.cuda.reset_peak_memory_stats()
        _patch_solver('torch')

    metrics: dict = {
        'mode': 'step',  # distinguish from direct-call bench metrics
        'solver': args.solver,
        'template': args.template,
        'work_dir': args.work_dir,
        'ts_basename': args.ts_basename,
    }

    t0 = time.perf_counter()
    rc = _drive_step(args.template, args.work_dir)
    if args.solver == 'torch':
        import torch
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    metrics['exit_code'] = rc
    metrics['wall_seconds'] = wall
    metrics['max_rss_kb'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    if args.solver == 'torch':
        import torch
        metrics['cuda_peak_alloc_bytes'] = int(torch.cuda.max_memory_allocated())
        metrics['cuda_peak_reserved_bytes'] = int(torch.cuda.max_memory_reserved())
        metrics['cuda_device_name'] = torch.cuda.get_device_name(0)

    (log_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))

    print(f'\n[step-bench] solver={args.solver}  wall={wall:.3f}s  '
          f'exit={rc}  max_rss={metrics["max_rss_kb"]/1024:.1f}MiB', flush=True)
    if args.solver == 'torch':
        print(f'[step-bench] cuda_peak_alloc='
              f'{metrics["cuda_peak_alloc_bytes"]/2**20:.1f}MiB  '
              f'device={metrics["cuda_device_name"]}', flush=True)

    return rc


if __name__ == '__main__':
    sys.exit(main())
