#!/usr/bin/env python3
"""Solver-aware torch.profiler harness for invert_network.

Variant of profile_torch.py that drives the GPU chunk loop with the
``cholesky`` *or* ``lstsq`` solver (selected via ``--solver`` or the
``MINTPY_SOLVER`` env var) and inserts ``prof.step()`` boundaries at the
right call site for that solver.

Why the separate harness:
  ``profile_torch.py`` patches ``torch.linalg.lstsq`` to mark per-chunk
  step boundaries. That works for the legacy ``lstsq`` path but is silent
  on the ``cholesky`` path, which calls ``torch.linalg.cholesky_ex`` /
  ``torch.cholesky_solve`` instead. To get a clean 1:1 step-to-chunk
  mapping for both solvers we patch the per-chunk solver dispatch
  (``mintpy.ifgram_inversion_gpu._solve_cholesky`` /
  ``_solve_lstsq``) directly -- exactly one call per chunk, by design.

OOM-safety follows profile_torch.py (see 2026-05-02 incident comment):
``schedule()`` bounds the active window, ``with_stack=False`` by default,
``ulimit -v`` capped via lib/setup_ulimit.sh.

Output layout (under --out-dir):
  tb_trace/                # Chrome trace JSON, one per active cycle
  key_averages.txt         # human-readable kernel summary
"""
import argparse
import os
import sys
from pathlib import Path

import torch
from torch.profiler import (
    ProfilerActivity,
    profile,
    schedule,
    tensorboard_trace_handler,
)


def _drive_invert_network(template_path: str, work_dir: str) -> int:
    """Run invert_network exactly as the CLI would.

    Force re-run by removing prior outputs (matches profile_torch.py).
    """
    for fname in ('timeseries.h5', 'temporalCoherence.h5', 'numInvIfgram.h5'):
        try:
            os.remove(os.path.join(work_dir, fname))
        except FileNotFoundError:
            pass

    os.chdir(work_dir)
    sys.argv = ['smallbaselineApp.py', template_path, '--dostep', 'invert_network']

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
    p.add_argument('--template', required=True)
    p.add_argument('--work-dir', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--solver',
                   choices=('cholesky', 'lstsq'),
                   default=os.environ.get('MINTPY_SOLVER'),
                   help='Override _SOLVER. Defaults to $MINTPY_SOLVER, '
                        'else the in-tree default.')
    p.add_argument('--wait', type=int, default=1)
    p.add_argument('--warmup', type=int, default=0)
    p.add_argument('--active', type=int, default=1)
    p.add_argument('--with-stack', action='store_true')
    p.add_argument('--record-shapes', action='store_true')
    p.add_argument('--profile-memory', action='store_true')
    args = p.parse_args()

    if not torch.cuda.is_available():
        print('ERROR: torch.cuda.is_available() is False', file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import mintpy.ifgram_inversion_gpu as gpu_mod

    if args.solver is not None:
        gpu_mod._SOLVER = args.solver
    active_solver = gpu_mod._SOLVER
    print(f'[profile_torch_solver] _SOLVER={active_solver}', flush=True)

    # Per-chunk step boundary by patching the active solver function.
    # estimate_timeseries_batch dispatches to exactly one of these per chunk.
    prof_ref: list = [None]
    if active_solver == 'cholesky':
        original = gpu_mod._solve_cholesky
        def patched(*a, **kw):
            result = original(*a, **kw)
            if prof_ref[0] is not None:
                prof_ref[0].step()
            return result
        gpu_mod._solve_cholesky = patched
        restore = lambda: setattr(gpu_mod, '_solve_cholesky', original)
    elif active_solver == 'lstsq':
        original = gpu_mod._solve_lstsq
        def patched(*a, **kw):
            result = original(*a, **kw)
            if prof_ref[0] is not None:
                prof_ref[0].step()
            return result
        gpu_mod._solve_lstsq = patched
        restore = lambda: setattr(gpu_mod, '_solve_lstsq', original)
    else:
        print(f'ERROR: unsupported _SOLVER={active_solver!r}', file=sys.stderr)
        return 2

    sched = schedule(
        wait=args.wait,
        warmup=args.warmup,
        active=args.active,
        repeat=1,
    )
    total_steps = args.wait + args.warmup + args.active
    print(f'[profile_torch_solver] schedule wait={args.wait} warmup={args.warmup} '
          f'active={args.active} repeat=1', flush=True)
    print(f'[profile_torch_solver] -> covers first {total_steps} chunks; trace '
          f'reflects the {args.active} active ones', flush=True)
    print(f'[profile_torch_solver] flags: with_stack={args.with_stack} '
          f'record_shapes={args.record_shapes} '
          f'profile_memory={args.profile_memory}', flush=True)

    rc = 1
    try:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=sched,
            on_trace_ready=tensorboard_trace_handler(str(out_dir)),
            record_shapes=args.record_shapes,
            with_stack=args.with_stack,
            profile_memory=args.profile_memory,
        ) as prof:
            prof_ref[0] = prof
            rc = _drive_invert_network(args.template, args.work_dir)

        try:
            table = prof.key_averages().table(
                sort_by='cuda_time_total',
                row_limit=30,
            )
            (out_dir / 'key_averages.txt').write_text(table)
            print(f'[profile_torch_solver] key_averages -> '
                  f'{out_dir / "key_averages.txt"}', flush=True)
        except (MemoryError, RuntimeError) as e:
            print(f'[profile_torch_solver] key_averages aggregation failed: '
                  f'{type(e).__name__}: {e}', flush=True)
    finally:
        restore()
        prof_ref[0] = None

    return rc


if __name__ == '__main__':
    sys.exit(main())
