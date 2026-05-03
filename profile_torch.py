#!/usr/bin/env python3
"""Profile invert_network's GPU chunk loop with torch.profiler.

Drives `smallbaselineApp <template> --dostep invert_network` in-process
under a torch.profiler context and emits a Chrome trace + a kernel-level
key_averages summary covering the first few chunks of
``estimate_timeseries_batch``.

Companion to run_profile_pyspy.sh: where py-spy answers "which Python
lines are hot", this script answers "which CUDA kernels inside
torch.linalg.lstsq dominate, and what is the H2D / compute / D2H
balance".

OOM-safety (see 2026-05-02 incident: torch.profiler at 94.9 GiB RSS):
  * ``schedule()`` bounds the active recording window to a few chunks.
  * ``on_trace_ready=tensorboard_trace_handler`` flushes each cycle to
    disk so the host event buffer does not grow unbounded.
  * ``record_shapes`` / ``with_stack`` / ``profile_memory`` default to
    False; ``with_stack`` in particular is the dominant host-RSS cost
    and was the trigger of the prior incident.
  * Outer shell additionally caps ``ulimit -v`` via lib/setup_ulimit.sh.

Per-chunk step boundary:
  ``estimate_timeseries_batch`` calls ``torch.linalg.lstsq`` exactly
  once per chunk. We monkey-patch that single function for the duration
  of the profile context so each call ends with ``prof.step()`` --
  giving a clean 1:1 mapping between profiler steps and chunks without
  touching the source tree. The patch is reverted in finally.
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
    """Run the invert_network step exactly as the CLI would.

    Force re-run by removing prior outputs (matches run_profile_pyspy.sh
    and run_chunk_sweep.sh: gpuChunkSize is not in MintPy's update-mode
    key list, so leftover h5 files would silently skip the step).
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
    p.add_argument('--template', required=True,
                   help='Path to the MintPy template (.txt)')
    p.add_argument('--work-dir', required=True,
                   help='MintPy work directory (where h5 outputs land)')
    p.add_argument('--out-dir', required=True,
                   help='Directory to write trace JSON + key_averages.txt')
    # Default profile only one chunk: 2026-05-03 first run with active=3
    # produced 12.5 GiB trace JSON and then std::bad_alloc inside
    # key_averages() at 50.7 GiB RSS. lstsq's gels decomposes into
    # hundreds of sub-kernels per chunk, so even active=3 is millions
    # of events. One chunk is sufficient -- workload is uniform across
    # chunks once CUDA lazy init is past, which wait=1 ensures.
    p.add_argument('--wait', type=int, default=1,
                   help='Profiler steps to skip (default: 1, ensures '
                        'CUDA lazy init / kernel cache fill is past)')
    p.add_argument('--warmup', type=int, default=0,
                   help='Profiler steps to warm up (default: 0)')
    p.add_argument('--active', type=int, default=1,
                   help='Profiler steps actively recorded (default: 1)')
    p.add_argument('--with-stack', action='store_true',
                   help='Capture Python stacks. HOST-RSS HEAVY: this was '
                        'the trigger of the 2026-05-02 OOM. Use only when '
                        'you specifically need source-line attribution.')
    p.add_argument('--record-shapes', action='store_true',
                   help='Record tensor shapes (moderate host cost).')
    p.add_argument('--profile-memory', action='store_true',
                   help='Track GPU allocator events (moderate host cost).')
    args = p.parse_args()

    if not torch.cuda.is_available():
        print('ERROR: torch.cuda.is_available() is False', file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-chunk step boundary via monkey-patch. estimate_timeseries_batch
    # calls torch.linalg.lstsq once per chunk (ifgram_inversion_gpu.py L203).
    # Restoring in finally keeps the patch out of any subsequent imports.
    original_lstsq = torch.linalg.lstsq
    prof_ref: list = [None]

    def lstsq_with_step(*a, **kw):
        result = original_lstsq(*a, **kw)
        if prof_ref[0] is not None:
            prof_ref[0].step()
        return result

    torch.linalg.lstsq = lstsq_with_step

    sched = schedule(
        wait=args.wait,
        warmup=args.warmup,
        active=args.active,
        repeat=1,
    )
    total_steps = args.wait + args.warmup + args.active
    print(f'[profile_torch] schedule wait={args.wait} warmup={args.warmup} '
          f'active={args.active} repeat=1', flush=True)
    print(f'[profile_torch] -> covers first {total_steps} chunks; trace '
          f'reflects the {args.active} active ones', flush=True)
    print(f'[profile_torch] flags: with_stack={args.with_stack} '
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

        # key_averages() materialises every recorded event as a Python
        # object. With active=1 this is ~hundreds of thousands of events
        # for one lstsq call's sub-kernels; with active>=3 it overflows
        # (2026-05-03: std::bad_alloc at 50.7 GiB RSS for active=3).
        # If it fails here, the on_trace_ready trace JSON is still on
        # disk and remains the canonical artifact -- key_averages.txt
        # is a convenience, not the primary output.
        try:
            table = prof.key_averages().table(
                sort_by='cuda_time_total',
                row_limit=30,
            )
            (out_dir / 'key_averages.txt').write_text(table)
            print(f'[profile_torch] key_averages -> {out_dir / "key_averages.txt"}',
                  flush=True)
        except (MemoryError, RuntimeError) as e:
            print(f'[profile_torch] key_averages aggregation failed: '
                  f'{type(e).__name__}: {e}', flush=True)
            print(f'[profile_torch] trace JSON in {out_dir}/ remains the '
                  f'authoritative artifact', flush=True)
    finally:
        torch.linalg.lstsq = original_lstsq
        prof_ref[0] = None

    return rc


if __name__ == '__main__':
    sys.exit(main())
