#!/usr/bin/env python3
"""Extract kernel-level breakdown from a torch.profiler Chrome trace JSON.

Used when ``key_averages().table()`` fails with ``MemoryError`` on
aggregation (2026-05-03 incident: even active=1 was too many events
for the kineto -> Python materialisation step). The trace JSON is the
authoritative artifact; this script converts it into the same kernel
summary that key_averages would have produced.

Streams the JSON via stdlib only (no ijson / orjson dependency). Uses
``json.load`` since chunk-of-1 trace is bounded; if a future trace
exceeds ~10 GiB this will need a streaming parser.

Usage:
    parse_trace.py <trace.json>            -> stdout
    parse_trace.py <trace.json> <out.txt>  -> file

Reports:
  - Total events, categorised counts
  - Top GPU kernels by inclusive duration (cat == 'kernel')
  - Memcpy breakdown (HtoD / DtoH / DtoD)
  - Top CPU ops by inclusive duration (cat == 'cpu_op')
  - User annotations (ProfilerStep, region markers) with their durations
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path


def fmt_us(us: float) -> str:
    if us >= 1e6:
        return f'{us / 1e6:8.3f} s'
    if us >= 1e3:
        return f'{us / 1e3:8.2f} ms'
    return f'{us:8.1f} us'


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    trace_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    print(f'Loading {trace_path} ({trace_path.stat().st_size / 2**30:.2f} GiB)',
          flush=True)
    t0 = time.monotonic()
    with open(trace_path) as f:
        data = json.load(f)
    print(f'  parsed in {time.monotonic() - t0:.1f} s', flush=True)

    events = data.get('traceEvents', [])
    print(f'  total events: {len(events):,}', flush=True)

    # Category histogram (events with ph='X' are completed events with dur)
    cat_count: dict[str, int] = defaultdict(int)
    cat_dur_us: dict[str, float] = defaultdict(float)
    for e in events:
        cat = e.get('cat', '<none>')
        cat_count[cat] += 1
        if e.get('ph') == 'X' and 'dur' in e:
            cat_dur_us[cat] += e['dur']

    # Per-name breakdown for GPU work (kernel + memcpy + memset)
    gpu_cats = ('kernel', 'gpu_memcpy', 'gpu_memset', 'gpu_user_annotation')
    name_count_gpu: dict[str, int] = defaultdict(int)
    name_dur_us_gpu: dict[str, float] = defaultdict(float)
    for e in events:
        if e.get('cat') not in gpu_cats:
            continue
        if e.get('ph') != 'X' or 'dur' not in e:
            continue
        name_count_gpu[e['name']] += 1
        name_dur_us_gpu[e['name']] += e['dur']

    # Per-name breakdown for CPU ops
    name_count_cpu: dict[str, int] = defaultdict(int)
    name_dur_us_cpu: dict[str, float] = defaultdict(float)
    for e in events:
        if e.get('cat') != 'cpu_op':
            continue
        if e.get('ph') != 'X' or 'dur' not in e:
            continue
        name_count_cpu[e['name']] += 1
        name_dur_us_cpu[e['name']] += e['dur']

    # User annotations (e.g. ProfilerStep#1)
    user_annotations = []
    for e in events:
        if e.get('cat') == 'user_annotation' and e.get('ph') == 'X':
            user_annotations.append((e.get('name', ''), e.get('dur', 0)))

    # Memcpy direction split
    memcpy_dur_us: dict[str, float] = defaultdict(float)
    memcpy_count: dict[str, int] = defaultdict(int)
    for name, dur in name_dur_us_gpu.items():
        if 'Memcpy' not in name and 'memcpy' not in name.lower():
            continue
        for direction in ('HtoD', 'DtoH', 'DtoD', 'HtoH'):
            if direction in name:
                memcpy_dur_us[direction] += dur
                memcpy_count[direction] += name_count_gpu[name]
                break

    # ---- format report ----
    out_lines: list[str] = []
    p = out_lines.append

    p(f'# torch.profiler trace breakdown')
    p(f'')
    p(f'Source: `{trace_path.name}` ({trace_path.stat().st_size / 2**30:.2f} GiB)')
    p(f'Total events: {len(events):,}')
    p(f'')

    p(f'## Category histogram')
    p(f'')
    p(f'| Category | Events | Total dur |')
    p(f'|---|--:|--:|')
    for cat in sorted(cat_count.keys(), key=lambda c: -cat_dur_us[c]):
        p(f'| `{cat}` | {cat_count[cat]:,} | {fmt_us(cat_dur_us[cat])} |')
    p(f'')

    p(f'## User annotations (regions)')
    p(f'')
    p(f'| Name | dur |')
    p(f'|---|--:|')
    for name, dur in sorted(user_annotations, key=lambda x: -x[1]):
        p(f'| `{name}` | {fmt_us(dur)} |')
    p(f'')

    total_gpu_us = sum(name_dur_us_gpu.values())
    p(f'## Top GPU kernels / memops by inclusive duration')
    p(f'')
    p(f'Total GPU time across all `{",".join(gpu_cats)}` events: '
      f'**{fmt_us(total_gpu_us)}**')
    p(f'')
    p(f'| Rank | % | Total | Calls | Avg | Name |')
    p(f'|--:|--:|--:|--:|--:|---|')
    top = sorted(name_dur_us_gpu.items(), key=lambda x: -x[1])[:30]
    for i, (name, dur) in enumerate(top, 1):
        cnt = name_count_gpu[name]
        avg = dur / cnt if cnt else 0
        pct = dur / total_gpu_us * 100 if total_gpu_us else 0
        p(f'| {i} | {pct:.2f}% | {fmt_us(dur)} | {cnt:,} | {fmt_us(avg)} | `{name}` |')
    p(f'')

    p(f'## Memcpy direction breakdown')
    p(f'')
    p(f'| Direction | Calls | Total |')
    p(f'|---|--:|--:|')
    for d in ('HtoD', 'DtoH', 'DtoD', 'HtoH'):
        if d in memcpy_dur_us or d in memcpy_count:
            p(f'| {d} | {memcpy_count[d]:,} | {fmt_us(memcpy_dur_us[d])} |')
    p(f'')

    total_cpu_us = sum(name_dur_us_cpu.values())
    p(f'## Top CPU ops by inclusive duration')
    p(f'')
    p(f'Total CPU op time: **{fmt_us(total_cpu_us)}**')
    p(f'')
    p(f'| Rank | % | Total | Calls | Avg | Name |')
    p(f'|--:|--:|--:|--:|--:|---|')
    top_cpu = sorted(name_dur_us_cpu.items(), key=lambda x: -x[1])[:20]
    for i, (name, dur) in enumerate(top_cpu, 1):
        cnt = name_count_cpu[name]
        avg = dur / cnt if cnt else 0
        pct = dur / total_cpu_us * 100 if total_cpu_us else 0
        p(f'| {i} | {pct:.2f}% | {fmt_us(dur)} | {cnt:,} | {fmt_us(avg)} | `{name}` |')

    text = '\n'.join(out_lines) + '\n'
    if out_path is None:
        print(text)
    else:
        out_path.write_text(text)
        print(f'wrote {out_path}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
