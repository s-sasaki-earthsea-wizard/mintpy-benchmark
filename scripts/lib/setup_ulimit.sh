# Cap virtual address space at 80% of physical RAM, on whatever host
# this runs on. Replaces a previous hardcoded `ulimit -v 80g` that was
# sized for a 93 GiB workstation: on a smaller box the limit is unreachable
# (no protection), on a larger box it caps too aggressively.
#
# Why this safety net exists: 2026-05-02 incident, torch.profiler with
# the full chunk loop and with_stack=True grew host RSS to 94.9 GiB
# (vs. 93 GiB physical) and required a hard reboot. Capping virtual
# memory at ~80% of physical lets the kernel kill the runaway Python
# process before the system itself enters thrash.
#
# Why /proc/meminfo and not `free -h`: MemTotal is in KiB (matches
# `ulimit -v`'s default unit, no conversion), language-independent
# (free's labels can vary by locale), and present even in containers
# where `free` is missing.
#
# Why 80%: leaves headroom for kernel page cache + shell + display
# server. CUDA driver may reserve large amounts of *virtual* address
# space at init even when RSS is small, so 90%+ is safer if the
# initial 80% triggers spurious failures during CUDA init; tune
# upward in that case.

mem_kb=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)
limit_kb=$(( mem_kb * 80 / 100 ))
ulimit -v "${limit_kb}"
echo "[setup_ulimit] virtual memory capped at $((limit_kb / 1024 / 1024)) GiB (80% of $((mem_kb / 1024 / 1024)) GiB physical)"
