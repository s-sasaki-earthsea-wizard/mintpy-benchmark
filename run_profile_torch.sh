#!/usr/bin/env bash
# Profile invert_network step with torch.profiler (kernel-level breakdown).
#
# Companion to run_profile_pyspy.sh. py-spy answers "which Python lines
# are hot"; this script answers "which CUDA kernels inside
# torch.linalg.lstsq dominate, and what is the H2D / compute / D2H
# balance". Two profilers, two questions.
#
# OOM-safety: the prior incident (2026-05-02) was caused by
# torch.profiler with no schedule + with_stack=True. The harness
# (profile_torch.py) defaults to schedule(wait=1, warmup=1, active=3,
# repeat=1) with on_trace_ready disk flush, with_stack=False, and the
# outer ulimit guard from lib/setup_ulimit.sh.
#
# Output layout (under ${LOG_DIR}, all untracked per .gitignore):
#   tb_trace/                # Chrome trace JSON, one per active cycle
#   key_averages.txt         # human-readable kernel summary
#   run.log                  # full stdout/stderr (incl. "Time used:")
#   run.time                 # /usr/bin/time -v rusage
#   nvidia-smi.txt / free.txt / top.txt / uname.txt   # env snapshot
#   template.txt             # the template used (with explicit gpuChunkSize=0)

set -u

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK_DIR="${WORK_DIR:-$HOME/MintPy_bench/FernandinaSenDT128/mintpy}"
TEMPLATE_BASE="${TEMPLATE:-${REPO_ROOT}/benchmark/FernandinaSenDT128_torch.txt}"
LOG_DIR="${1:-${REPO_ROOT}/benchmark/logs_profile_torch_$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
HARNESS="${REPO_ROOT}/benchmark/profile_torch.py"

# Cap host VA at 80% of physical RAM (see lib/setup_ulimit.sh for why)
source "$(dirname "$0")/lib/setup_ulimit.sh"

mkdir -p "${LOG_DIR}"

# env snapshot (untracked, kept locally for diagnostics)
nvidia-smi > "${LOG_DIR}/nvidia-smi.txt" 2>&1 || true
free -h    > "${LOG_DIR}/free.txt"
top -bn1 -o %CPU | head -25 > "${LOG_DIR}/top.txt"
uname -a   > "${LOG_DIR}/uname.txt"

# Per-run template. Always emit gpuChunkSize even for the auto case:
# MintPy preserves stale values in work_dir/smallbaselineApp.cfg when
# the new template does not override (matches run_profile_pyspy.sh).
TEMPLATE="${LOG_DIR}/template.txt"
cp "${TEMPLATE_BASE}" "${TEMPLATE}"
printf '\n# profile run: explicit auto chunk size to defeat cfg-merge stale value\nmintpy.networkInversion.gpuChunkSize = 0\n' >> "${TEMPLATE}"

LOG="${LOG_DIR}/run.log"
TIMEFILE="${LOG_DIR}/run.time"

{
  echo "Profile run (torch.profiler) starting at $(date -Iseconds)"
  echo "Repo root:     ${REPO_ROOT}"
  echo "Work dir:      ${WORK_DIR}"
  echo "Template:      ${TEMPLATE}"
  echo "Log dir:       ${LOG_DIR}"
  echo
} | tee "${LOG}"

# PYTHONUNBUFFERED ensures smallbaselineApp's "Time used:" line reaches
# the log even when stdout is redirected (block-buffered otherwise).
export PYTHONUNBUFFERED=1

/usr/bin/time -v -o "${TIMEFILE}" \
    "${PYTHON_BIN}" "${HARNESS}" \
        --template "${TEMPLATE}" \
        --work-dir "${WORK_DIR}" \
        --out-dir "${LOG_DIR}/tb_trace" \
    >> "${LOG}" 2>&1
EC=$?

{
  echo
  echo "Profile run (torch.profiler) finished at $(date -Iseconds), exit=${EC}"
} | tee -a "${LOG}"

WALL=$(awk -F': ' '/Elapsed \(wall clock\)/ {print $2}' "${TIMEFILE}")
RSS=$(awk -F': ' '/Maximum resident set size/ {print $2}' "${TIMEFILE}")
echo "Wall:  ${WALL}"   | tee -a "${LOG}"
echo "RSS:   ${RSS} KB" | tee -a "${LOG}"

INTERNAL=$(awk '/^Time used:/ {
    t = 0;
    for (i = 1; i <= NF; i++) {
        if      ($i == "mins" || $i == "min")   t += $(i-1) * 60;
        else if ($i == "secs" || $i == "sec")   t += $(i-1);
        else if ($i == "hrs"  || $i == "hr")    t += $(i-1) * 3600;
    }
    print t; exit;
}' "${LOG}")
[ -z "${INTERNAL}" ] && INTERNAL="NA"
echo "Internal: ${INTERNAL} s" | tee -a "${LOG}"

exit "${EC}"
