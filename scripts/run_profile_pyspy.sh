#!/usr/bin/env bash
# Profile invert_network step with py-spy (Python sampling, hot path).
#
# All output under ${LOG_DIR} is untracked per benchmark/.gitignore;
# numerical findings are transcribed into ../reports/report_profile.md
# so the report itself is the canonical record.
#
# Output layout (under ${LOG_DIR}, all untracked):
#   pyspy.svg                # flamegraph
#   run.log                  # full stdout/stderr (incl. "Time used:")
#   run.time                 # /usr/bin/time -v rusage
#   nvidia-smi.txt / free.txt / top.txt / uname.txt   # env snapshot
#   template.txt             # the template used (with explicit gpuChunkSize=0)

set -u

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WORK_DIR="${WORK_DIR:-$HOME/MintPy_bench/FernandinaSenDT128/mintpy}"
TEMPLATE_BASE="${TEMPLATE:-${REPO_ROOT}/benchmark/fixtures/FernandinaSenDT128_torch.txt}"
LOG_DIR="${1:-${REPO_ROOT}/benchmark/logs_profile_pyspy_$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
PYSPY_BIN="${REPO_ROOT}/.venv/bin/py-spy"

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
# the new template does not override (see logs from chunk_sweep session).
TEMPLATE="${LOG_DIR}/template.txt"
cp "${TEMPLATE_BASE}" "${TEMPLATE}"
printf '\n# profile run: explicit auto chunk size to defeat cfg-merge stale value\nmintpy.networkInversion.gpuChunkSize = 0\n' >> "${TEMPLATE}"

cd "${WORK_DIR}"

# Force re-inversion. --dostep invert_network honours --update; gpuChunkSize
# is not in the update-mode key list, so leftover outputs would skip the step.
rm -f "${WORK_DIR}/timeseries.h5" \
      "${WORK_DIR}/temporalCoherence.h5" \
      "${WORK_DIR}/numInvIfgram.h5"

LOG="${LOG_DIR}/run.log"
TIMEFILE="${LOG_DIR}/run.time"

{
  echo "Profile run (py-spy) starting at $(date -Iseconds)"
  echo "Repo root:     ${REPO_ROOT}"
  echo "Work dir:      ${WORK_DIR}"
  echo "Template:      ${TEMPLATE}"
  echo "Log dir:       ${LOG_DIR}"
  echo
} | tee "${LOG}"

# PYTHONUNBUFFERED ensures smallbaselineApp's "Time used:" line reaches the
# log even when stdout is redirected (block-buffered otherwise).
export PYTHONUNBUFFERED=1

/usr/bin/time -v -o "${TIMEFILE}" \
    "${PYSPY_BIN}" record \
        -o "${LOG_DIR}/pyspy.svg" \
        --format flamegraph \
        --rate 100 \
        --idle \
        --subprocesses \
        -- "${PYTHON_BIN}" -m mintpy.cli.smallbaselineApp "${TEMPLATE}" --dostep invert_network \
    >> "${LOG}" 2>&1
EC=$?

{
  echo
  echo "Profile run (py-spy) finished at $(date -Iseconds), exit=${EC}"
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
