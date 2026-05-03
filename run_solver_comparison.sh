#!/usr/bin/env bash
# Compare _SOLVER='cholesky' vs _SOLVER='lstsq' on the FernandinaSenDT128
# fixture: 3-shot wall-time mean per solver + RMS diff of resulting
# timeseries.h5 + per-solver torch.profiler kernel breakdown.
#
# All MintPy source is left untouched: this harness uses
# run_smallbaseline_with_solver.py (which monkey-patches _SOLVER) for
# the bench shots, and profile_torch_solver.py for the profile pass.
#
# Output layout under ${LOG_DIR} (default
# benchmark/logs_solver_<timestamp>/, all untracked per .gitignore):
#   bench/<solver>/shot_<n>/
#       run.log run.time            full stdout/stderr + /usr/bin/time -v
#   bench/<solver>/timeseries.h5    backup of last shot (for RMS diff)
#   bench/summary.tsv               per-shot wall + internal time
#   profile/<solver>/               profile_torch_solver.py output
#   compare/                        compare_solutions.py output
#   _overview.log                   driver log
#   nvidia-smi.txt / free.txt / uname.txt / template.txt
set -u

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK_DIR="${WORK_DIR:-$HOME/MintPy_bench/FernandinaSenDT128/mintpy}"
TEMPLATE_BASE="${TEMPLATE:-${REPO_ROOT}/benchmark/FernandinaSenDT128_torch.txt}"
LOG_DIR="${1:-${REPO_ROOT}/benchmark/logs_solver_$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
WRAPPER="${REPO_ROOT}/benchmark/run_smallbaseline_with_solver.py"
PROFILER="${REPO_ROOT}/benchmark/profile_torch_solver.py"
COMPARER="${REPO_ROOT}/benchmark/compare_solutions.py"
SHOTS="${SHOTS:-3}"

# Cap host VA at 80% of physical RAM (see lib/setup_ulimit.sh for why)
source "$(dirname "$0")/lib/setup_ulimit.sh"

mkdir -p "${LOG_DIR}" "${LOG_DIR}/bench" "${LOG_DIR}/profile" "${LOG_DIR}/compare"
OVERVIEW="${LOG_DIR}/_overview.log"
SUMMARY="${LOG_DIR}/bench/summary.tsv"
printf 'solver\tshot\twall_seconds\tinternal_seconds\tmax_rss_kb\texit_code\n' > "${SUMMARY}"

# Env snapshot (untracked, for diagnostics)
nvidia-smi > "${LOG_DIR}/nvidia-smi.txt" 2>&1 || true
free -h    > "${LOG_DIR}/free.txt"
uname -a   > "${LOG_DIR}/uname.txt"

# Per-run template -- always emit gpuChunkSize=0 to defeat MintPy's
# update-mode skip + cfg-merge stale value (see feedback_mintpy_bench_gotchas).
TEMPLATE="${LOG_DIR}/template.txt"
cp "${TEMPLATE_BASE}" "${TEMPLATE}"
printf '\n# solver comparison: explicit auto chunk size to defeat cfg-merge stale value\nmintpy.networkInversion.gpuChunkSize = 0\n' >> "${TEMPLATE}"

{
  echo "Solver comparison started at $(date -Iseconds)"
  echo "Repo root:  ${REPO_ROOT}"
  echo "Work dir:   ${WORK_DIR}"
  echo "Template:   ${TEMPLATE}"
  echo "Log dir:    ${LOG_DIR}"
  echo "Shots/solver: ${SHOTS}"
  echo
} | tee "${OVERVIEW}"

export PYTHONUNBUFFERED=1

run_one_shot() {
    local solver="$1"
    local shot="$2"
    local shot_dir="${LOG_DIR}/bench/${solver}/shot_${shot}"
    mkdir -p "${shot_dir}"
    local log="${shot_dir}/run.log"
    local timefile="${shot_dir}/run.time"

    # Pre-clean h5 outputs: invert_network is in MintPy's update-mode skip
    # list, so leftover h5 files would silently no-op the step
    # (feedback_mintpy_bench_gotchas).
    local f
    for f in timeseries.h5 temporalCoherence.h5 numInvIfgram.h5; do
        rm -f "${WORK_DIR}/${f}"
    done

    echo "==== solver=${solver} shot=${shot} @ $(date -Iseconds) ====" \
        | tee -a "${OVERVIEW}"

    (
        cd "${WORK_DIR}"
        MINTPY_SOLVER="${solver}" /usr/bin/time -v -o "${timefile}" \
            "${PYTHON_BIN}" "${WRAPPER}" "${TEMPLATE}" \
                --dostep invert_network \
            > "${log}" 2>&1
    )
    local ec=$?

    local wall rss secs internal
    wall=$(awk -F': ' '/Elapsed \(wall clock\)/ {print $2}' "${timefile}")
    rss=$(awk -F': ' '/Maximum resident set size/ {print $2}' "${timefile}")
    secs=$(echo "$wall" | awk -F: '{
        if (NF==3) print $1*3600+$2*60+$3;
        else if (NF==2) print $1*60+$2;
        else print $1;
    }')
    internal=$(awk '/^Time used:/ {
        t = 0;
        for (i = 1; i <= NF; i++) {
            if      ($i == "mins" || $i == "min")   t += $(i-1) * 60;
            else if ($i == "secs" || $i == "sec")   t += $(i-1);
            else if ($i == "hrs"  || $i == "hr")    t += $(i-1) * 3600;
        }
        print t; exit;
    }' "${log}")
    [ -z "${internal}" ] && internal="NA"

    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${solver}" "${shot}" "${secs}" \
        "${internal}" "${rss}" "${ec}" >> "${SUMMARY}"
    echo "  -> wall=${secs}s internal=${internal}s rss=${rss}kb exit=${ec}" \
        | tee -a "${OVERVIEW}"

    return "${ec}"
}

backup_timeseries() {
    local solver="$1"
    local dst="${LOG_DIR}/bench/${solver}/timeseries.h5"
    if [ -f "${WORK_DIR}/timeseries.h5" ]; then
        cp "${WORK_DIR}/timeseries.h5" "${dst}"
        echo "  -> backed up timeseries.h5 -> ${dst}" | tee -a "${OVERVIEW}"
    else
        echo "  !! ${WORK_DIR}/timeseries.h5 missing, cannot back up" \
            | tee -a "${OVERVIEW}"
    fi
}

run_profile_one() {
    local solver="$1"
    local out_dir="${LOG_DIR}/profile/${solver}"
    local log="${out_dir}/run.log"
    local timefile="${out_dir}/run.time"
    mkdir -p "${out_dir}"

    # Pre-clean (same reason as bench shots)
    local f
    for f in timeseries.h5 temporalCoherence.h5 numInvIfgram.h5; do
        rm -f "${WORK_DIR}/${f}"
    done

    echo "==== profile solver=${solver} @ $(date -Iseconds) ====" \
        | tee -a "${OVERVIEW}"

    /usr/bin/time -v -o "${timefile}" \
        "${PYTHON_BIN}" "${PROFILER}" \
            --template "${TEMPLATE}" \
            --work-dir "${WORK_DIR}" \
            --out-dir "${out_dir}/tb_trace" \
            --solver "${solver}" \
        > "${log}" 2>&1
    local ec=$?
    echo "  -> profile exit=${ec}" | tee -a "${OVERVIEW}"
    return "${ec}"
}

# === Bench shots ===
for solver in cholesky lstsq; do
    for shot in $(seq 1 "${SHOTS}"); do
        run_one_shot "${solver}" "${shot}" || true
    done
    backup_timeseries "${solver}"
done

# === RMS comparison ===
echo "==== compare_solutions @ $(date -Iseconds) ====" | tee -a "${OVERVIEW}"
"${PYTHON_BIN}" "${COMPARER}" \
    --a "${LOG_DIR}/bench/cholesky/timeseries.h5" \
    --b "${LOG_DIR}/bench/lstsq/timeseries.h5" \
    --json-out "${LOG_DIR}/compare/rms_cholesky_vs_lstsq.json" \
    > "${LOG_DIR}/compare/summary.txt" 2>&1
COMPARE_EC=$?
echo "  -> compare exit=${COMPARE_EC}" | tee -a "${OVERVIEW}"
[ "${COMPARE_EC}" -eq 0 ] && cat "${LOG_DIR}/compare/summary.txt" | tee -a "${OVERVIEW}"

# === Profile pass ===
for solver in cholesky lstsq; do
    run_profile_one "${solver}" || true
done

echo | tee -a "${OVERVIEW}"
echo "Solver comparison finished at $(date -Iseconds)" | tee -a "${OVERVIEW}"
echo "Summary TSV:  ${SUMMARY}"                        | tee -a "${OVERVIEW}"
echo "RMS JSON:     ${LOG_DIR}/compare/rms_cholesky_vs_lstsq.json" | tee -a "${OVERVIEW}"
