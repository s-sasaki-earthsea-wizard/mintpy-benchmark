#!/usr/bin/env bash
# End-to-end smallbaselineApp.py CPU vs torch bench (Issue #21).
#
# For one scene, runs the full 18-step smallbaselineApp pipeline twice:
#   1) with the upstream template (defaults → cpu solver for both
#      invert_network and correct_topography)
#   2) with the fixtures/<scene>_torch.txt template, which flips both
#      mintpy.networkInversion.solver = torch and
#      mintpy.topographicResidual.solver = torch
# and then diffs:
#   - step wall times    (compare_step_walls.py)
#   - 7 HDF5 products    (compare_h5_outputs.py)
#
# Workdir strategy (per Issue #21 design discussion 2026-05-17):
# cpu and torch get *separate* workdirs under the scene root so that
# both runs' products remain on disk for re-diffing, and so that the
# update-mode skip trap (see memory feedback_mintpy_bench_gotchas.md)
# is structurally avoided. Large raw inputs (merged/, geometry/) are
# shared via the templates' relative ../path references — only the
# MintPy workdir output is duplicated, which is ~100MB-2GB per scene.
#
# ERA5 cache (when the scene's template uses pyaps tropo correction):
# if a pre-existing ERA5.h5 is reachable via the scene's ERA5_CACHE
# path, both workdirs symlink it in Phase 0 to make the cpu and torch
# correct_troposphere walls comparable (otherwise the first run pays
# the ~70 min CDS download as part of its step wall and the second
# does not, breaking the ±5% regression control). If the cache is
# absent, the cpu run downloads it inline and the torch run picks it
# up via post-cpu-run symlink.
#
# Re-run policy: default is to wipe the workdirs at the start so the
# bench is idempotent. Set FRESH=0 to keep existing products (useful
# during harness development, not for headline numbers).
set -u

source "$(dirname "$0")/lib/setup_ulimit.sh"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BENCH_ROOT="${REPO_ROOT}/benchmark"

usage() {
    cat <<EOF >&2
usage: $0 <scene>

  scene: one of
    FernandinaSenDT128    GalapagosSenDT128    KujuAlosAT422F650
    SanFranSenDT42        SanFranBaySenD42

env overrides:
  PYTHON_BIN     (default: \$HOME/MintPy_bench/.venv/bin/python — the
                  sibling venv used by Issue #19 Tier 1 bench; needed
                  for ARIA/GMTSAR ingest paths that require GDAL)
  SBA_BIN        (default: \$HOME/MintPy_bench/.venv/bin/smallbaselineApp.py)
  FRESH          1 = wipe workdirs before runs (default), 0 = keep
EOF
    exit "${1:-2}"
}

SCENE="${1:-}"
[[ -z "${SCENE}" ]] && usage

PYTHON_BIN="${PYTHON_BIN:-${HOME}/MintPy_bench/.venv/bin/python}"
SBA_BIN="${SBA_BIN:-${HOME}/MintPy_bench/.venv/bin/smallbaselineApp.py}"
FRESH="${FRESH:-1}"

# Per-scene metadata. NEEDS_ERA5 = 1 means the template's
# troposphericDelay.method defaults to pyaps and will pull ERA5;
# 0 means the template explicitly sets `... = no`. ERA5_CACHE is a
# pre-existing ERA5.h5 to symlink (may not exist on first run).
case "${SCENE}" in
    FernandinaSenDT128)
        # SSD-resident layout (warm-SSD per Issue #21 acceptance
        # criteria). The in-repo NAS-mounted copy is left untouched
        # for upstream/tutorial code paths that read it; only this
        # bench's workdirs and ERA5 cache live on the SSD sibling tree.
        SCENE_ROOT="${HOME}/MintPy_bench/FernandinaSenDT128"
        CPU_TEMPLATE="${REPO_ROOT}/docs/templates/FernandinaSenDT128.txt"
        NEEDS_ERA5=1
        ERA5_CACHE="${HOME}/MintPy_bench/FernandinaSenDT128/mintpy/inputs/ERA5.h5"
        ;;
    GalapagosSenDT128)
        SCENE_ROOT="${HOME}/MintPy_bench/GalapagosSenDT128"
        CPU_TEMPLATE="${HOME}/MintPy_bench/GalapagosSenDT128/GalapagosSenDT128.template"
        NEEDS_ERA5=1
        ERA5_CACHE="${HOME}/MintPy_bench/GalapagosSenDT128/mintpy/inputs/ERA5.h5"
        ;;
    KujuAlosAT422F650)
        SCENE_ROOT="${HOME}/MintPy_bench/KujuAlosAT422F650"
        CPU_TEMPLATE="${REPO_ROOT}/docs/templates/KujuAlosAT422F650.txt"
        NEEDS_ERA5=0
        ERA5_CACHE=""
        ;;
    SanFranSenDT42)
        SCENE_ROOT="${HOME}/MintPy_bench/SanFranSenDT42"
        CPU_TEMPLATE="${REPO_ROOT}/docs/templates/SanFranSenDT42.txt"
        NEEDS_ERA5=1
        ERA5_CACHE="${HOME}/MintPy_bench/SanFranSenDT42/mintpy/inputs/ERA5.h5"
        ;;
    SanFranBaySenD42)
        SCENE_ROOT="${HOME}/MintPy_bench/SanFranBaySenD42"
        CPU_TEMPLATE="${REPO_ROOT}/docs/templates/SanFranBaySenD42.txt"
        NEEDS_ERA5=0
        ERA5_CACHE=""
        ;;
    -h|--help)
        usage 0
        ;;
    *)
        echo "Unknown scene: ${SCENE}" >&2
        usage
        ;;
esac

TORCH_TEMPLATE="${BENCH_ROOT}/fixtures/${SCENE}_torch.txt"
WORK_DIR_CPU="${SCENE_ROOT}/mintpy_e2e_cpu"
WORK_DIR_TORCH="${SCENE_ROOT}/mintpy_e2e_torch"
LOG_DIR_CPU="${BENCH_ROOT}/logs_e2e_${SCENE}_cpu"
LOG_DIR_TORCH="${BENCH_ROOT}/logs_e2e_${SCENE}_torch"
WALLS_OUT="${BENCH_ROOT}/logs_e2e_${SCENE}_walls_diff.json"
PRODUCTS_OUT="${BENCH_ROOT}/logs_e2e_${SCENE}_products_diff.json"

# Pre-flight checks.
for path in "${CPU_TEMPLATE}" "${TORCH_TEMPLATE}" "${PYTHON_BIN}" "${SBA_BIN}"; do
    if [[ ! -e "${path}" ]]; then
        echo "ERROR: required path missing: ${path}" >&2
        exit 3
    fi
done
if [[ ! -d "${SCENE_ROOT}" ]]; then
    echo "ERROR: SCENE_ROOT does not exist: ${SCENE_ROOT}" >&2
    echo "       (raw data must be unpacked here before running this bench)" >&2
    exit 3
fi

echo "==== End-to-end bench: ${SCENE} ===="
echo "  scene root:     ${SCENE_ROOT}"
echo "  cpu template:   ${CPU_TEMPLATE}"
echo "  torch template: ${TORCH_TEMPLATE}"
echo "  cpu workdir:    ${WORK_DIR_CPU}"
echo "  torch workdir:  ${WORK_DIR_TORCH}"
echo "  python:         ${PYTHON_BIN}"
echo "  ERA5 needed:    ${NEEDS_ERA5} (cache: ${ERA5_CACHE:-<none>})"
echo "  FRESH:          ${FRESH}"

# ---- Phase 0: workdir setup ----
if [[ "${FRESH}" -eq 1 ]]; then
    echo "==== [Phase 0] Wiping workdirs (FRESH=1) ===="
    rm -rf "${WORK_DIR_CPU}" "${WORK_DIR_TORCH}"
fi
mkdir -p "${WORK_DIR_CPU}/inputs" "${WORK_DIR_TORCH}/inputs"

# Pre-seed ERA5 cache (best-effort): if a previous run's ERA5.h5 is
# reachable, symlink it into both workdirs so the cpu and torch
# correct_troposphere steps both reuse it and remain comparable.
# Otherwise the cpu run downloads it inline (visible in its step wall)
# and the torch run gets a post-cpu-run symlink before its run starts.
if [[ "${NEEDS_ERA5}" -eq 1 && -n "${ERA5_CACHE}" && -f "${ERA5_CACHE}" ]]; then
    ln -sf "${ERA5_CACHE}" "${WORK_DIR_CPU}/inputs/ERA5.h5"
    ln -sf "${ERA5_CACHE}" "${WORK_DIR_TORCH}/inputs/ERA5.h5"
    echo "[Phase 0] Pre-seeded ERA5.h5 in both workdirs from ${ERA5_CACHE}"
fi

# ---- Phase 1: CPU run ----
echo
echo "==== [Phase 1] CPU run ===="
TEMPLATE="${CPU_TEMPLATE}" WORK_DIR="${WORK_DIR_CPU}" \
    PYTHON_BIN="${PYTHON_BIN}" SBA_BIN="${SBA_BIN}" \
    bash "${BENCH_ROOT}/scripts/run_bench.sh" "${LOG_DIR_CPU}"
CPU_EC=$?

# ---- Phase 2: ERA5 cache symlink for torch run ----
# If ERA5 was downloaded by the cpu run (i.e. wasn't already cached
# in Phase 0), wire it into the torch workdir now so the torch run's
# correct_troposphere step doesn't re-download.
if [[ "${NEEDS_ERA5}" -eq 1 && -f "${WORK_DIR_CPU}/inputs/ERA5.h5" \
      && ! -e "${WORK_DIR_TORCH}/inputs/ERA5.h5" ]]; then
    ln -sf "${WORK_DIR_CPU}/inputs/ERA5.h5" "${WORK_DIR_TORCH}/inputs/ERA5.h5"
    echo "[Phase 2] Linked ERA5.h5 from cpu workdir into torch workdir"
fi

# ---- Phase 3: Torch run ----
echo
echo "==== [Phase 3] Torch run ===="
TEMPLATE="${TORCH_TEMPLATE}" WORK_DIR="${WORK_DIR_TORCH}" \
    PYTHON_BIN="${PYTHON_BIN}" SBA_BIN="${SBA_BIN}" \
    bash "${BENCH_ROOT}/scripts/run_bench.sh" "${LOG_DIR_TORCH}"
TORCH_EC=$?

# ---- Phase 4: Step wall diff ----
echo
echo "==== [Phase 4] Step wall diff ===="
"${PYTHON_BIN}" "${BENCH_ROOT}/tools/compare_step_walls.py" \
    --cpu-summary "${LOG_DIR_CPU}/summary.tsv" \
    --gpu-summary "${LOG_DIR_TORCH}/summary.tsv" \
    --out "${WALLS_OUT}" > /dev/null
WALLS_EC=$?

# ---- Phase 5: HDF5 product diff ----
echo "==== [Phase 5] HDF5 product diff ===="
"${PYTHON_BIN}" "${BENCH_ROOT}/tools/compare_h5_outputs.py" \
    --cpu-workdir "${WORK_DIR_CPU}" \
    --gpu-workdir "${WORK_DIR_TORCH}" \
    --out "${PRODUCTS_OUT}" > /dev/null
PRODUCTS_EC=$?

# ---- Phase 6: Headline ----
echo
echo "==== Summary: ${SCENE} ===="
echo "CPU run exit:        ${CPU_EC}"
echo "Torch run exit:      ${TORCH_EC}"
echo "Walls diff exit:     ${WALLS_EC} (0 = no CPU-only step regression > ±5%)"
echo "Products diff exit:  ${PRODUCTS_EC} (0 = all gates pass at rms/scale < 1e-5)"
echo
echo "Artifacts:"
echo "  cpu logs:      ${LOG_DIR_CPU}"
echo "  torch logs:    ${LOG_DIR_TORCH}"
echo "  walls diff:    ${WALLS_OUT}"
echo "  products diff: ${PRODUCTS_OUT}"
echo "  cpu workdir:   ${WORK_DIR_CPU}"
echo "  torch workdir: ${WORK_DIR_TORCH}"

# Pull the headline numbers out of the diff JSONs for at-a-glance view.
if [[ -f "${WALLS_OUT}" ]]; then
    echo
    "${PYTHON_BIN}" - "${WALLS_OUT}" "${PRODUCTS_OUT}" <<'PY'
import json, os, sys
walls = json.loads(open(sys.argv[1]).read())
prods_path = sys.argv[2] if len(sys.argv) > 2 else None
prods = (json.loads(open(prods_path).read())
         if prods_path and os.path.isfile(prods_path) else None)
t = walls["totals"]
print(f"  GPU-able subtotal:  cpu={t['gpu_able_cpu_wall_s']:.1f}s  "
      f"torch={t['gpu_able_gpu_wall_s']:.1f}s  speedup={t['gpu_able_speedup']:.2f}x")
print(f"  CPU-only subtotal:  cpu={t['cpu_only_cpu_wall_s']:.1f}s  "
      f"torch={t['cpu_only_gpu_wall_s']:.1f}s  ratio={t['cpu_only_ratio_gpu_over_cpu']:.3f}")
if walls["cpu_only_regressions"]:
    print(f"  CPU-only regressions: {walls['cpu_only_regressions']}")
if prods:
    s = prods["summary"]
    print(f"  Product gates: {s['n_gate_pass']}/{s['n_compared']} pass "
          f"(threshold rms/scale < {prods['gate_threshold_rms_over_scale']})")
    for p in prods["products"]:
        st = p.get("status")
        if st == "compared" and not p.get("gate_pass"):
            print(f"    FAIL: {p['file']}  rms/scale={p['rms_over_scale']:.2e}")
        elif st in ("absent_in_one", "filename_mismatch",
                    "dataset_key_error") and p.get("required"):
            print(f"    {st}: {p['file']}")
else:
    print(f"  (products diff JSON missing — tool failed earlier)")
PY
fi

if [[ ${CPU_EC} -ne 0 || ${TORCH_EC} -ne 0 \
      || ${WALLS_EC} -ne 0 || ${PRODUCTS_EC} -ne 0 ]]; then
    exit 1
fi
