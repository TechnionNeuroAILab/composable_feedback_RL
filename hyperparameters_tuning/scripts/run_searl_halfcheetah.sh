#!/usr/bin/env bash
# Run SEARL TD3 on HalfCheetah-v2 with the paper Table-2 config.
# Fair protocol: ALL population env steps count (handled inside SEARL logging).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_SEARL="${ROOT}/vendor/searl"
CONFIG="${ROOT}/configs/searl_td3_halfcheetah.yml"
OUT_DIR="${ROOT}/results/searl/halfcheetah_td3_paper"
SMOKE=0
SEED_TAG=""

usage() {
  cat <<EOF
Usage: $(basename "$0") [--smoke] [--config PATH] [--out DIR]

  --smoke     Use tiny smoke config (pop=2, 5k frames)
  --config    Override YAML config
  --out       Experiment output directory
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) SMOKE=1; shift ;;
    --config) CONFIG="$2"; shift 2 ;;
    --out) OUT_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ ! -d "${VENDOR_SEARL}" ]]; then
  echo "SEARL vendor missing. Run: ${ROOT}/scripts/clone_vendors.sh" >&2
  exit 1
fi

if [[ "${SMOKE}" -eq 1 ]]; then
  CONFIG="${ROOT}/configs/searl_td3_halfcheetah_smoke.yml"
  OUT_DIR="${ROOT}/results/searl/halfcheetah_td3_smoke"
fi

mkdir -p "${OUT_DIR}"
export PYTHONPATH="${VENDOR_SEARL}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON_BIN="${SEARL_PYTHON:-python3}"

echo "[searl] config=${CONFIG}"
echo "[searl] out=${OUT_DIR}"
echo "[searl] python=${PYTHON_BIN}"
# Use compat launcher (gym 0.26 seed/reset/step shim); does not edit vendor sources.
"${PYTHON_BIN}" "${ROOT}/scripts/run_searl_td3_compat.py" \
  --config_file "${CONFIG}" \
  --expt_dir "${OUT_DIR}"
