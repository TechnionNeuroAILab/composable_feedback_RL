#!/usr/bin/env bash
# Run HyperController PPO HalfCheetah-v4 paper sweep (or a smoke subset).
# Methods: HyperController, Random, Random_Start, HyperBand, GP-UCB, PB2
# Seeds: 0..9 ; total_frames=1e6 ; t_ready=5
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_HC="${ROOT}/vendor/hypercontroller"
OUT_DIR="${ROOT}/results/hypercontroller/halfcheetah_v4"
SMOKE=0
SEEDS="0 1 2 3 4 5 6 7 8 9"
METHODS="HyperController Random Random_Start HyperBand GP-UCB PB2"
TOTAL_FRAMES=1000000
T_READY=5
PYTHON_BIN="${HYPERCONTROLLER_PYTHON:-python3}"
FORCE=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [--smoke] [--force] [--seeds "0 1"] [--methods "HyperController Random"]
                        [--total-frames N] [--python PATH] [--out DIR]

  --smoke   2 seeds, HyperController+Random only, 20k frames (~20 iters)
            writes to results/hypercontroller/smoke_halfcheetah_v4
  --force   Delete existing per-run log dirs before launching (author code
            skips runs when logs.csv already exists)
  --out     Output folder (default: halfcheetah_v4 or smoke_… when --smoke)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke)
      SMOKE=1
      SEEDS="0 1"
      METHODS="HyperController Random"
      TOTAL_FRAMES=20000
      OUT_DIR="${ROOT}/results/hypercontroller/smoke_halfcheetah_v4"
      shift
      ;;
    --force) FORCE=1; shift ;;
    --out) OUT_DIR="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --methods) METHODS="$2"; shift 2 ;;
    --total-frames) TOTAL_FRAMES="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ ! -d "${VENDOR_HC}" ]]; then
  echo "HyperController vendor missing. Run: ${ROOT}/scripts/clone_vendors.sh" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
# Log machine info (wall-clock claims are hardware-specific)
{
  echo "timestamp=$(date -Is)"
  echo "hostname=$(hostname)"
  echo "python=${PYTHON_BIN}"
  "${PYTHON_BIN}" - <<'PY' 2>/dev/null || true
import platform, sys
print("platform", platform.platform())
print("python", sys.version.replace("\n", " "))
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu", torch.cuda.get_device_name(0))
except Exception as e:
    print("torch", e)
PY
  nvidia-smi -L 2>/dev/null || true
} > "${OUT_DIR}/machine_info.txt"

FOLDER_NAME="$(basename "${OUT_DIR}")"
# Author code writes under --folder relative to cwd
ABS_FOLDER="${OUT_DIR}"

cd "${VENDOR_HC}"
export PYTHONPATH="${VENDOR_HC}${PYTHONPATH:+:${PYTHONPATH}}"

for seed in ${SEEDS}; do
  for method in ${METHODS}; do
    run_dir="${ABS_FOLDER}/HalfCheetah-v4_${method}_seed${seed}"
    if [[ "${FORCE}" -eq 1 && -d "${run_dir}" ]]; then
      echo "[hypercontroller] --force: removing ${run_dir}"
      rm -rf "${run_dir}"
    fi
    echo "[hypercontroller] seed=${seed} method=${method} frames=${TOTAL_FRAMES}"
    "${PYTHON_BIN}" rl_tests.py \
      --seed "${seed}" \
      --env "HalfCheetah-v4" \
      --total_frames "${TOTAL_FRAMES}" \
      --t_ready "${T_READY}" \
      --method "${method}" \
      --folder "${ABS_FOLDER}"
  done
done

echo "[hypercontroller] done. Logs under ${OUT_DIR}/"
echo "[hypercontroller] machine info: ${OUT_DIR}/machine_info.txt"
