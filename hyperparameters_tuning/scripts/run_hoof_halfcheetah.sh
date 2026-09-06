#!/usr/bin/env bash
# Run HOOF HalfCheetah-v2 experiments (A2C and/or TNPG) via author Docker or local hoof/.
#
# MuJoCo is free/open — no personal license key required. This wrapper calls
# ensure_hoof_mujoco_free.sh to install DeepMind's public unlocked mjkey and a
# Docker template that wget's it during build.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_HOOF="${ROOT}/vendor/hoof"
MODE="a2c"          # a2c | npg | both | smoke
USE_DOCKER=1
ENV_KEY="cheetah"   # maps to HalfCheetah-v2 in experiment_params.yaml
OUT_DIR="${ROOT}/results/hoof"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--mode a2c|npg|both|smoke] [--no-docker] [--env-key cheetah]

  --mode a2c     Full A2C HalfCheetah suite (10 seeds x methods in run_all_a2c_experiments.sh)
  --mode npg     Full TNPG/TRPO suite
  --mode both    A2C then NPG
  --mode smoke   Single HOOF_A2C_LR seed 0 only (still full 5e6 steps unless you edit yaml)
  --no-docker    Run python directly from vendor/hoof/hoof (needs Baselines; free mjkey auto-fetched)

No personal MuJoCo key is required. Free unlocked key is fetched automatically.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --no-docker) USE_DOCKER=0; shift ;;
    --env-key) ENV_KEY="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ ! -d "${VENDOR_HOOF}" ]]; then
  echo "HOOF vendor missing. Run: ${ROOT}/scripts/clone_vendors.sh" >&2
  exit 1
fi

# Install free/open MuJoCo activation + keyless Docker template (idempotent).
bash "${ROOT}/scripts/ensure_hoof_mujoco_free.sh"

mkdir -p "${OUT_DIR}"

# Optional: still honor MUJOCO_KEY if user wants to override the free key
if [[ -n "${MUJOCO_KEY:-}" && -f "${MUJOCO_KEY}" ]]; then
  cp -f "${MUJOCO_KEY}" "${VENDOR_HOOF}/mjkey.txt"
  echo "[hoof] using MUJOCO_KEY override -> vendor/hoof/mjkey.txt"
fi

run_inside_hoof() {
  local cmd="$1"
  if [[ "${USE_DOCKER}" -eq 1 ]]; then
    if [[ ! -f "${VENDOR_HOOF}/mjkey.txt" ]]; then
      echo "[hoof] free mjkey missing after ensure_hoof_mujoco_free.sh" >&2
      exit 1
    fi
    if ! command -v docker >/dev/null 2>&1; then
      echo "[hoof] docker not found; pass --no-docker if a local Baselines env exists" >&2
      exit 1
    fi
    echo "[hoof] building / starting Docker (keyless template; free mjkey auto-fetched)"
    (
      cd "${VENDOR_HOOF}"
      if ! docker images --format '{{.Repository}}' | grep -qx hoof; then
        bash build.sh
      fi
    )
    echo "[hoof] invoking: ${cmd}"
    # Non-interactive: mount repo and run command inside image
    docker run --rm \
      -v "${VENDOR_HOOF}:/project:rw" \
      -w /project/hoof \
      -e MUJOCO_PY_MJKEY_PATH=/project/mjkey.txt \
      hoof \
      bash -lc "${cmd}"
  else
    echo "[hoof] --no-docker: running from ${VENDOR_HOOF}/hoof"
    export MUJOCO_PY_MJKEY_PATH="${VENDOR_HOOF}/mjkey.txt"
    if [[ ! -f "${MUJOCO_PY_MJKEY_PATH}" ]]; then
      export MUJOCO_PY_MJKEY_PATH="${HOME}/.mujoco/mjkey.txt"
    fi
    (
      cd "${VENDOR_HOOF}/hoof"
      bash -lc "${cmd}"
    )
  fi
}

link_results() {
  mkdir -p "${OUT_DIR}"
  for d in results_A2C results_NPG; do
    if [[ -d "${VENDOR_HOOF}/hoof/${d}" ]]; then
      ln -sfn "${VENDOR_HOOF}/hoof/${d}" "${OUT_DIR}/${d}" || true
    fi
  done
}

case "${MODE}" in
  smoke)
    echo "[hoof] smoke: HOOF_A2C_LR RMSProp KL=0.03 seed start=0"
    run_inside_hoof "python a2c_experiments.py ${ENV_KEY} HOOF_A2C_LR RMSProp 0.03 0"
    ;;
  a2c)
    run_inside_hoof "bash run_all_a2c_experiments.sh ${ENV_KEY}"
    ;;
  npg)
    run_inside_hoof "bash run_all_npg_experiments.sh ${ENV_KEY}"
    ;;
  both)
    run_inside_hoof "bash run_all_a2c_experiments.sh ${ENV_KEY}"
    run_inside_hoof "bash run_all_npg_experiments.sh ${ENV_KEY}"
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    usage
    exit 1
    ;;
esac

link_results
echo "[hoof] done. Results under ${OUT_DIR} (and vendor/hoof/hoof/results_*)."
