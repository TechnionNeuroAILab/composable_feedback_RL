#!/usr/bin/env bash
# Cheap end-to-end smoke for all three wrappers.
# Does NOT launch full paper budgets.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

echo "======== clone vendors if needed ========"
if [[ ! -d vendor/searl || ! -d vendor/hoof || ! -d vendor/hypercontroller ]]; then
  bash scripts/clone_vendors.sh
fi

echo "======== HyperController smoke ========"
if bash scripts/run_hypercontroller_halfcheetah.sh --smoke; then
  echo "[smoke] HyperController OK"
else
  echo "[smoke] HyperController FAILED (deps: torchrl/ray/gym/mujoco — see requirements-notes.md)" >&2
fi

echo "======== SEARL smoke ========"
if bash scripts/run_searl_halfcheetah.sh --smoke; then
  echo "[smoke] SEARL OK"
else
  echo "[smoke] SEARL FAILED (deps: mujoco-py / HalfCheetah-v2 — see requirements-notes.md)" >&2
fi

echo "======== HOOF smoke (free MuJoCo; Docker preferred) ========"
bash scripts/ensure_hoof_mujoco_free.sh
if command -v docker >/dev/null 2>&1; then
  if bash scripts/run_hoof_halfcheetah.sh --mode smoke; then
    echo "[smoke] HOOF OK"
  else
    echo "[smoke] HOOF FAILED — see requirements-notes.md (Docker/Baselines)" >&2
  fi
elif bash scripts/run_hoof_halfcheetah.sh --mode smoke --no-docker; then
  echo "[smoke] HOOF OK (--no-docker)"
else
  echo "[smoke] HOOF FAILED — need Docker image 'hoof' or local Baselines; free mjkey is already installed" >&2
fi

echo "======== smoke_all finished ========"
