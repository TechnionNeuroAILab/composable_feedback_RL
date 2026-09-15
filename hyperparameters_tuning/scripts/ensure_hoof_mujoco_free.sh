#!/usr/bin/env bash
# Ensure HOOF can use free/open MuJoCo without a user-supplied license key.
#
# DeepMind unlocked MuJoCo; the public activation file at roboti.us works for
# mujoco-py 2.0. MuJoCo 2.1+ needs no key at all.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_HOOF="${ROOT}/vendor/hoof"
FREE_KEY_URL="${HOOF_FREE_MJKEY_URL:-https://www.roboti.us/file/mjkey.txt}"
TEMPLATE_SRC="${ROOT}/configs/hoof_Dockerfile.cuda.nokey.template"
TEMPLATE_DST="${VENDOR_HOOF}/Dockerfile.cuda.template"

if [[ ! -d "${VENDOR_HOOF}" ]]; then
  echo "[hoof-mujoco] vendor missing; run scripts/clone_vendors.sh first" >&2
  exit 1
fi

# 1) Drop in keyless Docker template (wget public unlocked key during build).
if [[ -f "${TEMPLATE_SRC}" ]]; then
  cp -f "${TEMPLATE_SRC}" "${TEMPLATE_DST}"
  echo "[hoof-mujoco] installed keyless Dockerfile.cuda.template"
fi

# 2) Fetch free unlocked mjkey for local / bind-mount use (no user key needed).
mkdir -p "${VENDOR_HOOF}" "${HOME}/.mujoco"
if [[ ! -f "${VENDOR_HOOF}/mjkey.txt" ]]; then
  echo "[hoof-mujoco] downloading free unlocked mjkey.txt -> vendor/hoof/mjkey.txt"
  wget -q -O "${VENDOR_HOOF}/mjkey.txt" "${FREE_KEY_URL}"
fi
if [[ ! -f "${HOME}/.mujoco/mjkey.txt" ]]; then
  cp -f "${VENDOR_HOOF}/mjkey.txt" "${HOME}/.mujoco/mjkey.txt"
  echo "[hoof-mujoco] installed ~/.mujoco/mjkey.txt"
fi

# 3) Soften author README requirement (local note only; vendor README may be reset on re-clone).
if [[ -f "${VENDOR_HOOF}/README.md" ]] && ! grep -q "free unlocked" "${VENDOR_HOOF}/README.md"; then
  cat >> "${VENDOR_HOOF}/README.md" <<'EOF'

## MuJoCo license (updated)

MuJoCo is free and open source. This fork of the Docker setup **does not require
you to provide a personal license key**. `scripts/ensure_hoof_mujoco_free.sh`
downloads DeepMind's public unlocked `mjkey.txt` (for mujoco-py 2.0) and the
Dockerfile fetches the same file during image build.
EOF
fi

echo "[hoof-mujoco] ready (no user key required)."
echo "  key: ${VENDOR_HOOF}/mjkey.txt"
echo "  docker template: ${TEMPLATE_DST}"
