#!/usr/bin/env bash
# Clone / update vendor repos at pinned commits.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR="${ROOT}/vendor"
mkdir -p "${VENDOR}"

clone_pin() {
  local name="$1"
  local url="$2"
  local commit="$3"
  local dest="${VENDOR}/${name}"
  if [[ -d "${dest}/.git" ]]; then
    echo "[clone_vendors] ${name}: already present, fetching..."
    git -C "${dest}" fetch --depth 1 origin "${commit}" 2>/dev/null \
      || git -C "${dest}" fetch --depth 1 origin
    git -C "${dest}" checkout --detach "${commit}"
  else
    echo "[clone_vendors] ${name}: cloning ${url} @ ${commit}"
    rm -rf "${dest}"
    git clone "${url}" "${dest}"
    git -C "${dest}" checkout --detach "${commit}"
  fi
  echo "[clone_vendors] ${name} -> $(git -C "${dest}" rev-parse HEAD)"
}

clone_pin searl \
  https://github.com/automl/SEARL.git \
  bac75d8c9540ff4f0b5b340c612ec384b189bd84

clone_pin hoof \
  https://github.com/supratikp/HOOF.git \
  a95d73588feed910148280df0f962a24cbd5690d

clone_pin hypercontroller \
  https://github.com/jongornet14/HyperController.git \
  a3a5430ffe30d594ecfb9c2747b4fc637a58151f

# Re-apply free/open MuJoCo setup (no personal key) after every clone.
bash "${ROOT}/scripts/ensure_hoof_mujoco_free.sh"

echo "[clone_vendors] done."
