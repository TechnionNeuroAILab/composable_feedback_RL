#!/usr/bin/env bash
# Train composable PPO v4 (backprop routing, no B) on HalfCheetah-v4 and plot figures.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DEVICE="${DEVICE:-cuda}"
RESULTS_DIR="${RESULTS_DIR:-results/ppo_local_channel_control/HalfCheetah-v4__local_control_v4__seed1_5000ep}"
FIG_DIR="${FIG_DIR:-paper/figures/ppo_local_channel_control_halfcheetah_v4}"
LOG_DIR="${LOG_DIR:-logs/ppo_local_channel_control}"
mkdir -p "$LOG_DIR"

echo "[$(date -Is)] Starting HalfCheetah v4 (backprop, no B) on ${DEVICE}" | tee -a "${LOG_DIR}/halfcheetah_v4.log"

python code/ppo_local_channel_control/train.py \
  --halfcheetah-v4 \
  --device "$DEVICE" \
  --seed 1 \
  --output-dir "$RESULTS_DIR" \
  2>&1 | tee -a "${LOG_DIR}/halfcheetah_v4_train.log"

echo "[$(date -Is)] Plotting v4 diagnostics" | tee -a "${LOG_DIR}/halfcheetah_v4.log"
python code/ppo_local_channel_control/plot_results.py \
  --results-dir "$RESULTS_DIR" \
  --output-dir "$FIG_DIR"

ABLATION_DIR="${ABLATION_DIR:-results/ppo_local_channel_control/HalfCheetah-v4__ablation_k1_backprop__seed1_5000ep}"

if [[ ! -f "${ABLATION_DIR}/episode_returns.json" ]]; then
  echo "[$(date -Is)] Running fair K=1 backprop ablation on ${DEVICE}" | tee -a "${LOG_DIR}/halfcheetah_v4.log"
  python code/ppo_local_channel_control/train.py \
    --halfcheetah-v4-ablation-k1 \
    --device "$DEVICE" \
    --seed 1 \
    --output-dir "$ABLATION_DIR" \
    2>&1 | tee -a "${LOG_DIR}/halfcheetah_v4_ablation_k1.log"
fi

echo "[$(date -Is)] Benchmark comparison (fair K=1 backprop ablation + legacy sparse-B + CleanRL)" | tee -a "${LOG_DIR}/halfcheetah_v4.log"
python code/ppo_local_channel_control/plot_benchmark_comparison.py \
  --composable "$RESULTS_DIR" \
  --composable-label "Composable PPO v4 (K=4, backprop, no B)" \
  --ablation "$ABLATION_DIR" \
  --ablation-label "Fair ablation: K=1, backprop (no B)" \
  --ablation-legacy results/ppo_local_channel_control/HalfCheetah-v4__ablation_k1__seed1_5000ep \
  --ablation-legacy-label "Legacy ablation: K=1, sparse B (v3 routing)" \
  --cleanrl results/cleanrl_ppo_baseline/HalfCheetah-v4__cleanrl__seed1_5000ep \
  --title "HalfCheetah-v4: v4 vs. fair K=1 ablation and benchmarks" \
  --output "${FIG_DIR}/06_benchmark_comparison.png"

echo "[$(date -Is)] Done. Figures in ${FIG_DIR}" | tee -a "${LOG_DIR}/halfcheetah_v4.log"
