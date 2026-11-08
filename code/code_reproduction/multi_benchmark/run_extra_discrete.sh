#!/usr/bin/env bash
# Bike-panel methods (SEARL DQN, random search, CleanRL DQN, CleanRL PPO, HyperController, PB2),
# 3 seeds, on Acrobot, LunarLander, Hopper (9 outputs), Freeway RAM, and FrozenLake (sanity).
# Budgets match bike (480k) except FrozenLake (100k).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export PATH=/home/yaarit/miniconda3/envs/cleanrl_dqn_repro/bin:$PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
R=results_extra
mkdir -p logs_extra
run() { env=$1; steps=$2; kind=$3; s=$4
  slug=${env//\//_}
  base=$R/$slug
  mkdir -p "$base/searl" "$base/rs" "$base/cleanrl_dqn" "$base/ppo"
  export CUDA_VISIBLE_DEVICES=""
  case $kind in
    searl) (cd ../searl && python searl_discrete.py --task gym --env "$env" --seed "$s" --pop 4 --frames "$steps" \
              --out "../multi_benchmark/$base/searl/seed$s") ;;
    rs) python searl_mlp_dqn.py --method random_search --env "$env" --seed "$s" --n_configs 10 \
              --frames_per_config $((steps / 10)) --out "$base/rs/seed$s" ;;
    dqn) python cleanrl_dqn_bike.py --env_id "$env" --seed "$s" --total_timesteps "$steps" --no-cuda \
              --csv_path "$base/cleanrl_dqn/seed$s.csv.tmp" \
              && mv "$base/cleanrl_dqn/seed$s.csv.tmp" "$base/cleanrl_dqn/seed$s.csv" ;;
    ppo|HyperController|PB2)
       h=$kind; [ "$kind" = ppo ] && h=none
       python ppo_hpo.py --task gym --env_id "$env" --hpo "$h" --seed "$s" --total_timesteps "$steps" \
              --csv_path "$base/ppo/${h}_seed$s.csv.tmp" \
              && mv "$base/ppo/${h}_seed$s.csv.tmp" "$base/ppo/${h}_seed$s.csv" ;;
  esac > "logs_extra/${slug}_${kind}_seed${s}.log" 2>&1; }
export -f run; export R
{
for spec in "Acrobot-v1 480000" "LunarLander-v2 480000" "Hopper-v4 480000" "ALE/Freeway-ram-v5 480000" "FrozenLake-v1 100000"; do
  # shellcheck disable=SC2086
  set -- $spec
  env=$1; steps=$2
  slug=${env//\//_}
  for s in 0 1 2; do
    [ -f "$R/$slug/searl/seed$s/history.csv" ] || echo "$env $steps searl $s"
    [ -f "$R/$slug/rs/seed$s/history.csv" ] || echo "$env $steps rs $s"
    [ -f "$R/$slug/cleanrl_dqn/seed$s.csv" ] || echo "$env $steps dqn $s"
    for h in ppo HyperController PB2; do
      hh=$h; [ "$h" = ppo ] && hh=none
      [ -f "$R/$slug/ppo/${hh}_seed$s.csv" ] || echo "$env $steps $h $s"
    done
  done
done
} | xargs -P "${PARALLEL:-12}" -L 1 bash -c 'run "$@"' _
echo "ALL DONE"
