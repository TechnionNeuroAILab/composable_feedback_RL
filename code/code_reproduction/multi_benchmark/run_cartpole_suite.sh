#!/usr/bin/env bash
# CartPole-v1 (the task of code/conf_gate/confgate_fourway.py): SEARL, random search, CleanRL DQN, CleanRL PPO,
# HyperController, PB2 -- 1,000,000 env steps each (the fourway runs use 5000 episodes = 0.6-1.2M steps), 10 seeds.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export PATH=/home/yaarit/miniconda3/envs/cleanrl_dqn_repro/bin:$PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
R=results_cartpole; mkdir -p $R/searl $R/rs $R/cleanrl_dqn $R/ppo logs_cartpole
run() { kind=$1; arg=$2; s=$3
  export CUDA_VISIBLE_DEVICES=$(( RANDOM % 2 ))   # only HyperController's own code touches the GPU (tiny tensors)
  case $kind in
    searl) (cd ../searl && python searl_discrete.py --task gym --env CartPole-v1 --seed $s --pop 4 --frames 1000000 \
              --out ../multi_benchmark/$R/searl/seed$s) ;;
    rs)    python searl_mlp_dqn.py --method random_search --env CartPole-v1 --seed $s --n_configs 10 --frames_per_config 100000 \
              --out $R/rs/seed$s ;;
    dqn)   python cleanrl_dqn_bike.py --env_id CartPole-v1 --seed $s --total_timesteps 1000000 --no-cuda \
              --csv_path $R/cleanrl_dqn/seed$s.csv.tmp && mv $R/cleanrl_dqn/seed$s.csv.tmp $R/cleanrl_dqn/seed$s.csv ;;
    ppo)   python ppo_hpo.py --task gym --env_id CartPole-v1 --hpo $arg --seed $s --total_timesteps 1000000 \
              --csv_path $R/ppo/${arg}_seed$s.csv.tmp && mv $R/ppo/${arg}_seed$s.csv.tmp $R/ppo/${arg}_seed$s.csv ;;
  esac > logs_cartpole/${kind}_${arg}_seed$s.log 2>&1; }
export -f run; export R
{
for s in 0 1 2 3 4 5 6 7 8 9; do [ -f $R/searl/seed$s/history.csv ] || echo "searl x $s"; done
for s in 0 1 2 3 4 5 6 7 8 9; do [ -f $R/rs/seed$s/history.csv ] || echo "rs x $s"; done
for s in 0 1 2 3 4 5 6 7 8 9; do
  [ -f $R/cleanrl_dqn/seed$s.csv ] || echo "dqn x $s"
  for h in none HyperController PB2; do [ -f $R/ppo/${h}_seed$s.csv ] || echo "ppo $h $s"; done
done
} | xargs -P "${PARALLEL:-26}" -L 1 bash -c 'run "$@"' _
echo "ALL DONE"
