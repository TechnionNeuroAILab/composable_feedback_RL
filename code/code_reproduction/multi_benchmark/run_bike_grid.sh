#!/usr/bin/env bash
# Bike grid: lanes {2,3,5,10} x solutions {1,8,100}, 3 seeds, 480k agent decisions, 7 variants:
#   searl_B      SEARL DQN (searl_discrete_confgate.py --no-controller: heads present but inert; identical to plain SEARL)
#   searl_A2     SEARL + 2 heads (left / right view)       searl_Aobs   SEARL + one head per observation
#   dqn          CleanRL DQN                               dqn_A2 / dqn_Aobs   single agent + 2 heads / per-observation heads
#   rainbow      Rainbow (Kaixhin port), bike settings of rainbow_dqn/run_discrete_tasks.sh
# Every variant writes <out>/L<l>_S<s>/<variant>/seed<k>/eval.csv = global_step,greedy_return from the SAME function
# (common_bike.greedy_return: no exploration, 10 fixed fresh boards, seeds 5000-5009, every ~10k steps; not training steps).
# Usage: run_bike_grid.sh            (everything)   SOLS="1 8" LANES="2 3" VARIANTS="dqn" STEPS=480000 PAR=100 ./run_bike_grid.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; CR="$HERE/.."
export PATH=/home/yaarit/miniconda3/envs/cleanrl_dqn_repro/bin:$PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
OUT=${OUT:-/home/yaarit/projects/msc_project/composable_feedback_RL/results/bike_grid}
STEPS=${STEPS:-480000}; EVERY=${EVERY:-10000}
LANES=${LANES:-"2 3 5 10"}; SOLS=${SOLS:-"100 8 1"}; SEEDS=${SEEDS:-"0 1 2"}
VARIANTS=${VARIANTS:-"searl_B searl_A2 searl_Aobs dqn dqn_A2 dqn_Aobs rainbow"}
EV="--eval-episodes 10 --eval-seed0 5000"
run() { v=$1; L=$2; S=$3; s=$4; d=$OUT/L${L}_S${S}/$v/seed$s; [ -f $d/eval.csv ] && return; mkdir -p $d
  case $v in
    searl_B)    (cd $CR/searl && python searl_discrete_confgate.py --task bike --env bike --lanes $L --solutions $S --seed $s --pop 4 --frames $STEPS $EV --eval-every-frames $EVERY --no-controller --out $d) ;;
    searl_A2)   (cd $CR/searl && python searl_discrete_confgate.py --task bike --env bike --lanes $L --solutions $S --seed $s --pop 4 --frames $STEPS $EV --eval-every-frames $EVERY --out $d) ;;
    searl_Aobs) (cd $CR/searl && python searl_discrete_confgate.py --task bike --env bike --lanes $L --solutions $S --seed $s --pop 4 --frames $STEPS $EV --eval-every-frames $EVERY --heads per_obs --out $d) ;;
    dqn)        (cd $HERE && python cleanrl_dqn_bike.py --env_id bike_v2.1 --lanes $L --solutions $S --seed $s --total_timesteps $STEPS --no-cuda --csv_path $d/train.csv --eval-every $EVERY --eval-csv $d/eval.csv.tmp && mv $d/eval.csv.tmp $d/eval.csv) ;;
    dqn_A2)     (cd $HERE && python cleanrl_dqn_confgate.py --env_id bike_v2.1 --lanes $L --solutions $S --seed $s --total_timesteps $STEPS --no-cuda --csv_path $d/train.csv --conf-log-path $d/conf.csv --eval-every $EVERY --eval-csv $d/eval.csv.tmp && mv $d/eval.csv.tmp $d/eval.csv) ;;
    dqn_Aobs)   (cd $HERE && python cleanrl_dqn_confgate.py --env_id bike_v2.1 --lanes $L --solutions $S --seed $s --total_timesteps $STEPS --no-cuda --heads per_obs --csv_path $d/train.csv --conf-log-path $d/conf.csv --eval-every $EVERY --eval-csv $d/eval.csv.tmp && mv $d/eval.csv.tmp $d/eval.csv) ;;
    rainbow)    (cd $CR/rainbow_dqn && python main.py --task gym --env bike --lanes $L --solutions $S --seed $s --T-max $STEPS --learn-start 2000 --target-update 1000 --memory-capacity 100000 --replay-frequency 2 --learning-rate 2.5e-4 --reward-scale 0.005 --V-min -10 --V-max 10 --obs-scale 1 --csv-path $d/train.csv --eval-every $EVERY --eval-episodes 10 --eval-seed0 5000 --eval-csv $d/eval.csv.tmp && mv $d/eval.csv.tmp $d/eval.csv) ;;
  esac > $d/run.log 2>&1; }
export -f run; export OUT STEPS EVERY EV CR HERE
for S in $SOLS; do for L in $LANES; do for v in $VARIANTS; do for s in $SEEDS; do echo "$v $L $S $s"; done; done; done; done \
  | xargs -P ${PAR:-100} -L 1 bash -c 'run "$@"' _
echo "ALL DONE"
