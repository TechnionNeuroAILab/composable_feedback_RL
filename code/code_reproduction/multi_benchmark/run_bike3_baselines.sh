#!/usr/bin/env bash
# bike v3.0 learnability ladder: levels L1 (route = v2.1), L2 (+ speed with speed limits), L4.1 (route + balance, free
# speed, no speed limits, turn kick 2 at speed 5); 5 lanes, 8 solutions,
# 3 seeds, 480k agent decisions. Methods: plain SEARL DQN, CleanRL DQN, Rainbow (Kaixhin port), with the bike-grid settings
# of run_bike_grid.sh. Every run writes <OUT>/L<level>/<method>/seed<k>/eval.csv = global_step,greedy_return from the SAME
# function (common_bike.greedy_return with level: no exploration, 10 fixed fresh boards, seeds 5000-5009, every ~10k steps).
# Extra methods: dqn_vanilla = CleanRL DQN with ONE Q-head over the 9 joint (steer, throttle) actions (the plain
# 'dqn' uses two independent heads); dqn_vanilla_look4 = the same + the hole grid of the next 4 rows in the observation.
# searl_vanilla / rainbow_vanilla = SEARL / Rainbow with the same single head over the 9 joint actions; otherwise the
# exact v2.1 bike-grid code path and settings (run_bike_grid.sh) of each method.
# Usage: run_bike3_baselines.sh      LEVELS="4.1" METHODS="dqn_vanilla" SEEDS="0" ./run_bike3_baselines.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; CR="$HERE/.."
export PATH=/home/yaarit/miniconda3/envs/cleanrl_dqn_repro/bin:$PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
OUT=${OUT:-/home/yaarit/projects/msc_project/composable_feedback_RL/results/bike3}
STEPS=${STEPS:-480000}; EVERY=${EVERY:-10000}
LEVELS=${LEVELS:-"1 2 4.1"}; SEEDS=${SEEDS:-"0 1 2"}; METHODS=${METHODS:-"searl dqn rainbow"}
EV="--eval-episodes 10 --eval-seed0 5000"
run() { m=$1; l=$2; s=$3; d=$OUT/L$l/$m/seed$s; [ -f $d/eval.csv ] && return; mkdir -p $d
  case $m in
    searl)   (cd $CR/searl && python searl_discrete.py --task bike3 --env L$l --seed $s --pop 4 --frames $STEPS $EV --eval-every-frames $EVERY --out $d) ;;
    dqn)     (cd $HERE && python cleanrl_dqn_bike.py --env_id bike_v3.0_L$l --seed $s --total_timesteps $STEPS --no-cuda --csv_path $d/train.csv --eval-every $EVERY $EV --eval-csv $d/eval.csv.tmp && mv $d/eval.csv.tmp $d/eval.csv) ;;
    dqn_vanilla)       (cd $HERE && python cleanrl_dqn_bike.py --env_id bike_v3.0_L$l --seed $s --total_timesteps $STEPS --no-cuda --joint-actions --csv_path $d/train.csv --eval-every $EVERY $EV --eval-csv $d/eval.csv.tmp && mv $d/eval.csv.tmp $d/eval.csv) ;;
    dqn_vanilla_look4) (cd $HERE && python cleanrl_dqn_bike.py --env_id bike_v3.0_L$l --seed $s --total_timesteps $STEPS --no-cuda --joint-actions --lookahead 4 --csv_path $d/train.csv --eval-every $EVERY $EV --eval-csv $d/eval.csv.tmp && mv $d/eval.csv.tmp $d/eval.csv) ;;
    searl_vanilla)   (cd $CR/searl && python searl_discrete.py --task bike3 --env L$l --seed $s --pop 4 --frames $STEPS $EV --eval-every-frames $EVERY --joint-actions --out $d) ;;
    rainbow_vanilla) (cd $CR/rainbow_dqn && python main.py --task gym --env bike3_L$l --seed $s --T-max $STEPS --learn-start 2000 --target-update 1000 --memory-capacity 100000 --replay-frequency 2 --learning-rate 2.5e-4 --reward-scale 0.005 --V-min -10 --V-max 10 --obs-scale 1 --joint-actions --csv-path $d/train.csv --eval-every $EVERY $EV --eval-csv $d/eval.csv.tmp && mv $d/eval.csv.tmp $d/eval.csv) ;;
    rainbow) (cd $CR/rainbow_dqn && python main.py --task gym --env bike3_L$l --seed $s --T-max $STEPS --learn-start 2000 --target-update 1000 --memory-capacity 100000 --replay-frequency 2 --learning-rate 2.5e-4 --reward-scale 0.005 --V-min -10 --V-max 10 --obs-scale 1 --csv-path $d/train.csv --eval-every $EVERY $EV --eval-csv $d/eval.csv.tmp && mv $d/eval.csv.tmp $d/eval.csv) ;;
  esac > $d/run.log 2>&1; }
export -f run; export OUT STEPS EVERY EV CR HERE
for l in $LEVELS; do for m in $METHODS; do for s in $SEEDS; do echo "$m $l $s"; done; done; done | xargs -P ${PAR:-27} -L 1 bash -c 'run "$@"' _
echo "ALL DONE"
