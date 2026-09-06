# Dependency notes (HalfCheetah reproduction)

These three stacks **do not share** one conda env. Prefer each vendor’s own environment.

## HyperController (easiest modern stack)

Dedicated conda env used for smokes on this machine:

```bash
conda create -y -n hypercontroller python=3.11
conda activate hypercontroller
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  torchrl==0.6.0 tensordict==0.6.2 \
  gym==0.26.2 mujoco==3.2.7 \
  'ray[tune]==2.43.0' pyarrow GPy==1.13.2 \
  pandas==2.2.3 numpy==1.26.4 matplotlib==3.10.0 \
  tensorboard==2.18.0 tqdm==4.67.1 scikit-learn==1.6.1 scipy==1.12.0
```

Or `conda env create -f vendor/hypercontroller/environment.yml` (heavier; includes unused genAI packages).

```bash
export HYPERCONTROLLER_PYTHON=/home/stefano/miniconda3/envs/hypercontroller/bin/python
bash scripts/run_hypercontroller_halfcheetah.sh --smoke
bash scripts/run_hypercontroller_halfcheetah.sh   # full 10×6×1M
```

HalfCheetah-v4 via TorchRL `GymEnv`.

## SEARL (TD3 + MuJoCo)

Same `hypercontroller` env works for smoke with:

```bash
pip install fastrand pyaml   # into that env
export SEARL_PYTHON=/home/stefano/miniconda3/envs/hypercontroller/bin/python
bash scripts/run_searl_halfcheetah.sh --smoke
```

Launcher [`scripts/run_searl_td3_compat.py`](scripts/run_searl_td3_compat.py) applies a **gym 0.26** seed/reset/step/`_max_episode_steps` shim and a spawn-Pool initializer. Vendor sources are not edited.

- **Smoke config** uses `HalfCheetah-v4` (modern mujoco).
- **Paper config** keeps `HalfCheetah-v2` — needs mujoco-py / Gym v2 stack for faithful dynamics.

```bash
bash scripts/run_searl_halfcheetah.sh   # paper YAML, HalfCheetah-v2
```

## HOOF (Docker + Baselines; **no personal MuJoCo key**)

MuJoCo is free/open. You do **not** need a personal license:

```bash
bash scripts/ensure_hoof_mujoco_free.sh   # also run from clone_vendors.sh
```

This:

1. Installs DeepMind’s **public unlocked** `mjkey.txt` (for mujoco-py 2.0) into `vendor/hoof/` and `~/.mujoco/`.
2. Replaces the vendor Docker template with [`configs/hoof_Dockerfile.cuda.nokey.template`](configs/hoof_Dockerfile.cuda.nokey.template), which **wget**s that free key during image build (no `COPY ./mjkey.txt`).

Then:

```bash
bash scripts/run_hoof_halfcheetah.sh --mode smoke   # builds Docker if needed
bash scripts/run_hoof_halfcheetah.sh --mode a2c
bash scripts/run_hoof_halfcheetah.sh --mode npg
```

Optional override: `export MUJOCO_KEY=/path/to/other/mjkey.txt`.

Docker still pins CUDA 8 / Ubuntu 16.04 / Python 3.5 / OpenAI Baselines — that is separate from the license. If the ancient CUDA image cannot run on this host, a modern Baselines port is a separate follow-up (do not silently swap A2C).

## Machine info

HyperController wall-clock claims (&lt;0.1 min vs &gt;10 min for GP-UCB/PB2) are **hardware-specific**. Each HC run writes `results/hypercontroller/.../machine_info.txt`.
