# Vendored AutoRL codebases

Official clones are **not** committed here (see root `.gitignore`).
Run [`../scripts/clone_vendors.sh`](../scripts/clone_vendors.sh) to fetch pinned commits:

| Repo | URL | Pinned commit |
|------|-----|---------------|
| SEARL | https://github.com/automl/SEARL | `bac75d8c9540ff4f0b5b340c612ec384b189bd84` |
| HOOF | https://github.com/supratikp/HOOF | `a95d73588feed910148280df0f962a24cbd5690d` (+ free-MuJoCo Docker patch via `ensure_hoof_mujoco_free.sh`) |
| HyperController | https://github.com/jongornet14/HyperController | `a3a5430ffe30d594ecfb9c2747b4fc637a58151f` |

Do not rewrite their training loops. Paper-faithful overrides live under `../configs/` and `../scripts/`.
