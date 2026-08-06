"""
Meta-RL HPO ablation: controller adjusts only train_freq, only target_freq,
or both (lr_mult / eps_end stay fixed at baseline).

Modes:
  train_freq_only   — meta-controller picks train_freq; target_freq=500 fixed
  target_freq_only  — meta-controller picks target_freq; train_freq=10 fixed
  both_freq         — meta-controller picks both freqs

Usage:
    python code/confgate_meta_hpo_onetwoparams.py --run-all --tasks CartPole-v1 --seeds 1
    python code/confgate_meta_hpo_onetwoparams.py --knob-mode train_freq_only --plot-only
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import confgate_meta_hpo as base

# Re-use inner-loop building blocks from the full meta-HPO script.
ReplayBuffer = base.ReplayBuffer
DQNAgent = base.DQNAgent
MetaAction = base.MetaAction
_build_meta_state = base.build_meta_state
_controller_step = base._controller_step
_linear_schedule = base._linear_schedule
_set_seed = base._set_seed
_smooth = base._smooth
_parse_list = base._parse_list
_parse_seeds = base._parse_seeds
_filter_results = base._filter_results
plot_eval_curves = base.plot_eval_curves

ROOT = Path(__file__).resolve().parent.parent
CKPT_ROOT = ROOT / "paper" / "_tmp_b_feedb_cg" / "ckpt_meta_hpo_v1"
MAIN_BASELINE_DIR = ROOT / "paper" / "_tmp_b_feedb_cg" / "ckpt_meta_hpo"
FIG_DIR = ROOT / "paper" / "figures" / "conf_meta_hpo_v1(onetwoparams)"

KNOB_MODES = ("train_freq_only", "target_freq_only", "both_freq")
MODE_LABELS = {
    "train_freq_only": "meta-RL: train_freq only",
    "target_freq_only": "meta-RL: target_freq only",
    "both_freq": "meta-RL: train_freq + target_freq",
}
MODE_COLORS = {
    "train_freq_only": base.COLOR_META,   # orange
    "target_freq_only": "#9467bd",        # purple
    "both_freq": "#800020",               # burgundy
}

DEFAULT_TASKS = ["CartPole-v1"]
DEFAULT_SEEDS = [1]
DEFAULT_META_ITERS = base.DEFAULT_META_ITERS
DEFAULT_INNER_STEPS = base.DEFAULT_INNER_STEPS
DEFAULT_EVAL_EPISODES = 2000


class MetaHPOControllerFreqAblated(nn.Module):
    """GRU policy over categorical train/target frequency knobs only."""

    def __init__(self, n_tasks: int, knob_mode: str):
        super().__init__()
        if knob_mode not in KNOB_MODES:
            raise ValueError(f"Unknown knob_mode {knob_mode!r}; expected one of {KNOB_MODES}")
        self.knob_mode = knob_mode
        self.control_train = knob_mode in ("train_freq_only", "both_freq")
        self.control_target = knob_mode in ("target_freq_only", "both_freq")
        self.n_tasks = n_tasks
        self.state_dim = base.BASE_STATE_DIM
        self.task_emb = nn.Embedding(n_tasks, base.TASK_EMB_DIM)
        in_dim = base.BASE_STATE_DIM + base.TASK_EMB_DIM
        self.gru = nn.GRUCell(in_dim, base.CTRL_HIDDEN)
        self.trunk = nn.Sequential(nn.Linear(base.CTRL_HIDDEN, base.CTRL_HIDDEN), nn.ReLU())
        self.tf_head = (
            nn.Linear(base.CTRL_HIDDEN, len(base.TRAIN_FREQ_CHOICES))
            if self.control_train
            else None
        )
        self.tgt_head = (
            nn.Linear(base.CTRL_HIDDEN, len(base.TARGET_FREQ_CHOICES))
            if self.control_target
            else None
        )
        self.value_head = nn.Linear(base.CTRL_HIDDEN, 1)
        self._init_biases()

    def _init_biases(self) -> None:
        with torch.no_grad():
            if self.tf_head is not None:
                self.tf_head.bias.zero_()
                self.tf_head.bias[-1] = 1.0
                nn.init.zeros_(self.tf_head.weight)
            if self.tgt_head is not None:
                self.tgt_head.bias.zero_()
                self.tgt_head.bias[-1] = 1.0
                nn.init.zeros_(self.tgt_head.weight)

    def zero_hidden(
        self, batch: int, device: torch.device, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        return torch.zeros(batch, base.CTRL_HIDDEN, device=device, dtype=dtype)

    def act(
        self,
        state: torch.Tensor,
        task_id: torch.Tensor,
        h_prev: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[MetaAction, torch.Tensor]:
        emb = self.task_emb(task_id)
        x = torch.cat([state, emb], dim=-1)
        h = self.gru(x, h_prev)
        feat = self.trunk(h)

        log_prob_parts: List[torch.Tensor] = []
        entropy_parts: List[torch.Tensor] = []

        train_freq = base.DEFAULT_TRAIN_FREQ
        target_freq = base.DEFAULT_TARGET_FREQ

        if self.control_train:
            assert self.tf_head is not None
            dist_tf = Categorical(logits=self.tf_head(feat))
            tf_idx = dist_tf.probs.argmax(dim=-1) if deterministic else dist_tf.sample()
            train_freq = base.TRAIN_FREQ_CHOICES[int(tf_idx[0].item())]
            log_prob_parts.append(dist_tf.log_prob(tf_idx))
            entropy_parts.append(dist_tf.entropy())

        if self.control_target:
            assert self.tgt_head is not None
            dist_tgt = Categorical(logits=self.tgt_head(feat))
            tgt_idx = dist_tgt.probs.argmax(dim=-1) if deterministic else dist_tgt.sample()
            target_freq = base.TARGET_FREQ_CHOICES[int(tgt_idx[0].item())]
            log_prob_parts.append(dist_tgt.log_prob(tgt_idx))
            entropy_parts.append(dist_tgt.entropy())

        log_prob = sum(log_prob_parts) if log_prob_parts else torch.zeros(state.shape[0], device=state.device)
        entropy = sum(entropy_parts) if entropy_parts else torch.zeros(state.shape[0], device=state.device)
        value = self.value_head(feat).squeeze(-1)

        action = MetaAction(
            lr_mult=1.0,
            eps_end=base.END_E,
            train_freq=train_freq,
            target_freq=target_freq,
            log_prob=log_prob,
            entropy=entropy,
            value=value,
        )
        return action, h.detach()


def _active_knob_keys(knob_mode: str) -> List[str]:
    if knob_mode == "train_freq_only":
        return ["train_freq"]
    if knob_mode == "target_freq_only":
        return ["target_freq"]
    return ["train_freq", "target_freq"]


def run_inner_task(
    env_id: str,
    task_idx: int,
    controller: MetaHPOControllerFreqAblated,
    opt_ctrl: optim.Optimizer,
    device: torch.device,
    inner_steps: int,
    seed: int,
    knob_mode: str,
    use_controller: bool = True,
    update_controller: bool = True,
    fixed_knobs: Optional[Dict] = None,
    deterministic_ctrl: bool = False,
    total_episodes: Optional[int] = None,
) -> Dict:
    env = __import__("gymnasium").make(env_id)
    env = __import__("gymnasium").wrappers.RecordEpisodeStatistics(env)
    obs_dim = int(np.prod(env.observation_space.shape))
    n_actions = int(env.action_space.n)

    agent = DQNAgent(obs_dim, n_actions, device)
    rb = ReplayBuffer(base.BUFFER_SIZE, env.observation_space.shape, device)

    episode_returns: List[float] = []
    knob_history: List[Dict] = []
    active_keys = _active_knob_keys(knob_mode)

    h = controller.zero_hidden(1, device)
    task_id_t = torch.tensor([task_idx], device=device, dtype=torch.long)

    if fixed_knobs is not None:
        lr_mult = float(fixed_knobs.get("lr_mult", 1.0))
        eps_end = float(fixed_knobs.get("eps_end", base.END_E))
        train_freq = int(fixed_knobs.get("train_freq", base.DEFAULT_TRAIN_FREQ))
        target_freq = int(fixed_knobs.get("target_freq", base.DEFAULT_TARGET_FREQ))
    else:
        lr_mult, eps_end = 1.0, base.END_E
        train_freq, target_freq = base.DEFAULT_TRAIN_FREQ, base.DEFAULT_TARGET_FREQ

    agent.set_lr(lr_mult)

    loss_ema = 0.0
    grad_ema = 0.0
    return_ema = 0.0
    prev_return_ema = 0.0
    return_ema_ready = False
    steps_since_meta = 0
    n_meta_actions = 0
    ctrl_loss_sum = 0.0
    ctrl_loss_count = 0

    pending_log_prob: Optional[torch.Tensor] = None
    pending_value: Optional[torch.Tensor] = None
    pending_entropy: Optional[torch.Tensor] = None

    step_cap = max(inner_steps, 15_000_000) if total_episodes is not None else inner_steps
    obs, _ = env.reset(seed=seed)
    t = 0
    while t < step_cap and (total_episodes is None or len(episode_returns) < total_episodes):
        n_ep = len(episode_returns)
        eps = _linear_schedule(base.START_E, eps_end, base.EPSILON_DECAY_EPISODES, n_ep)
        action = agent.act(obs, eps)

        next_obs, reward, terminated, truncated, infos = env.step(action)
        done = terminated or truncated
        real_nxt = next_obs.copy()
        if truncated and "final_observation" in infos:
            real_nxt = infos["final_observation"]
        rb.add(obs, real_nxt, action, float(reward), float(done))

        if "episode" in infos:
            ret = float(np.asarray(infos["episode"]["r"]).item())
            episode_returns.append(ret)
            if not return_ema_ready:
                return_ema = ret
                prev_return_ema = ret
                return_ema_ready = True
            else:
                return_ema = (1 - base.RETURN_EMA_ALPHA) * return_ema + base.RETURN_EMA_ALPHA * ret

        obs = next_obs
        if done:
            obs, _ = env.reset()

        steps_since_meta += 1
        do_meta = use_controller and (steps_since_meta >= base.META_INTERVAL or t == 0)
        if do_meta:
            steps_since_meta = 0
            if pending_log_prob is not None and return_ema_ready:
                r = (return_ema - prev_return_ema) - base.STEP_COST
                loss_v = _controller_step(
                    pending_log_prob,
                    pending_value,  # type: ignore[arg-type]
                    pending_entropy,  # type: ignore[arg-type]
                    r,
                    controller,
                    opt_ctrl,
                    update_controller=update_controller,
                )
                if update_controller:
                    ctrl_loss_sum += loss_v
                    ctrl_loss_count += 1
                pending_log_prob = None
                pending_value = None
                pending_entropy = None
                prev_return_ema = return_ema
            elif pending_log_prob is not None and not return_ema_ready:
                pending_log_prob = None
                pending_value = None
                pending_entropy = None

            progress = (
                len(episode_returns) / max(1, total_episodes)
                if total_episodes is not None
                else t / max(1, inner_steps)
            )
            state = _build_meta_state(
                loss_ema, grad_ema, return_ema, eps, progress,
                lr_mult, train_freq, target_freq, device,
            )
            if update_controller:
                meta_act, h = controller.act(state, task_id_t, h, deterministic=deterministic_ctrl)
                pending_log_prob = meta_act.log_prob
                pending_value = meta_act.value
                pending_entropy = meta_act.entropy
            else:
                with torch.no_grad():
                    meta_act, h = controller.act(state, task_id_t, h, deterministic=deterministic_ctrl)
            lr_mult = meta_act.lr_mult
            eps_end = meta_act.eps_end
            train_freq = meta_act.train_freq
            target_freq = meta_act.target_freq
            agent.set_lr(lr_mult)
            n_meta_actions += 1

            if t % (base.META_INTERVAL * 10) == 0:
                snap = {
                    "step": t,
                    "episode": len(episode_returns),
                    "return_ema": return_ema,
                }
                for key in active_keys:
                    snap[key] = {"train_freq": train_freq, "target_freq": target_freq}[key]
                knob_history.append(snap)

        if t > base.LEARNING_STARTS and t % max(1, train_freq) == 0 and len(rb) >= base.BATCH_SIZE:
            loss = agent.td_update(rb.sample(base.BATCH_SIZE))
            loss_ema = loss if loss_ema == 0.0 else 0.9 * loss_ema + 0.1 * loss
            grad_ema = (
                agent.last_grad_norm if grad_ema == 0.0 else 0.9 * grad_ema + 0.1 * agent.last_grad_norm
            )

        if t % max(1, target_freq) == 0:
            agent.sync_target()

        t += 1

    env.close()

    if use_controller and pending_log_prob is not None and return_ema_ready:
        r = (return_ema - prev_return_ema) - base.STEP_COST
        loss_v = _controller_step(
            pending_log_prob,
            pending_value,  # type: ignore[arg-type]
            pending_entropy,  # type: ignore[arg-type]
            r,
            controller,
            opt_ctrl,
            update_controller=update_controller,
        )
        if update_controller:
            ctrl_loss_sum += loss_v
            ctrl_loss_count += 1

    last100 = (
        float(np.mean(episode_returns[-100:]))
        if len(episode_returns) >= 100
        else float(np.mean(episode_returns))
        if episode_returns
        else 0.0
    )
    return {
        "env_id": env_id,
        "task_idx": task_idx,
        "episode_returns": episode_returns,
        "knob_history": knob_history,
        "last100_mean": last100,
        "ctrl_loss": ctrl_loss_sum / max(1, ctrl_loss_count),
        "n_meta_steps": n_meta_actions,
        "use_controller": use_controller,
    }


def run_meta_train(
    tasks: Sequence[str],
    meta_iters: int,
    inner_steps: int,
    seed: int,
    device: torch.device,
    checkpoint_dir: Path,
    knob_mode: str,
) -> Dict:
    _set_seed(seed)
    n_tasks = len(tasks)
    controller = MetaHPOControllerFreqAblated(n_tasks=n_tasks, knob_mode=knob_mode).to(device)
    opt_ctrl = optim.Adam(controller.parameters(), lr=base.CTRL_LR)

    history: List[Dict] = []

    print("=" * 60, flush=True)
    print(f"confgate_meta_hpo_onetwoparams — {knob_mode}", flush=True)
    print(f"  Tasks       : {list(tasks)}", flush=True)
    print(f"  Meta iters  : {meta_iters}", flush=True)
    print(f"  Inner steps : {inner_steps:,}", flush=True)
    print(f"  Seed        : {seed}", flush=True)
    print(f"  Device      : {device}", flush=True)
    print("=" * 60, flush=True)

    t0 = time.perf_counter()
    task_order = list(range(n_tasks))
    random.shuffle(task_order)
    for meta_i in range(meta_iters):
        if meta_i % n_tasks == 0 and meta_i > 0:
            random.shuffle(task_order)
        task_idx = task_order[meta_i % n_tasks]
        env_id = tasks[task_idx]
        inner_seed = seed * 1000 + meta_i

        result = run_inner_task(
            env_id=env_id,
            task_idx=task_idx,
            controller=controller,
            opt_ctrl=opt_ctrl,
            device=device,
            inner_steps=inner_steps,
            seed=inner_seed,
            knob_mode=knob_mode,
            use_controller=True,
        )
        history.append(
            {
                "meta_iter": meta_i,
                "env_id": env_id,
                "last100_mean": result["last100_mean"],
                "ctrl_loss": result["ctrl_loss"],
                "n_episodes": len(result["episode_returns"]),
                "n_meta_steps": result["n_meta_steps"],
                "knob_history": result["knob_history"],
            }
        )
        print(
            f"  [meta {meta_i+1}/{meta_iters}] {env_id:16s}  "
            f"eps={len(result['episode_returns']):4d}  "
            f"last100={result['last100_mean']:7.1f}  "
            f"ctrl_loss={result['ctrl_loss']:.4f}",
            flush=True,
        )

    elapsed = time.perf_counter() - t0
    print(f"Meta-train done in {elapsed:.1f}s", flush=True)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / f"meta_controller_seed{seed}.pt"
    torch.save(
        {
            "controller": controller.state_dict(),
            "knob_mode": knob_mode,
            "tasks": list(tasks),
            "seed": seed,
            "meta_iters": meta_iters,
            "inner_steps": inner_steps,
            "history": history,
        },
        ckpt_path,
    )
    print(f"  Checkpoint: {ckpt_path}", flush=True)

    summary_path = checkpoint_dir / f"meta_train_summary_seed{seed}.json"
    summary_path.write_text(
        json.dumps(
            {
                "knob_mode": knob_mode,
                "seed": seed,
                "tasks": list(tasks),
                "meta_iters": meta_iters,
                "inner_steps": inner_steps,
                "history": [{k: v for k, v in h.items() if k != "knob_history"} for h in history],
            },
            indent=2,
        )
    )

    return {"controller": controller, "history": history, "ckpt_path": ckpt_path}


def _load_reused_baseline(seed: int, env_id: str) -> Dict:
    path = MAIN_BASELINE_DIR / f"eval_vs_baseline_seed{seed}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing main baseline checkpoint: {path}")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if env_id not in blob["results"] or "baseline" not in blob["results"][env_id]:
        raise KeyError(f"No baseline for {env_id} in {path}")
    return blob["results"][env_id]["baseline"]


def _merge_reused_baselines(
    results: Dict[str, Dict[str, Dict]], tasks: Sequence[str], seed: int
) -> Dict[str, Dict[str, Dict]]:
    out = {t: dict(results[t]) for t in tasks if t in results}
    for env_id in tasks:
        out.setdefault(env_id, {})
        out[env_id]["baseline"] = _load_reused_baseline(seed, env_id)
    return out


def run_eval_vs_baseline(
    tasks: Sequence[str],
    controller: MetaHPOControllerFreqAblated,
    inner_steps: int,
    seed: int,
    device: torch.device,
    checkpoint_dir: Path,
    knob_mode: str,
    total_episodes: Optional[int] = None,
    reuse_main_baseline: bool = False,
) -> Dict[str, Dict[str, Dict]]:
    results: Dict[str, Dict[str, Dict]] = {t: {} for t in tasks}
    fixed = {
        "lr_mult": 1.0,
        "eps_end": base.END_E,
        "train_freq": base.DEFAULT_TRAIN_FREQ,
        "target_freq": base.DEFAULT_TARGET_FREQ,
    }
    frozen_opt = optim.Adam(controller.parameters(), lr=0.0)

    for task_idx, env_id in enumerate(tasks):
        if reuse_main_baseline:
            print(f"\n>>> EVAL baseline  {env_id}  (reused from {MAIN_BASELINE_DIR.name})", flush=True)
            base_res = _load_reused_baseline(seed, env_id)
        else:
            print(f"\n>>> EVAL baseline  {env_id}", flush=True)
            base_res = run_inner_task(
                env_id=env_id,
                task_idx=task_idx,
                controller=controller,
                opt_ctrl=frozen_opt,
                device=device,
                inner_steps=inner_steps,
                seed=seed + 17 + task_idx,
                knob_mode=knob_mode,
                use_controller=False,
                update_controller=False,
                fixed_knobs=fixed,
                total_episodes=total_episodes,
            )
        results[env_id]["baseline"] = base_res

        print(f">>> EVAL meta-ctrl {env_id}", flush=True)
        meta_res = run_inner_task(
            env_id=env_id,
            task_idx=task_idx,
            controller=controller,
            opt_ctrl=frozen_opt,
            device=device,
            inner_steps=inner_steps,
            seed=seed + 17 + task_idx,
            knob_mode=knob_mode,
            use_controller=True,
            update_controller=False,
            deterministic_ctrl=True,
            total_episodes=total_episodes,
        )
        results[env_id]["meta"] = meta_res

        print(
            f"  {env_id}: baseline last100={base_res['last100_mean']:.1f}"
            f"  meta last100={meta_res['last100_mean']:.1f}",
            flush=True,
        )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    eval_path = checkpoint_dir / f"eval_vs_baseline_seed{seed}.pt"
    serializable = {
        env_id: {
            tag: {
                "episode_returns": res["episode_returns"],
                "last100_mean": res["last100_mean"],
                "knob_history": res.get("knob_history", []),
            }
            for tag, res in tags.items()
        }
        for env_id, tags in results.items()
    }
    torch.save(
        {"seed": seed, "knob_mode": knob_mode, "tasks": list(tasks), "results": serializable},
        eval_path,
    )
    print(f"  Eval checkpoint: {eval_path}", flush=True)
    return results


def plot_knob_trajectories(
    results: Dict[str, Dict[str, Dict]],
    knob_mode: str,
    out_jpg: Path,
) -> None:
    keys = _active_knob_keys(knob_mode)
    rows = []
    for env_id, tags in results.items():
        hist = tags.get("meta", {}).get("knob_history", [])
        if hist:
            rows.append((env_id, hist))
    if not rows:
        print("[knob plot] no knob_history — skipping.", flush=True)
        return

    fig, axes = plt.subplots(
        len(keys), len(rows), figsize=(4.5 * len(rows), 2.2 * len(keys)), squeeze=False
    )
    for col, (env_id, hist) in enumerate(rows):
        steps = [h["step"] for h in hist]
        for row, key in enumerate(keys):
            ax = axes[row][col]
            vals = [h[key] for h in hist]
            ax.plot(steps, vals, color=base.COLOR_META, lw=1.5)
            ax.set_ylabel(key, fontsize=8)
            ax.grid(True, alpha=0.3)
            if row == 0:
                ax.set_title(env_id, fontsize=10)
            if row == len(keys) - 1:
                ax.set_xlabel("Inner step", fontsize=9)
    fig.suptitle(f"Knob trajectories — {MODE_LABELS[knob_mode]}", fontsize=12)
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_jpg, dpi=150, bbox_inches="tight", format="jpeg")
    plt.close(fig)
    print(f"Wrote {out_jpg}", flush=True)


def _interp_hist(hist: List[Dict], x_key: str, y_key: str, grid: np.ndarray) -> np.ndarray:
    xs = np.array([float(h[x_key]) for h in hist])
    ys = np.array([float(h[y_key]) for h in hist])
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    ux: List[float] = []
    uy: List[float] = []
    for x, y in zip(xs, ys):
        if ux and abs(x - ux[-1]) < 1e-9:
            uy[-1] = float(y)
        else:
            ux.append(float(x))
            uy.append(float(y))
    return np.interp(grid, np.asarray(ux), np.asarray(uy), left=float("nan"), right=float("nan"))


def _plot_knob_trajectories_multi_seed(
    all_results: List[Dict[str, Dict[str, Dict]]],
    out_jpg: Path,
    knob_mode: str = "both_freq",
) -> None:
    keys = _active_knob_keys(knob_mode)
    tasks = list(all_results[0].keys())
    color = MODE_COLORS.get(knob_mode, base.COLOR_META)
    fig, axes = plt.subplots(
        len(keys), len(tasks), figsize=(4.5 * len(tasks), 2.2 * len(keys)), squeeze=False
    )
    for col, env_id in enumerate(tasks):
        hists = [
            r[env_id]["meta"].get("knob_history", [])
            for r in all_results
            if env_id in r and "meta" in r[env_id] and r[env_id]["meta"].get("knob_history")
        ]
        if not hists:
            continue
        starts = [float(h[0]["step"]) for h in hists]
        ends = [float(h[-1]["step"]) for h in hists]
        lo, hi = max(starts), min(ends)
        if hi <= lo:
            continue
        grid = np.linspace(lo, hi, 200)
        for row, key in enumerate(keys):
            ax = axes[row][col]
            rows = np.vstack([_interp_hist(h, "step", key, grid) for h in hists])
            mu = np.nanmean(rows, axis=0)
            for row_vals in rows:
                ax.plot(grid, row_vals, color=color, alpha=0.35, lw=0.9, zorder=1)
            ax.plot(grid, mu, color=color, lw=2.5, zorder=2)
            ax.set_ylabel(key, fontsize=8)
            ax.grid(True, alpha=0.3)
            if row == 0:
                ax.set_title(env_id, fontsize=10)
            if row == len(keys) - 1:
                ax.set_xlabel("Inner step", fontsize=9)
    n = len(all_results)
    fig.suptitle(
        f"Knob trajectories — {MODE_LABELS[knob_mode]} ({n} seeds, faint=individual, bold=mean)",
        fontsize=12,
    )
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_jpg, dpi=150, bbox_inches="tight", format="jpeg")
    plt.close(fig)
    print(f"Wrote {out_jpg}", flush=True)


def plot_baseline_vs_meta_multi_seed(
    all_results: List[Dict[str, Dict[str, Dict]]],
    out_jpg: Path,
    meta_color: str,
    meta_label: str,
    title_suffix: str = "",
) -> None:
    """Per-seed curves (semi-transparent) + bold mean for baseline vs meta."""
    if not all_results:
        return
    tasks = list(all_results[0].keys())
    n_seeds = len(all_results)
    fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 4), squeeze=False)

    for ax, env_id in zip(axes[0], tasks):
        for tag, color, label in (
            ("baseline", base.COLOR_BASE, "fixed hparams"),
            ("meta", meta_color, meta_label),
        ):
            arrays = [
                np.asarray(r[env_id][tag]["episode_returns"], dtype=np.float64)
                for r in all_results
                if env_id in r and tag in r[env_id]
            ]
            if not arrays:
                continue
            n = min(len(a) for a in arrays)
            M = np.stack([a[:n] for a in arrays], axis=0)
            mu = M.mean(0)
            w = max(1, n // 50)
            ep_sm = np.arange(w, n + 1)
            sm_mu = _smooth(mu, w)
            l100 = float(np.mean([r[env_id][tag]["last100_mean"] for r in all_results if env_id in r and tag in r[env_id]]))
            for arr in arrays:
                sm = _smooth(arr[:n], w)
                ax.plot(ep_sm, sm, color=color, alpha=0.35, lw=1.0, zorder=1)
            ax.plot(
                ep_sm,
                sm_mu,
                color=color,
                lw=3.0,
                label=f"{label} mean (last100={l100:.1f})",
                zorder=2,
            )
        ax.set_title(env_id, fontsize=11)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episodic return")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        f"Fixed hparams vs {meta_label} ({n_seeds} seeds, faint=individual, bold=mean){title_suffix}",
        fontsize=12,
        y=1.02,
    )
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_jpg, dpi=150, bbox_inches="tight", format="jpeg")
    plt.close(fig)
    print(f"Wrote {out_jpg}", flush=True)


def plot_mode_comparison(
    mode_results: Dict[str, Dict[str, Dict[str, Dict]]],
    out_jpg: Path,
    budget_tag: str,
) -> None:
    """Baseline vs each ablated meta-controller on one CartPole panel."""
    env_id = next(iter(next(iter(mode_results.values())).keys()))
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)

    base_arr = np.asarray(mode_results[KNOB_MODES[0]][env_id]["baseline"]["episode_returns"])
    w = max(1, len(base_arr) // 50)
    ep = np.arange(w, len(base_arr) + 1)
    ax.plot(ep, _smooth(base_arr, w), color=base.COLOR_BASE, lw=2.5,
            label=f"fixed hparams (last100={mode_results[KNOB_MODES[0]][env_id]['baseline']['last100_mean']:.1f})")

    for mode in KNOB_MODES:
        rets = np.asarray(mode_results[mode][env_id]["meta"]["episode_returns"])
        n = min(len(rets), len(base_arr))
        sm = _smooth(rets[:n], w)
        l100 = mode_results[mode][env_id]["meta"]["last100_mean"]
        ax.plot(
            ep[: len(sm)],
            sm,
            color=MODE_COLORS[mode],
            lw=2.0,
            label=f"{MODE_LABELS[mode]} (last100={l100:.1f})",
        )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Episodic return")
    ax.set_title(f"CartPole-v1 — freq-knob ablations ({budget_tag}, seed=1)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_jpg, dpi=150, bbox_inches="tight", format="jpeg")
    plt.close(fig)
    print(f"Wrote {out_jpg}", flush=True)


def _ckpt_dir(knob_mode: str) -> Path:
    return CKPT_ROOT / knob_mode


def run_one_mode(
    knob_mode: str,
    tasks: Sequence[str],
    seed: int,
    device: torch.device,
    meta_iters: int,
    inner_steps: int,
    total_episodes: Optional[int],
    eval_only: bool,
    plot_only: bool,
    train_missing_only: bool,
    reuse_main_baseline: bool = False,
    aggregate_only: bool = False,
) -> Dict[str, Dict[str, Dict]]:
    ckpt_dir = _ckpt_dir(knob_mode)
    ckpt_path = ckpt_dir / f"meta_controller_seed{seed}.pt"
    eval_path = ckpt_dir / f"eval_vs_baseline_seed{seed}.pt"
    budget_tag = f"ep{total_episodes}" if total_episodes else f"inner{inner_steps}"

    if plot_only:
        if not eval_path.exists():
            raise FileNotFoundError(f"Missing eval checkpoint: {eval_path}")
        blob = torch.load(eval_path, map_location="cpu", weights_only=False)
        results = _filter_results(blob["results"], tasks)
        if reuse_main_baseline:
            results = _merge_reused_baselines(results, tasks, seed)
        if not aggregate_only:
            plot_eval_curves(
                results,
                FIG_DIR / f"meta_vs_baseline_{knob_mode}_seed{seed}.jpg",
                title_suffix=f" ({MODE_LABELS[knob_mode]}, {budget_tag})",
            )
            plot_knob_trajectories(
                results, knob_mode, FIG_DIR / f"knob_trajectories_{knob_mode}_seed{seed}.jpg"
            )
        return results

    controller: Optional[MetaHPOControllerFreqAblated] = None
    if not eval_only:
        if train_missing_only and ckpt_path.exists():
            print(f"Controller present, skipping meta-train: {ckpt_path}", flush=True)
            blob = torch.load(ckpt_path, map_location=device, weights_only=False)
            controller = MetaHPOControllerFreqAblated(
                n_tasks=len(blob["tasks"]), knob_mode=knob_mode
            ).to(device)
            controller.load_state_dict(blob["controller"])
            tasks = blob.get("tasks", tasks)
        else:
            train_out = run_meta_train(
                tasks=tasks,
                meta_iters=meta_iters,
                inner_steps=inner_steps,
                seed=seed,
                device=device,
                checkpoint_dir=ckpt_dir,
                knob_mode=knob_mode,
            )
            controller = train_out["controller"]
    else:
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing controller checkpoint: {ckpt_path}")
        blob = torch.load(ckpt_path, map_location=device, weights_only=False)
        tasks = blob.get("tasks", tasks)
        controller = MetaHPOControllerFreqAblated(n_tasks=len(tasks), knob_mode=knob_mode).to(device)
        controller.load_state_dict(blob["controller"])

    results = run_eval_vs_baseline(
        tasks=tasks,
        controller=controller,
        inner_steps=inner_steps,
        seed=seed,
        device=device,
        checkpoint_dir=ckpt_dir,
        knob_mode=knob_mode,
        total_episodes=total_episodes,
        reuse_main_baseline=reuse_main_baseline,
    )
    if reuse_main_baseline:
        results = _merge_reused_baselines(results, tasks, seed)
        eval_path = ckpt_dir / f"eval_vs_baseline_seed{seed}.pt"
        serializable = {
            env_id: {
                tag: {
                    "episode_returns": res["episode_returns"],
                    "last100_mean": res["last100_mean"],
                    "knob_history": res.get("knob_history", []),
                }
                for tag, res in tags.items()
            }
            for env_id, tags in results.items()
        }
        torch.save(
            {
                "seed": seed,
                "knob_mode": knob_mode,
                "tasks": list(tasks),
                "reused_main_baseline": True,
                "results": serializable,
            },
            eval_path,
        )
    if not aggregate_only:
        plot_eval_curves(
            results,
            FIG_DIR / f"meta_vs_baseline_{knob_mode}_seed{seed}.jpg",
            title_suffix=f" ({MODE_LABELS[knob_mode]}, {budget_tag})",
        )
        plot_knob_trajectories(
            results, knob_mode, FIG_DIR / f"knob_trajectories_{knob_mode}_seed{seed}.jpg"
        )
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--knob-mode",
        type=str,
        choices=KNOB_MODES,
        default=None,
        help="Which frequency knob(s) the meta-controller may adjust",
    )
    p.add_argument(
        "--run-all",
        action="store_true",
        help="Run all three ablations (train_freq / target_freq / both)",
    )
    p.add_argument("--tasks", type=str, default=",".join(DEFAULT_TASKS))
    p.add_argument("--meta-iters", type=int, default=DEFAULT_META_ITERS)
    p.add_argument("--inner-steps", type=int, default=DEFAULT_INNER_STEPS)
    p.add_argument("--total-episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    p.add_argument("--seeds", type=str, default=",".join(map(str, DEFAULT_SEEDS)))
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--train-missing-only", action="store_true")
    p.add_argument(
        "--reuse-main-baseline",
        action="store_true",
        help="Load fixed-hparam baseline curves from ckpt_meta_hpo (skip baseline re-eval)",
    )
    p.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Skip per-seed figures; write multi-seed aggregate plots only",
    )
    args = p.parse_args()

    if not args.run_all and args.knob_mode is None:
        p.error("Specify --knob-mode or --run-all")

    tasks = _parse_list(args.tasks)
    seeds = _parse_seeds(args.seeds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    modes = list(KNOB_MODES) if args.run_all else [args.knob_mode]
    budget_tag = f"ep{args.total_episodes}" if args.total_episodes else f"inner{args.inner_steps}"
    all_mode_results: Dict[str, Dict[str, Dict[str, Dict]]] = {}
    all_eval_results: List[Dict[str, Dict[str, Dict]]] = []

    for mode in modes:
        for seed in seeds:
            results = run_one_mode(
                knob_mode=mode,
                tasks=tasks,
                seed=seed,
                device=device,
                meta_iters=args.meta_iters,
                inner_steps=args.inner_steps,
                total_episodes=args.total_episodes,
                eval_only=args.eval_only,
                plot_only=args.plot_only,
                train_missing_only=args.train_missing_only,
                reuse_main_baseline=args.reuse_main_baseline,
                aggregate_only=args.aggregate_only,
            )
            all_mode_results[mode] = results
            if not args.run_all or mode == "both_freq":
                all_eval_results.append(results)

    if args.run_all and len(seeds) == 1:
        plot_mode_comparison(
            all_mode_results,
            FIG_DIR / f"mode_comparison_{budget_tag}_seed{seeds[0]}.jpg",
            budget_tag=budget_tag,
        )

    if len(all_eval_results) > 1 and (not args.run_all and args.knob_mode == "both_freq"):
        n_tag = f"{len(all_eval_results)}seeds"
        plot_baseline_vs_meta_multi_seed(
            all_eval_results,
            FIG_DIR / f"meta_vs_baseline_both_freq_{budget_tag}_{n_tag}.jpg",
            meta_color=MODE_COLORS["both_freq"],
            meta_label=MODE_LABELS["both_freq"],
            title_suffix=f" ({budget_tag})",
        )
        _plot_knob_trajectories_multi_seed(
            all_eval_results,
            FIG_DIR / f"knob_trajectories_both_freq_{budget_tag}_{n_tag}.jpg",
            knob_mode="both_freq",
        )


if __name__ == "__main__":
    main()
