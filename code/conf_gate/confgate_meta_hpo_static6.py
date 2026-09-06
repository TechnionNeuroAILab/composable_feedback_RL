#!/usr/bin/env python3
"""Staged six-knob HPO experiment with one static configuration per DQN run.

The controller samples all six hyperparameters once, holds them fixed for the
whole inner run, and receives a whole-run objective:

    0.7 * normalized learning-curve AUC + 0.3 * normalized last-100 return.

This removes the short-horizon credit-assignment problem in the dynamic
controller and provides a stable first stage before introducing live schedules.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import confgate_meta_hpo as base


ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "paper" / "figures" / "conf_meta_hpo_v2"
CKPT_DIR = ROOT / "paper" / "_tmp_b_feedb_cg" / "ckpt_meta_hpo"

CTRL_LR = 3e-4
VALUE_COEF = 0.5
ENTROPY_START = 0.01
ENTROPY_END = 0.001
CARTPOLE_MAX_RETURN = 500.0

# Restrict the original categorical grids to plausible CartPole values.
ALLOWED = {
    "lr_mult": [1, 2, 3, 4, 5, 6],  # 0.25 ... 1.5
    "eps_end": [0, 2, 3, 5],  # 0.01, 0.03, 0.05, 0.10
    "eps_start": [6, 7, 8, 9],  # 0.7 ... 1.0
    "epsilon_decay_episodes": [2, 3, 4, 5, 6, 7, 8],  # 200 ... 2000
    "train_freq": [0, 1, 2, 3, 4],
    "target_freq": [1, 2, 3, 4],  # exclude target sync every step
}

FIXED_DEFAULTS = {
    "lr_mult": 1.0,
    "eps_end": base.END_E,
    "eps_start": base.START_E,
    "epsilon_decay_episodes": float(base.EPSILON_DECAY_EPISODES),
    "train_freq": float(base.DEFAULT_TRAIN_FREQ),
    "target_freq": float(base.DEFAULT_TARGET_FREQ),
}

# Clearly different from FIXED_DEFAULTS; used to bias the controller at init.
INIT_PRESETS: Dict[str, Dict[str, float]] = {
    "defaults": dict(FIXED_DEFAULTS),
    "alt": {
        "lr_mult": 0.5,
        "eps_end": 0.01,
        "eps_start": 0.7,
        "epsilon_decay_episodes": 200.0,
        "train_freq": 1.0,
        "target_freq": 50.0,
    },
    # Suboptimal for CartPole: high LR, weak/fast ε-schedule, train every step.
    "bad": {
        "lr_mult": 1.5,
        "eps_end": 0.1,
        "eps_start": 0.7,
        "epsilon_decay_episodes": 200.0,
        "train_freq": 1.0,
        "target_freq": 50.0,
    },
}


def _choice_lists() -> Dict[str, Sequence[float]]:
    return {
        "lr_mult": base.LR_MULT_CHOICES,
        "eps_end": base.EPS_END_CHOICES,
        "eps_start": base.EPS_START_CHOICES,
        "epsilon_decay_episodes": base.EPS_DECAY_EPISODES_CHOICES,
        "train_freq": base.TRAIN_FREQ_CHOICES,
        "target_freq": base.TARGET_FREQ_CHOICES,
    }


def _init_controller_bias(
    controller: base.MetaHPOController,
    knobs: Dict[str, float],
    bias: float = 1.0,
) -> None:
    """Bias categorical heads toward a starting six-knob configuration."""
    heads = {
        "lr_mult": controller.lr_head,
        "eps_end": controller.eps_end_head,
        "eps_start": controller.eps_start_head,
        "epsilon_decay_episodes": controller.eps_decay_head,
        "train_freq": controller.tf_head,
        "target_freq": controller.tgt_head,
    }
    choices = _choice_lists()
    with torch.no_grad():
        for key, head in heads.items():
            idx = base._nearest_choice_idx(choices[key], knobs[key])
            head.bias.zero_()
            head.bias[idx] = bias


def _pick(
    logits: torch.Tensor, allowed: Sequence[int], deterministic: bool
) -> Tuple[int, torch.Tensor, torch.Tensor]:
    """Sample from a masked categorical head and return original-grid index."""
    allowed_t = torch.tensor(allowed, dtype=torch.long, device=logits.device)
    dist = Categorical(logits=logits.index_select(-1, allowed_t))
    local_idx = dist.probs.argmax(-1) if deterministic else dist.sample()
    original_idx = int(allowed_t[local_idx[0]].item())
    return original_idx, dist.log_prob(local_idx), dist.entropy()


def select_static_config(
    controller: base.MetaHPOController,
    device: torch.device,
    deterministic: bool = False,
    state_knobs: Optional[Dict[str, float]] = None,
) -> base.MetaAction:
    """Select one constrained six-parameter configuration."""
    knobs = state_knobs or FIXED_DEFAULTS
    state = base.build_meta_state(
        loss_ema=0.0,
        grad_ema=0.0,
        return_ema=0.0,
        eps=knobs["eps_start"],
        progress=0.0,
        lr_mult=knobs["lr_mult"],
        eps_end=knobs["eps_end"],
        eps_start=knobs["eps_start"],
        epsilon_decay_episodes=int(knobs["epsilon_decay_episodes"]),
        train_freq=int(knobs["train_freq"]),
        target_freq=int(knobs["target_freq"]),
        device=device,
    )
    task_id = torch.zeros(1, dtype=torch.long, device=device)
    h = controller.gru(
        torch.cat([state, controller.task_emb(task_id)], dim=-1),
        controller.zero_hidden(1, device),
    )
    feat = controller.trunk(h)

    specs = (
        ("lr_mult", controller.lr_head(feat)),
        ("eps_end", controller.eps_end_head(feat)),
        ("eps_start", controller.eps_start_head(feat)),
        ("epsilon_decay_episodes", controller.eps_decay_head(feat)),
        ("train_freq", controller.tf_head(feat)),
        ("target_freq", controller.tgt_head(feat)),
    )
    picked = {
        name: _pick(logits, ALLOWED[name], deterministic) for name, logits in specs
    }
    log_prob = sum(v[1] for v in picked.values())
    entropy = sum(v[2] for v in picked.values())
    value = controller.value_head(feat).squeeze(-1)

    return base.MetaAction(
        lr_mult=float(base.LR_MULT_CHOICES[picked["lr_mult"][0]]),
        eps_end=float(base.EPS_END_CHOICES[picked["eps_end"][0]]),
        eps_start=float(base.EPS_START_CHOICES[picked["eps_start"][0]]),
        epsilon_decay_episodes=int(
            base.EPS_DECAY_EPISODES_CHOICES[picked["epsilon_decay_episodes"][0]]
        ),
        train_freq=int(base.TRAIN_FREQ_CHOICES[picked["train_freq"][0]]),
        target_freq=int(base.TARGET_FREQ_CHOICES[picked["target_freq"][0]]),
        log_prob=log_prob,
        entropy=entropy,
        value=value,
    )


def _knobs(action: base.MetaAction) -> Dict[str, float]:
    return {
        "lr_mult": action.lr_mult,
        "eps_end": action.eps_end,
        "eps_start": action.eps_start,
        "epsilon_decay_episodes": float(action.epsilon_decay_episodes),
        "train_freq": float(action.train_freq),
        "target_freq": float(action.target_freq),
    }


def _score(returns: Sequence[float]) -> Tuple[float, float, float]:
    """Return combined objective, normalized AUC, and normalized last-100."""
    arr = np.asarray(returns, dtype=np.float64)
    auc = float(np.clip(arr.mean() / CARTPOLE_MAX_RETURN, 0.0, 1.0))
    final = float(np.clip(arr[-100:].mean() / CARTPOLE_MAX_RETURN, 0.0, 1.0))
    return 0.7 * auc + 0.3 * final, auc, final


def train_controller(
    seed: int,
    meta_updates: int,
    batch_runs: int,
    train_episodes: int,
    device: torch.device,
    init_knobs: Dict[str, float],
) -> Tuple[base.MetaHPOController, List[Dict], Dict[str, float]]:
    base._set_seed(seed)
    controller = base.MetaHPOController(n_tasks=1).to(device)
    _init_controller_bias(controller, init_knobs, bias=1.0)
    optimizer = optim.Adam(controller.parameters(), lr=CTRL_LR)
    frozen_opt = optim.Adam(controller.parameters(), lr=0.0)
    history: List[Dict] = []
    best_overall = {"score": -1.0, "knobs": dict(init_knobs)}

    for update in range(meta_updates):
        frac = update / max(1, meta_updates - 1)
        entropy_coef = ENTROPY_START + frac * (ENTROPY_END - ENTROPY_START)
        actions: List[base.MetaAction] = []
        scores: List[float] = []
        aucs: List[float] = []
        finals: List[float] = []

        for run_idx in range(batch_runs):
            inner_seed = seed * 100_000 + update * batch_runs + run_idx
            base._set_seed(inner_seed)
            action = select_static_config(controller, device, deterministic=False)
            result = base.run_inner_task(
                env_id="CartPole-v1",
                task_idx=0,
                controller=controller,
                opt_ctrl=frozen_opt,
                device=device,
                inner_steps=50_000,
                seed=inner_seed,
                use_controller=False,
                update_controller=False,
                fixed_knobs=_knobs(action),
                total_episodes=train_episodes,
            )
            score, auc, final = _score(result["episode_returns"])
            knobs = _knobs(action)
            actions.append(action)
            scores.append(score)
            aucs.append(auc)
            finals.append(final)
            if score > best_overall["score"]:
                best_overall = {"score": score, "knobs": knobs}

        score_t = torch.tensor(scores, dtype=torch.float32, device=device)
        values = torch.stack([a.value.view(-1)[0] for a in actions])
        log_probs = torch.stack([a.log_prob.view(-1)[0] for a in actions])
        entropies = torch.stack([a.entropy.view(-1)[0] for a in actions])
        advantages = score_t - values.detach()
        if batch_runs > 1:
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8
            )

        policy_loss = -(advantages * log_probs).mean()
        value_loss = F.mse_loss(values, score_t)
        loss = (
            policy_loss
            + VALUE_COEF * value_loss
            - entropy_coef * entropies.mean()
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(controller.parameters(), 5.0)
        optimizer.step()

        deterministic = select_static_config(controller, device, deterministic=True)
        rec = {
            "update": update + 1,
            "mean_score": float(np.mean(scores)),
            "mean_auc": float(np.mean(aucs)),
            "mean_final": float(np.mean(finals)),
            "score_std": float(np.std(scores)),
            "entropy_coef": entropy_coef,
            "loss": float(loss.item()),
            "samples": [_knobs(a) for a in actions],
            "sample_scores": scores,
            "deterministic": _knobs(deterministic),
            "best_in_batch": max(
                zip(scores, [_knobs(a) for a in actions]), key=lambda x: x[0]
            )[1],
            "global_best": dict(best_overall["knobs"]),
            "global_best_score": float(best_overall["score"]),
        }
        history.append(rec)
        print(
            f"[update {update + 1:02d}/{meta_updates}] "
            f"score={rec['mean_score']:.3f}±{rec['score_std']:.3f} "
            f"auc={rec['mean_auc']:.3f} final={rec['mean_final']:.3f} "
            f"config={rec['deterministic']}",
            flush=True,
        )

    return controller, history, dict(best_overall["knobs"])


def evaluate(
    controller: base.MetaHPOController,
    seed: int,
    episodes: int,
    device: torch.device,
    learned_knobs: Optional[Dict[str, float]] = None,
    init_knobs: Optional[Dict[str, float]] = None,
) -> Tuple[Dict, Dict, Dict, Dict[str, float]]:
    action = select_static_config(controller, device, deterministic=True)
    deterministic = _knobs(action)
    learned = learned_knobs or deterministic
    fixed = dict(FIXED_DEFAULTS)
    frozen_opt = optim.Adam(controller.parameters(), lr=0.0)
    eval_seed = seed + 17

    base._set_seed(eval_seed)
    baseline = base.run_inner_task(
        "CartPole-v1",
        0,
        controller,
        frozen_opt,
        device,
        50_000,
        eval_seed,
        use_controller=False,
        update_controller=False,
        fixed_knobs=fixed,
        total_episodes=episodes,
    )
    base._set_seed(eval_seed)
    meta = base.run_inner_task(
        "CartPole-v1",
        0,
        controller,
        frozen_opt,
        device,
        50_000,
        eval_seed,
        use_controller=False,
        update_controller=False,
        fixed_knobs=learned,
        total_episodes=episodes,
    )
    init_result: Dict = {}
    if init_knobs is not None and init_knobs != learned:
        base._set_seed(eval_seed)
        init_result = base.run_inner_task(
            "CartPole-v1",
            0,
            controller,
            frozen_opt,
            device,
            50_000,
            eval_seed,
            use_controller=False,
            update_controller=False,
            fixed_knobs=init_knobs,
            total_episodes=episodes,
        )
    return baseline, meta, init_result, learned


def plot_results(
    baseline: Dict,
    meta: Dict,
    learned: Dict[str, float],
    history: List[Dict],
    seed: int,
    train_episodes: int,
    eval_episodes: int,
    run_tag: str = "",
    init_knobs: Optional[Dict[str, float]] = None,
    init_eval: Optional[Dict] = None,
) -> List[Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    tag_sfx = f"_{run_tag}" if run_tag else ""
    stem = f"cartpole_seed{seed}_train{train_episodes}_eval{eval_episodes}{tag_sfx}"
    outputs: List[Path] = []

    fig, ax = plt.subplots(figsize=(8, 4.5))
    series = [
        (baseline, base.COLOR_BASE, "fixed defaults", "-", 2),
        (meta, base.COLOR_META, "best learned static config", "--", 3),
    ]
    if init_eval:
        series.insert(
            1,
            (init_eval, "#888888", "init preset (eval)", ":", 1),
        )
    for result, color, label, linestyle, zorder in series:
        arr = np.asarray(result["episode_returns"], dtype=np.float64)
        window = max(1, len(arr) // 50)
        smoothed = base._smooth(arr, window)
        ax.plot(
            np.arange(window, len(arr) + 1),
            smoothed,
            color=color,
            lw=2.3,
            linestyle=linestyle,
            zorder=zorder,
            label=f"{label} (last100={result['last100_mean']:.1f})",
        )
    baseline_returns = np.asarray(baseline["episode_returns"], dtype=np.float64)
    meta_returns = np.asarray(meta["episode_returns"], dtype=np.float64)
    if init_knobs is not None:
        init_txt = ", ".join(f"{k}={init_knobs[k]}" for k in base.KNOB_PLOT_KEYS)
        ax.text(
            0.02,
            0.97,
            f"Init knobs: {init_txt}",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
        )
    if np.array_equal(baseline_returns, meta_returns):
        ax.text(
            0.02,
            0.82 if init_knobs is not None else 0.97,
            "Curves overlap exactly (learned config equals fixed defaults)",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
        )
    ax.set(
        title=(
            f"Static six-parameter controller vs baseline "
            f"(CartPole, seed={seed}{tag_sfx})"
        ),
        xlabel="Episode",
        ylabel="Episodic return",
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    path = FIG_DIR / f"meta_static6_vs_baseline_{stem}.jpg"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    outputs.append(path)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = [h["update"] for h in history]
    means = np.asarray([h["mean_score"] for h in history])
    stds = np.asarray([h["score_std"] for h in history])
    ax.plot(xs, means, "o-", color=base.COLOR_META, lw=2, label="batch objective")
    ax.fill_between(xs, means - stds, means + stds, color=base.COLOR_META, alpha=0.2)
    ax.plot(xs, [h["mean_auc"] for h in history], "--", label="normalized AUC")
    ax.plot(xs, [h["mean_final"] for h in history], "--", label="normalized last-100")
    ax.set(
        title="Static controller meta-training objective",
        xlabel="Meta-update",
        ylabel="Normalized score",
    )
    ax.grid(alpha=0.3)
    ax.legend()
    path = FIG_DIR / f"meta_static6_progress_{stem}.jpg"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    outputs.append(path)

    keys = list(base.KNOB_PLOT_KEYS)
    fig, axes = plt.subplots(3, 2, figsize=(10, 9), squeeze=False)
    for ax, key in zip(axes.flat, keys):
        for h in history:
            x = h["update"]
            vals = [sample[key] for sample in h["samples"]]
            ax.scatter([x] * len(vals), vals, color=base.COLOR_META, alpha=0.25, s=15)
        det = [h["deterministic"][key] for h in history]
        ax.plot(xs, det, "o-", color=base.COLOR_META, lw=2, label="deterministic")
        ax.axhline(learned[key], color="black", ls=":", lw=1)
        if key == keys[0]:
            ax.axhline(
                FIXED_DEFAULTS[key],
                color=base.COLOR_BASE,
                ls="--",
                lw=1,
                alpha=0.8,
                label="fixed defaults",
            )
        else:
            ax.axhline(
                FIXED_DEFAULTS[key],
                color=base.COLOR_BASE,
                ls="--",
                lw=1,
                alpha=0.8,
            )
        if init_knobs is not None:
            if key == keys[0]:
                ax.axhline(
                    init_knobs[key],
                    color="#888888",
                    ls=":",
                    lw=1,
                    alpha=0.8,
                    label="init preset",
                )
            else:
                ax.axhline(init_knobs[key], color="#888888", ls=":", lw=1, alpha=0.8)
        ax.set_title(key)
        ax.set_xlabel("Meta-update")
        ax.grid(alpha=0.25)
    fig.suptitle("Static six-knob samples and deterministic configuration")
    fig.tight_layout()
    path = FIG_DIR / f"meta_static6_knobs_{stem}.jpg"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    outputs.append(path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--meta-updates", type=int, default=10)
    parser.add_argument("--batch-runs", type=int, default=8)
    parser.add_argument("--train-episodes", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=2000)
    parser.add_argument(
        "--init-preset",
        type=str,
        default="defaults",
        choices=sorted(INIT_PRESETS),
        help="Starting six-knob bias for controller heads (baseline eval stays fixed)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Suffix for checkpoints/figures (e.g. initalt)",
    )
    args = parser.parse_args()

    init_knobs = dict(INIT_PRESETS[args.init_preset])
    run_tag = args.tag or ("" if args.init_preset == "defaults" else args.init_preset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training static six-knob controller on {device}", flush=True)
    print(f"Init preset: {args.init_preset} -> {init_knobs}", flush=True)
    controller, history, best_knobs = train_controller(
        args.seed,
        args.meta_updates,
        args.batch_runs,
        args.train_episodes,
        device,
        init_knobs,
    )
    baseline, meta, init_eval, learned = evaluate(
        controller,
        args.seed,
        args.eval_episodes,
        device,
        learned_knobs=best_knobs,
        init_knobs=init_knobs if args.init_preset != "defaults" else None,
    )

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    tag_sfx = f"_{run_tag}" if run_tag else ""
    ckpt = CKPT_DIR / (
        f"meta_static6_cartpole_seed{args.seed}_train{args.train_episodes}{tag_sfx}.pt"
    )
    torch.save(
        {
            "controller": controller.state_dict(),
            "seed": args.seed,
            "tag": run_tag,
            "init_preset": args.init_preset,
            "init_knobs": init_knobs,
            "best_knobs": best_knobs,
            "history": history,
            "learned_config": learned,
            "baseline_last100": baseline["last100_mean"],
            "meta_last100": meta["last100_mean"],
            "init_last100": init_eval.get("last100_mean") if init_eval else None,
            "train_episodes": args.train_episodes,
            "eval_episodes": args.eval_episodes,
        },
        ckpt,
    )
    outputs = plot_results(
        baseline,
        meta,
        learned,
        history,
        args.seed,
        args.train_episodes,
        args.eval_episodes,
        run_tag=run_tag,
        init_knobs=init_knobs,
        init_eval=init_eval if init_eval else None,
    )
    print(f"Init config: {init_knobs}", flush=True)
    print(f"Best config (eval): {learned}", flush=True)
    print(
        f"Evaluation last100: baseline={baseline['last100_mean']:.1f}, "
        f"static6={meta['last100_mean']:.1f}",
        flush=True,
    )
    print(f"Checkpoint: {ckpt}", flush=True)
    for output in outputs:
        print(f"Wrote {output}", flush=True)


if __name__ == "__main__":
    main()
