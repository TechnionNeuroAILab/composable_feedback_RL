"""Draw the v4 (backprop, no B) composable PPO architecture schematic."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def _box(ax, xy, w, h, text, fc="#eef4ff", ec="#2c5282", fontsize=9, bold=False, ha="center"):
    patch = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.4,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    weight = "bold" if bold else "normal"
    tx = xy[0] + w / 2 if ha == "center" else xy[0] + 0.12
    ax.text(tx, xy[1] + h / 2, text, ha=ha, va="center", fontsize=fontsize, weight=weight, linespacing=1.5)
    return patch


def _arrow(ax, start, end, color="#444444", style="-|>", lw=1.3, connection="arc3,rad=0.0", dashed=False):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=13,
        linewidth=lw,
        color=color,
        connectionstyle=connection,
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(arrow)


def _label(ax, xy, text, color="#234e52", fontsize=7.5):
    ax.text(
        xy[0],
        xy[1],
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=color,
        style="italic",
        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85),
    )


def plot_v4_architecture(output_path: Path, num_channels: int = 4, num_modules: int = 4) -> None:
    fig, ax = plt.subplots(figsize=(16, 11))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 11)
    ax.axis("off")

    ax.text(
        8,
        10.6,
        "Composable PPO v4 — backprop routing (no matrix B)",
        ha="center",
        va="center",
        fontsize=15,
        weight="bold",
    )
    ax.text(
        8,
        10.15,
        f"HalfCheetah-v4 preset: K={num_channels} value channels, M={num_modules} shared encoder modules",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#555555",
    )

    fwd = "#3a3a3a"  # solid forward-pass arrow color

    # ---- Row 1: forward pass through the shared network -----------------
    obs = (0.4, 8.5, 1.5, 0.8)
    _box(ax, obs[:2], obs[2], obs[3], "Observation\n$s_t$", fc="#fff8e6", ec="#b7791f")

    enc = (2.3, 8.0, 2.4, 1.7)
    _box(
        ax,
        enc[:2],
        enc[2],
        enc[3],
        "ModularEncoder\n$M$ parallel MLP branches\n$\\rightarrow z_1, \\ldots, z_M$",
        fc="#ebf8ff",
        ec="#2b6cb0",
        bold=True,
    )

    concat = (5.1, 8.5, 2.0, 0.8)
    _box(ax, concat[:2], concat[2], concat[3], "Shared features\n$[z_1;\\ldots;z_M]$", fc="#faf5ff", ec="#6b46c1")

    actor = (7.5, 9.15, 2.1, 0.75)
    _box(ax, actor[:2], actor[2], actor[3], "SharedActor\nMLP + Gaussian head", fc="#fff5f5", ec="#c53030", bold=True, fontsize=8.5)

    policy = (9.9, 9.15, 1.6, 0.75)
    _box(ax, policy[:2], policy[2], policy[3], r"$\pi(a\,|\,s)$", fc="#fff5f5", ec="#c53030")

    critics = (7.5, 7.95, 2.1, 0.75)
    _box(
        ax,
        critics[:2],
        critics[2],
        critics[3],
        f"$K$={num_channels} Critics $V_1,\\ldots,V_K$\nsame input, separate heads",
        fc="#f0fff4",
        ec="#276749",
        bold=True,
        fontsize=8.5,
    )

    values = (9.9, 7.95, 1.6, 0.75)
    _box(ax, values[:2], values[2], values[3], r"$v_{k,t}$", fc="#f0fff4", ec="#276749")

    _arrow(ax, (1.9, 8.9), (2.3, 8.85), color=fwd)
    _arrow(ax, (4.7, 8.85), (5.1, 8.85), color=fwd)
    _arrow(ax, (6.6, 9.05), (7.5, 9.5), color=fwd, connection="arc3,rad=0.2")
    _arrow(ax, (6.6, 8.6), (7.5, 8.35), color=fwd, connection="arc3,rad=-0.2")
    _arrow(ax, (9.6, 9.525), (9.9, 9.525), color=fwd)
    _arrow(ax, (9.6, 8.325), (9.9, 8.325), color=fwd)

    _label(ax, (6.9, 9.55), "actor input =\nfull concat\n(no gating)", color="#c53030", fontsize=7)
    _label(ax, (6.9, 7.75), "critic $k$ input =\nsame full concat\n(no routing/gating)", color="#276749", fontsize=7)

    # ---- Row 2: rollout -> per-channel GAE/TD -> gate-weighted targets --
    rollout = (0.4, 5.6, 1.9, 0.9)
    _box(ax, rollout[:2], rollout[2], rollout[3], "Rollout buffer\n$r_t$, done$_t$", fc="#fff8e6", ec="#b7791f", fontsize=8.5)

    gae = (2.6, 5.15, 3.3, 1.55)
    _box(
        ax,
        gae[:2],
        gae[2],
        gae[3],
        "Per-channel GAE + TD\n$A_{k,t},\\ \\delta_{k,t},\\ R_{k,t}$\nusing own $(\\gamma_k, \\lambda_k)$",
        fc="#edf2f7",
        ec="#4a5568",
        bold=True,
        fontsize=8.5,
    )

    comb_adv = (6.35, 6.55, 2.9, 0.9)
    _box(
        ax,
        comb_adv[:2],
        comb_adv[2],
        comb_adv[3],
        "Combined advantage\n$A_t=\\sum_k g_k A_{k,t}\\,/\\,\\sum_k g_k$",
        fc="#fefcbf",
        ec="#975a16",
        fontsize=8,
    )

    critic_tgt = (6.35, 5.15, 2.9, 0.9)
    _box(
        ax,
        critic_tgt[:2],
        critic_tgt[2],
        critic_tgt[3],
        "Critic-$k$ target\n$\\alpha_k R_{k,t}+(1-\\alpha_k)\\,\\mathrm{shared}_t$",
        fc="#fefcbf",
        ec="#975a16",
        fontsize=8,
    )

    ppo_loss = (11.9, 5.0, 3.4, 2.5)
    _box(
        ax,
        ppo_loss[:2],
        ppo_loss[2],
        ppo_loss[3],
        "PPO objective (per minibatch)\n"
        "policy loss: clipped ratio $\\times A_t$\n"
        "critic loss: $\\sum_k (v_k-\\mathrm{target}_k)^2$\n"
        "$-\\,c_H\\cdot$entropy\n\n"
        "backprop updates encoder,\nactor & all $K$ critic heads",
        fc="#f7fafc",
        ec="#c53030",
        bold=True,
        fontsize=8,
    )

    _arrow(ax, (2.3, 6.05), (2.6, 6.0), color=fwd)
    _arrow(ax, (10.7, 7.95), (5.9, 6.9), color=fwd, connection="arc3,rad=0.2")
    _label(ax, (8.3, 7.55), "$v_{k,t}$", color="#276749", fontsize=7.5)
    _arrow(ax, (5.9, 6.6), (6.35, 6.95), color=fwd, connection="arc3,rad=-0.15")
    _arrow(ax, (5.9, 5.6), (6.35, 5.6), color=fwd)
    _arrow(ax, (9.25, 7.0), (11.9, 7.0), color=fwd)
    _arrow(ax, (9.25, 5.6), (11.9, 5.6), color=fwd)
    _arrow(ax, (11.5, 9.3), (12.6, 7.5), color=fwd, connection="arc3,rad=0.25")
    _label(ax, (12.15, 8.6), "$\\pi(a|s)$", color="#c53030", fontsize=7.5)
    _arrow(ax, (11.5, 8.3), (13.6, 7.5), color=fwd, connection="arc3,rad=-0.2")
    _label(ax, (12.9, 8.55), "$v_{k,t}$", color="#276749", fontsize=7.5)

    # ---- Row 3: decentralized local controllers (feedback only) ---------
    probe = (0.4, 2.5, 2.2, 1.0)
    _box(
        ax,
        probe[:2],
        probe[2],
        probe[3],
        "Gradient-conflict probe\n$\\cos(\\nabla_{\\mathrm{enc}}L_k, \\nabla_{\\mathrm{enc}}L_j)$",
        fc="#edf2f7",
        ec="#4a5568",
        fontsize=7.5,
    )
    _arrow(ax, (2.6, 8.05), (1.6, 3.5), color="#4a5568", connection="arc3,rad=0.35", lw=1.0)
    _label(ax, (2.0, 5.6), "shared\nencoder\ngrads", color="#4a5568", fontsize=7)

    controller = (3.0, 0.9, 7.3, 2.15)
    _box(
        ax,
        controller[:2],
        controller[2],
        controller[3],
        f"LocalChannelController × K  (decentralized, no shared state)\n"
        "reads (own channel only): TD variance, usefulness (own $V_k$ fit),\n"
        "redundancy (corr with other $\\delta_j$), gradient conflict (from probe)\n"
        "writes for the next window: $g_k$, critic-$k$ LR, $\\lambda_k$, ($\\gamma_k$), $\\alpha_k$",
        fc="#e6fffa",
        ec="#234e52",
        bold=True,
        fontsize=8,
    )
    _arrow(ax, (4.25, 5.15), (5.6, 3.05), color="#4a5568", lw=1.0)
    _label(ax, (5.4, 4.1), "$\\delta_{k,t}, R_{k,t}$,\nfit quality", color="#4a5568", fontsize=7)
    _arrow(ax, (2.6, 3.0), (3.0, 2.3), color="#4a5568", lw=1.0)

    # dashed controller -> gate/blend feedback (applied next window, no gradient)
    ctl = "#234e52"
    _arrow(ax, (9.4, 3.05), (9.25, 6.6), color=ctl, connection="arc3,rad=-0.35", lw=1.1, dashed=True)
    _label(ax, (10.35, 4.6), "$g_k$", color=ctl)
    _arrow(ax, (9.9, 2.4), (9.25, 5.55), color=ctl, connection="arc3,rad=-0.3", lw=1.1, dashed=True)
    _label(ax, (10.75, 3.3), "$\\alpha_k$", color=ctl)
    _arrow(ax, (10.3, 3.05), (12.6, 5.0), color=ctl, connection="arc3,rad=0.2", lw=1.1, dashed=True)
    _label(ax, (11.7, 3.7), "critic-$k$ LR,\n$\\lambda_k(,\\gamma_k)$", color=ctl, fontsize=7)

    # ---- Legend -----------------------------------------------------------
    leg_x0, leg_y0 = 11.9, 1.55
    ax.plot([leg_x0, leg_x0 + 0.7], [leg_y0 + 1.05, leg_y0 + 1.05], color=fwd, lw=1.6)
    ax.text(leg_x0 + 0.85, leg_y0 + 1.05, "forward pass / loss (this window)", fontsize=8, va="center")
    ax.plot([leg_x0, leg_x0 + 0.7], [leg_y0 + 0.55, leg_y0 + 0.55], color=ctl, lw=1.6, linestyle="--")
    ax.text(leg_x0 + 0.85, leg_y0 + 0.55, "controller update (next window,\nno gradient)", fontsize=8, va="center")

    # ---- Key-difference callout -------------------------------------------
    ax.text(
        8,
        0.15,
        "No routing matrix $B$: the actor and every critic read the identical shared feature vector.\n"
        "Channels differ only via per-channel $(\\gamma_k,\\lambda_k)$ discounting and via $g_k,$ critic-$k$ LR, $\\alpha_k$ set by independent local controllers.",
        ha="center",
        va="center",
        fontsize=9,
        color="#2d3748",
        style="italic",
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/figures/ppo_local_channel_control_halfcheetah_v4/00_architecture.png"),
    )
    parser.add_argument("--num-channels", type=int, default=4)
    parser.add_argument("--num-modules", type=int, default=4)
    args = parser.parse_args()
    plot_v4_architecture(args.output, args.num_channels, args.num_modules)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
