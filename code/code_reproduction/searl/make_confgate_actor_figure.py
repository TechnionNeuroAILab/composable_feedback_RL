"""Architecture figure for the proposed SEARL + ConfGate TD3 agent.

Top: one population member's actor (shared W_z + trunk, three heads fed by full / pose-only /
velocity-only observations, critic-driven confidence gate). Bottom: where it plugs into SEARL.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

C = dict(obs="#dbe9f6", core="#e5e5e5", full="#cfe8cf", pose="#fde3c4", vel="#e2d4f0",
         gate="#fbd5d5", critic="#fff3b0", evo="#eef2f7", ink="#222222")


def box(ax, x, y, w, h, text, fc, fs=9, ls="-", bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=C["ink"], lw=1.2, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=C["ink"])


def arrow(ax, p, q, ls="-", color=None, rad=0.0, lw=1.4):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, lw=lw, ls=ls,
                                 color=color or C["ink"], connectionstyle=f"arc3,rad={rad}"))


fig, ax = plt.subplots(figsize=(16, 10.5))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10.5)
ax.axis("off")

ax.text(0.2, 10.15, "SEARL population member: TD3 actor with ConfGate partial-observation heads",
        fontsize=14, fontweight="bold")
ax.text(0.2, 9.8, "HalfCheetah example: obs 17-d = pose s[0:8] (z, angle, joints) | velocity s[8:17]; action 6-d",
        fontsize=9.5, color="#555555")

# ---- observation + masks ------------------------------------------------------------
box(ax, 0.3, 7.0, 1.7, 1.4, "observation s\n(17-d)", C["obs"], bold=True)
box(ax, 2.5, 8.1, 1.9, 0.75, "full s", C["full"])
box(ax, 2.5, 6.95, 1.9, 0.75, "mask velocity\n(s[8:17]=0)", C["pose"], fs=8.5)
box(ax, 2.5, 5.8, 1.9, 0.75, "mask pose\n(s[0:8]=0)", C["vel"], fs=8.5)
for y in (8.47, 7.33, 6.18):
    arrow(ax, (2.0, 7.7), (2.5, y))

# ---- shared W_z + trunk ---------------------------------------------------------------
box(ax, 5.1, 5.8, 2.0, 3.05, "shared\nW_z\nLinear(17 → z_dim)", C["core"], bold=True)
box(ax, 7.7, 5.8, 2.2, 3.05, "shared trunk\nact → Linear → act\n(evolvable widths /\nlayers / activation)",
    C["core"], bold=True, fs=9)
arrow(ax, (4.4, 8.47), (5.1, 8.2))
arrow(ax, (4.4, 7.33), (5.1, 7.33))
arrow(ax, (4.4, 6.18), (5.1, 6.4))
arrow(ax, (7.1, 7.33), (7.7, 7.33))
ax.text(5.9, 5.62, "full path: z = k_pose·W_pose s_pose\n+ k_vel·W_vel s_vel + b", fontsize=8.2,
        va="top", ha="right", color="#aa3333")
ax.text(12.9, 5.62, "aux paths use z.detach():\naux heads do not train the trunk", fontsize=8.2,
        ha="right", color="#555555", va="top")

# ---- heads ------------------------------------------------------------------------------
box(ax, 10.6, 8.0, 2.1, 0.85, "head_full\nLinear → tanh", C["full"], bold=True)
box(ax, 10.6, 6.9, 2.1, 0.85, "head_pose\nLinear → tanh", C["pose"], bold=True)
box(ax, 10.6, 5.8, 2.1, 0.85, "head_vel\nLinear → tanh", C["vel"], bold=True)
for y in (8.42, 7.32, 6.22):
    arrow(ax, (9.9, 7.33), (10.6, y))
box(ax, 13.3, 8.0, 2.4, 0.85, "a_full  → env\n(executed action)", C["full"], bold=True)
box(ax, 13.3, 6.9, 2.4, 0.85, "a_pose  (aux)", C["pose"])
box(ax, 13.3, 5.8, 2.4, 0.85, "a_vel  (aux)", C["vel"])
for y in (8.42, 7.32, 6.22):
    arrow(ax, (12.7, y), (13.3, y))

# ---- confidence gate ----------------------------------------------------------------------
box(ax, 10.2, 3.55, 3.0, 1.3, "Critic Q1(s, a_pose), Q1(s, a_vel)\n(analog of Q-margin: how good is\nwhat each partial view proposes)",
    C["critic"], fs=8.6)
box(ax, 6.0, 3.55, 3.4, 1.3, "z-normalise over batch → softmax\n→ k_pose, k_vel ∈ [k_min, 1]\n(detached, k_min evolvable)",
    C["gate"], fs=8.8, bold=True)
arrow(ax, (14.5, 6.9), (12.4, 4.85), ls="--", rad=-0.15)
arrow(ax, (14.5, 5.8), (12.9, 4.85), ls="--", rad=-0.1)
arrow(ax, (10.2, 4.2), (9.4, 4.2))
arrow(ax, (7.2, 4.85), (6.1, 5.8), ls="--", color="#aa3333")
ax.text(5.9, 4.9, "k_pose, k_vel scale W_z blocks", fontsize=8.2, color="#aa3333", ha="right")

# ---- losses -------------------------------------------------------------------------------
box(ax, 0.3, 3.55, 5.2, 1.3,
    "Losses (one TD3 step)\n"
    "actor:  −Q1(s, a_full)   → W_z + trunk + head_full  (opt_main)\n"
    "aux:    −Q1(s, a_pose) − Q1(s, a_vel) → heads only  (opt_aux)\n"
    "critics: unchanged twin-Q TD3 loss on full obs",
    C["evo"], fs=8.6)

# ---- SEARL loop ---------------------------------------------------------------------------
ax.add_patch(FancyBboxPatch((0.3, 0.3), 15.4, 2.65, boxstyle="round,pad=0.02,rounding_size=0.1",
                            fc="white", ec="#888888", lw=1.0, ls="--"))
ax.text(0.5, 2.7, "Where it plugs into SEARL (evolution loop unchanged, genome extended)", fontsize=10.5,
        fontweight="bold")
steps = [
    ("population of N\nConfGate-TD3 agents\n(each: actor above + twin critics)", C["obs"]),
    ("evaluate\n(act with a_full;\ngate needs critic_1)\nfill shared replay", C["full"]),
    ("tournament\nselection\n(+ elitism)", C["evo"]),
    ("mutate\nactor + critics", C["pose"]),
    ("train on shared\nreplay (TD3 +\naux head loss)", C["vel"]),
]
x = 0.6
for i, (t, fc) in enumerate(steps):
    box(ax, x, 1.15, 2.2, 1.3, t, fc, fs=8.3)
    if i < len(steps) - 1:
        arrow(ax, (x + 2.2, 1.8), (x + 2.55, 1.8))
    x += 2.55
ax.plot([x - 1.45, x - 1.45, 1.7], [1.15, 0.75, 0.75], color=C["ink"], ls=":", lw=1.4)
arrow(ax, (1.7, 0.75), (1.7, 1.15), ls=":")
box(ax, 13.45, 0.45, 2.1, 1.95,
    "New mutation targets\n• trunk width / depth\n• z_dim (W_z out)\n• k_min, k_max\n• activation\n• weights incl. heads",
    "#fffaf0", fs=7.6)

out = Path(__file__).parent / "figures" / "confgate_td3_actor_architecture.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
print(out)
