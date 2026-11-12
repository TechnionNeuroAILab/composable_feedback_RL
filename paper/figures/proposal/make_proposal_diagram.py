#!/usr/bin/env python
"""Schematic: current SEARL + ConfGate aux heads (a) vs. the proposed feature-specific TD decomposition with a
per-channel control loop (b). Writes feature_specific_loop.{png,pdf} next to this file."""
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"
CH = ["#2a78d6", "#eb6834", "#1baf7a"]      # feature channels / observation groups
JOINT = "#898781"                            # joint channel (full observation)
LOOP = "#6250d6"                             # control loop (fast, within lifetime)
EVO = "#52514e"                              # evolution (slow, across generations)
ERR = INK                                    # shared TD error / gradient


def box(ax, x, y, w, h, text, fc="white", ec=INK2, tc=INK, fs=9, lw=1.2, ls="-", bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fc, ec=ec, lw=lw, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal", linespacing=1.25)


def arrow(ax, p, q, color=INK2, lw=1.3, ls="-", rad=0.0, head=9, z=1):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>,head_length=0.55,head_width=0.28", mutation_scale=head * 1.4,
                                 connectionstyle=f"arc3,rad={rad}", color=color, lw=lw, ls=ls,
                                 shrinkA=2, shrinkB=2, zorder=z))


def label(ax, x, y, text, color=INK2, fs=8, ha="center", style="italic", **kw):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fs, color=color, style=style, **kw)


def obs_strip(ax, x, y, w, h, groups):
    """Observation s drawn as coloured group segments."""
    seg = w / len(groups)
    for i, (name, c) in enumerate(groups):
        ax.add_patch(FancyBboxPatch((x + i * seg + 0.03, y), seg - 0.06, h, boxstyle="round,pad=0,rounding_size=0.08",
                                    fc=c, ec="none", alpha=0.9))
        ax.text(x + (i + 0.5) * seg, y + h / 2, name, ha="center", va="center", fontsize=8, color="white")


def panel_current(ax):
    ax.set_title("(a) Current: SEARL + ConfGate auxiliary heads", fontsize=11, loc="left", color=INK, pad=8)
    # observation
    obs_strip(ax, 0.6, 0.3, 4.2, 0.55, [("group 1", CH[0]), ("group 2", CH[1])])
    label(ax, 2.7, 0.08, "observation s", style="normal")
    # Q-network
    box(ax, 0.6, 1.6, 4.2, 1.5, "Q-network trunk\n(SEARL MLP, unchanged)", fc="#f3f3f1")
    box(ax, 1.4, 3.75, 2.6, 0.7, r"$Q_{\rm full}(s,a)$  → acts")
    arrow(ax, (2.7, 0.9), (2.7, 1.58))
    arrow(ax, (2.7, 3.12), (2.7, 3.73))
    # TD error, gradient into the trunk
    box(ax, 0.6, 5.15, 4.2, 0.75, r"TD error  $\delta = y - Q_{\rm full}(s,a)$", ec=ERR, lw=1.6)
    arrow(ax, (2.7, 4.47), (2.7, 5.13))
    arrow(ax, (0.75, 5.13), (0.75, 3.12), color=ERR, lw=1.8)
    label(ax, 0.42, 4.1, "gradient", color=ERR, rotation=90)

    # aux heads: masked obs through the (detached) trunk
    for k, (yy, c) in enumerate([(3.95, CH[0]), (2.75, CH[1])]):
        box(ax, 5.55, yy, 1.55, 0.7, rf"head $Q_{k + 1}$" + "\n" + rf"on $M_{k + 1}s$", ec=c, lw=1.8, fs=8.5)
        arrow(ax, (4.82, 2.35 + 0.15 * k), (5.53, yy + 0.35), color=c, ls=(0, (3, 2)))
        box(ax, 7.55, yy, 2.15, 0.7, rf"own error $e_{k + 1}=y-Q_{k + 1}$", ec=c, fs=8)
        arrow(ax, (7.12, yy + 0.35), (7.53, yy + 0.35), color=c)
    label(ax, 5.6, 1.75, "trunk features,\ndetached", fs=7.5)
    label(ax, 8.62, 2.45, "separate fit, own Adam;\nerrors do not sum", fs=7.5)

    # confidence and the gate
    box(ax, 5.55, 5.15, 4.15, 0.75, r"$m$ = agreement of heads' greedy actions" + "\n(128 replay states, once per phase)",
        ec=LOOP, fs=8)
    arrow(ax, (6.33, 4.67), (6.75, 5.13), color=LOOP)
    box(ax, 5.55, 6.45, 4.15, 0.75, r"one global knob:  $\mathrm{lr}_{\rm eff}=\mathrm{lr}_{\rm base}\,(f+\sigma m)\leq \mathrm{lr}_{\rm base}$",
        ec=LOOP, lw=1.8, fs=8.5)
    arrow(ax, (7.6, 5.92), (7.6, 6.43), color=LOOP)
    arrow(ax, (5.53, 6.82), (4.82, 2.6), color=LOOP, lw=1.8, rad=0.25)
    label(ax, 4.0, 7.0, "sets lr for the\nwhole TD phase", color=LOOP, fs=7.5)
    label(ax, 7.6, 7.5, "loop: m → global lr → trunk → heads' features → m  (long, indirect)", color=LOOP, fs=7.5)

    # evolution
    box(ax, 0.6, 8.05, 9.1, 0.65, r"SEARL evolution (slow): mutates $\mathrm{lr}_{\rm base}$, architecture, activation, and $f,\ \sigma$ ($\lambda$ in mix)",
        fc="#f3f3f1", ec=EVO, ls=(0, (4, 2)), fs=8.5)


def panel_proposal(ax):
    ax.set_title("(b) Proposal: feature-specific TD decomposition + per-channel control loop", fontsize=11,
                 loc="left", color=INK, pad=8)
    chans = [("1", CH[0]), ("2", CH[1]), ("3", CH[2]), ("joint", JOINT)]
    xs = [0.6, 2.05, 3.5, 4.95]
    w = 1.25
    obs_strip(ax, 0.6, 0.3, 4.2, 0.55, [("group 1", CH[0]), ("group 2", CH[1]), ("group 3", CH[2])])
    label(ax, 2.7, 0.08, "observation s", style="normal")
    box(ax, 4.95, 0.3, 1.25, 0.55, "full s", fc=JOINT, ec="none", tc="white", fs=8)

    for (name, c), x in zip(chans, xs):
        sub = r"$M_{%s}s$" % name if name != "joint" else "s"
        arrow(ax, (x + w / 2, 0.9), (x + w / 2, 1.48), color=c)
        box(ax, x, 1.5, w, 0.95, f"sub-encoder\n$f_{{{name}}}$ ({sub})", ec=c, lw=1.8, fs=8)
        box(ax, x, 2.95, w, 0.75, rf"$g_{{{name}}}\cdot Q_{{{name}}}(s,a)$", ec=c, lw=1.8, fs=8.5)
        arrow(ax, (x + w / 2, 2.47), (x + w / 2, 2.93), color=c)
        arrow(ax, (x + w / 2, 3.72), (3.4, 4.25), color=c)
    # sum -> Q
    ax.add_patch(plt.Circle((3.4, 4.5), 0.25, fc="white", ec=INK, lw=1.4))
    ax.text(3.4, 4.5, r"$\Sigma$", ha="center", va="center", fontsize=11)
    box(ax, 2.45, 5.0, 1.9, 0.6, r"$Q=\sum_i Q_i$ → acts", fs=8.5)
    arrow(ax, (3.4, 4.76), (3.4, 4.98))

    # error decomposition
    box(ax, 0.6, 6.1, 5.6, 0.75,
        r"per-channel errors  $\delta_i=\frac{r}{N}+\gamma\,Q_i(s',a^*)-Q_i(s,a)$,     $\sum_i\delta_i=\delta$ (the DQN TD error)",
        ec=ERR, lw=1.6, fs=7.3)
    arrow(ax, (3.4, 5.62), (3.4, 6.08))
    for (name, c), x in zip(chans, xs):
        ax.add_patch(plt.Circle((x + w / 2, 7.15), 0.13, fc=c, ec="none"))
        ax.text(x + w / 2 + 0.2, 7.15, rf"$\delta_{{{name}}}$", fontsize=8, va="center", color=INK)
    label(ax, 3.4, 7.5, "distributed error vector (the paper's feature-specific RPEs)", fs=7.5)
    # shared gradient back down to every channel
    arrow(ax, (0.6, 6.48), (0.45, 2.0), color=ERR, lw=1.8, rad=0.15)
    label(ax, -0.05, 4.3, r"shared $\delta$ trains all channels", color=ERR, rotation=90)

    # controller (control loop)
    box(ax, 6.75, 5.85, 3.0, 1.55,
        "controller (fast loop)\nper-channel SNR$_i$ of $\\delta_i$\ncancellation $c=|\\sum\\delta_i|/\\sum|\\delta_i|$\nconcentration of $|\\delta_i|$",
        ec=LOOP, lw=1.8, fs=8)
    arrow(ax, (6.22, 6.48), (6.73, 6.48), color=LOOP, lw=1.6)
    box(ax, 6.75, 2.6, 3.0, 2.25,
        "per-channel knobs\n" + r"$\alpha_i$: step size of channel $i$" + "\n" + r"$g_i$: weight of channel $i$ in Q"
        + "\nreplay priority from $\\delta_i$\n(global: lr, $\\tau$, replay ratio from $c$)", ec=LOOP, lw=1.8, fs=8)
    arrow(ax, (8.25, 5.83), (8.25, 4.87), color=LOOP, lw=1.6)
    arrow(ax, (6.73, 3.33), (6.22, 3.33), color=LOOP, lw=1.8)
    arrow(ax, (6.73, 2.9), (6.22, 1.98), color=LOOP, lw=1.8)
    label(ax, 8.25, 2.2, r"loop: $\delta_i$ → $\alpha_i, g_i$ → channel $i$ → $\delta_i$  (short, per update)", color=LOOP, fs=7.5)

    box(ax, 0.6, 8.05, 9.15, 0.65,
        "SEARL evolution (slow): mutates base lr, controller gains, architecture (per channel or shared)",
        fc="#f3f3f1", ec=EVO, ls=(0, (4, 2)), fs=8.5)


def panel_multi_error(ax, tag="(c)"):
    ax.set_title(f"{tag} Option F: multiple observation-derived errors, each training its own channel", fontsize=11,
                 loc="left", color=INK, pad=8)
    chans = [("1", CH[0]), ("2", CH[1]), ("res", JOINT)]
    xs = [0.6, 2.75, 4.9]
    w, inner = 1.9, 0.4          # column width; boxes start `inner` right of the column so the gradient line fits
    obs_strip(ax, 0.6, 0.3, 4.05, 0.55, [("group 1", CH[0]), ("group 2", CH[1])])
    label(ax, 2.6, 0.08, "observation s", style="normal")
    box(ax, 4.9, 0.3, 1.9, 0.55, "full s", fc=JOINT, ec="none", tc="white", fs=8)

    enc = {"1": "encoder → $\\psi_1(s,a)$\npredicts future $\\phi_1$",
           "2": "encoder → $\\psi_2(s,a)$\npredicts future $\\phi_2$",
           "res": "residual net\n$Q_{\\rm res}(s,a)$"}
    val = {"1": "$Q_1=w_1\\cdot\\psi_1$", "2": "$Q_2=w_2\\cdot\\psi_2$", "res": "$Q_{\\rm res}$"}
    err = {"1": "$\\delta^{\\psi}_1=\\phi_1(s)+\\gamma\\psi_1(s')-\\psi_1(s)$",
           "2": "$\\delta^{\\psi}_2=\\phi_2(s)+\\gamma\\psi_2(s')-\\psi_2(s)$",
           "res": "$\\delta_{\\rm res}=r+\\gamma Q(s')-Q(s)$"}
    for (name, c), x in zip(chans, xs):
        bx = x + inner
        bw = w - inner
        arrow(ax, (bx + bw / 2, 0.9), (bx + bw / 2, 1.48), color=c)
        box(ax, bx, 1.5, bw, 0.95, enc[name], ec=c, lw=1.8, fs=7.5)
        arrow(ax, (bx + bw / 2, 2.47), (bx + bw / 2, 2.93), color=c)
        box(ax, bx, 2.95, bw, 0.7, ("$g_{%s}\\cdot$" % name) + val[name], ec=c, lw=1.8, fs=8)
        arrow(ax, (bx + bw / 2, 3.67), (3.7, 4.25), color=c)
        # its own error, and that error's own gradient back to its own channel
        box(ax, x, 6.0, w, 0.75, err[name], ec=c, lw=2.0, fs=7)
        arrow(ax, (x + 0.15, 5.98), (x + 0.15, 1.75), color=c, lw=2.0)
    ax.add_patch(plt.Circle((3.7, 4.5), 0.25, fc="white", ec=INK, lw=1.4))
    ax.text(3.7, 4.5, r"$\Sigma$", ha="center", va="center", fontsize=11)
    box(ax, 2.95, 5.0, 1.5, 0.55, r"$Q$ → acts", fs=8.5)
    arrow(ax, (3.7, 4.76), (3.7, 4.98))
    arrow(ax, (4.47, 5.3), (5.6, 5.98), color=INK2)
    label(ax, 1.85, 5.65, "targets are the\nobservations themselves", fs=7)

    # reward-weight error: shared across groups (the paper's competition between features)
    box(ax, 0.6, 7.0, 6.2, 0.6, r"reward error  $\delta^{r}=r-\sum_g w_g\cdot\phi_g(s)$  trains the $w_g$ (shared: groups compete for reward)",
        ec=ERR, lw=1.6, fs=7.6)
    arrow(ax, (0.58, 7.3), (0.95, 3.3), color=ERR, lw=1.6, rad=0.25)

    # controller and knobs
    box(ax, 7.05, 5.6, 2.7, 2.0,
        "controller (fast loop)\nper-group $|\\delta^{\\psi}_g|$, SNR$_g$\nreward share $|w_g\\cdot\\phi_g|$\n"
        "residual share $|\\delta_{\\rm res}|$\nagreement of the errors", ec=LOOP, lw=1.8, fs=7.8)
    arrow(ax, (6.82, 6.38), (7.03, 6.38), color=LOOP, lw=1.6)
    box(ax, 7.05, 2.3, 2.7, 2.6,
        "per-group knobs\n$\\alpha_g$: step size of group $g$\n$\\gamma_g$: horizon of group $g$\n$g_g$: weight in Q\n"
        "$\\beta$: residual vs. groups\nreplay priority from $\\delta^{\\psi}_g$", ec=LOOP, lw=1.8, fs=7.8)
    arrow(ax, (8.4, 5.58), (8.4, 4.92), color=LOOP, lw=1.6)
    arrow(ax, (7.03, 3.3), (6.82, 3.3), color=LOOP, lw=1.8)
    arrow(ax, (7.03, 2.6), (6.82, 1.98), color=LOOP, lw=1.8)
    label(ax, 8.4, 1.95, r"loop: errors$_g$ → $\alpha_g,\gamma_g,g_g$ → channel $g$ → errors$_g$", color=LOOP, fs=7.3)

    box(ax, 0.6, 8.05, 9.15, 0.65,
        "SEARL evolution (slow): mutates base lr, controller gains, which observation groups get a channel",
        fc="#f3f3f1", ec=EVO, ls=(0, (4, 2)), fs=8.5)


LEGEND = ("Legend:  black = shared error / gradient    violet = fast control loop (within lifetime)    "
          "dashed grey = evolution (across generations)    coloured = observation groups / channels (and, in (c), their own errors)")


def render(panels, name, width, legend=LEGEND):
    fig, axes = plt.subplots(1, len(panels), figsize=(width, 7.6), facecolor=SURFACE)
    axes = [axes] if len(panels) == 1 else axes
    for ax, draw in zip(axes, panels):
        ax.set_xlim(-0.3, 10)
        ax.set_ylim(-0.1, 8.9)
        ax.set_facecolor(SURFACE)
        ax.axis("off")
        draw(ax)
    fig.text(0.5, 0.015, legend, ha="center", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.06 if "\n" in legend else 0.03, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"{name}.{ext}"), dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", os.path.join(HERE, f"{name}.png"))


render([panel_current, panel_proposal], "feature_specific_loop", 17)
render([lambda ax: panel_multi_error(ax, "")], "multi_error_channels", 9.5, LEGEND.replace("    violet", "\nviolet", 1).replace("    coloured", "\ncoloured", 1))
render([panel_current, panel_proposal, panel_multi_error], "all_options", 25.5)
