"""Learning curves (seed 0, 1M steps, discrete HalfCheetah) of the channel variants vs. CleanRL DQN and the current heads,
plus each channel's step-size multiplier alpha_i chosen by the loop. Writes figures/channels_halfcheetah_seed0.png."""
import matplotlib.pyplot as plt
import pandas as pd

R = "results_halfcheetah_1M/"
BASE = [("CleanRL DQN", "cleanrl_dqn", "#52514e", (0, (4, 2))), ("current 2 heads (lr gate)", "cleanrl_dqn_confgate", "#898781", (0, (1, 1.5))),
        ("current per-obs heads (lr gate)", "cleanrl_dqn_confgate_perobs", "#c3c2b7", (0, (1, 1.5)))]
NEW = [("additive, no loop", "channels_additive", "#2a78d6"), ("A+C", "channels_ac", "#eb6834"),
       ("F-lite", "channels_flite", "#1baf7a"), ("F", "channels_f", "#6250d6")]
CH = {"pose": "#2a78d6", "velocity": "#eb6834", "full": "#898781"}

fig = plt.figure(figsize=(14, 7.5))
ax = fig.add_subplot(2, 1, 1)
for name, d, c, ls in BASE + [(n, d, c, "-") for n, d, c in NEW]:
    df = pd.read_csv(R + d + "/seed0.csv")
    ax.plot(df.global_step, df.episodic_return.rolling(20, min_periods=1).mean(), color=c, ls=ls, lw=2 if ls == "-" else 1.6, label=name)
ax.set_xlabel("environment steps")
ax.set_ylabel("episode return (20-episode mean)")
ax.set_title("Discrete HalfCheetah-v4, seed 0: training-episode return", loc="left", fontsize=11)
ax.legend(ncol=4, fontsize=8.5, frameon=False, loc="upper left")
ax.grid(alpha=0.25)
for i, (name, d, _) in enumerate(NEW[1:]):
    a = fig.add_subplot(2, 3, 4 + i)
    c = pd.read_csv(R + d + "/seed0_channels.csv")
    for ch, g in c.groupby("channel"):
        a.plot(g.global_step, g.alpha.rolling(25, min_periods=1).mean(), color=CH[ch], lw=1.8, label=ch)
    a.axhline(1, color="#c3c2b7", lw=1)
    a.set_ylim(0.2, 2.1)
    a.set_title(f"{name}: step-size multiplier alpha_i", loc="left", fontsize=10)
    a.set_xlabel("environment steps")
    a.grid(alpha=0.25)
    if i == 0:
        a.legend(fontsize=8.5, frameon=False)
fig.tight_layout()
fig.savefig("figures/channels_halfcheetah_seed0.png", dpi=170)
print("wrote figures/channels_halfcheetah_seed0.png")
