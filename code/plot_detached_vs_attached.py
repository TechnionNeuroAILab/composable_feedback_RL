"""Compare detached W_z (B-only) vs attached ctrl_lr+opt2 learning curves."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "paper" / "figures"

ATTACHED_CKPT = ROOT / "paper" / "_tmp_b_feedb_cg" / "ckpt_ctrl_lr_opt2_10s_decay3500"
DETACHED_CKPT = ROOT / "paper" / "_tmp_b_feedb_cg_det" / "ckpt_decay3500"

COLOR_ATT_BASE = "#2ca02c"
COLOR_ATT_CTRL = "#e377c2"
COLOR_DET_BASE = "#1a6b1a"
COLOR_DET_GATE = "#9467bd"


def _smooth(arr: np.ndarray, w: int) -> np.ndarray:
    return np.convolve(arr, np.ones(w) / w, mode="valid") if w >= 2 else arr


def _load_ckpt(path: Path) -> Dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    ep = ckpt["episode_returns"]
    last100 = float(np.mean(ep[-100:])) if len(ep) >= 100 else float(np.mean(ep)) if ep else 0.0
    return {"episode_returns": ep, "last100_mean": last100}


def _load_seed_series(ckpt_dir: Path, pattern: str) -> List[Dict]:
    files = sorted(
        f for f in ckpt_dir.glob(pattern)
        if f.stem.count("_") == 1  # skip duplicate *_seedN_seedN.pt
    )
    if not files:
        raise FileNotFoundError(f"No checkpoints matching {pattern} in {ckpt_dir}")
    return [_load_ckpt(f) for f in files]


def _prep_multi(results: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays = [np.asarray(r["episode_returns"], dtype=float) for r in results]
    n = min(len(a) for a in arrays)
    M = np.stack([a[:n] for a in arrays], axis=0)
    ep = np.arange(1, n + 1)
    mu = M.mean(0)
    sd = M.std(0, ddof=1) if M.shape[0] > 1 else np.zeros(n)
    return ep, mu, sd


def _prep_single(result: Dict) -> Tuple[np.ndarray, np.ndarray]:
    ep = np.asarray(result["episode_returns"], dtype=float)
    return np.arange(1, len(ep) + 1), ep


def plot_comparison(out_jpg: Path) -> None:
    att_base = _load_seed_series(ATTACHED_CKPT, "base_seed*.pt")
    att_ctrl = _load_seed_series(ATTACHED_CKPT, "ctrllr_seed*.pt")
    det_base = [_load_ckpt(DETACHED_CKPT / "b_feedb_cg_det_seed1.pt")]
    det_gate = [_load_ckpt(DETACHED_CKPT / "b_feedb_cg_det_gate_seed1.pt")]

    ep_ab, mu_ab, sd_ab = _prep_multi(att_base)
    ep_ac, mu_ac, sd_ac = _prep_multi(att_ctrl)
    ep_db, y_db = _prep_single(det_base[0])
    ep_dg, y_dg = _prep_single(det_gate[0])

    n = min(len(ep_ab), len(ep_ac))
    W = max(1, n // 200)

    def sm(ep, y, sd=None):
        y = y[: len(ep)]
        ep_s = ep[W - 1 :]
        y_s = _smooth(y, W)
        if sd is not None:
            sd = sd[: len(ep)]
            return ep_s, y_s, _smooth(sd, W)
        return ep_s, y_s

    ep_abs, sm_abs, sd_abs = sm(ep_ab, mu_ab, sd_ab)
    ep_acs, sm_acs, sd_acs = sm(ep_ac, mu_ac, sd_ac)
    ep_dbs, sm_dbs = sm(ep_db, y_db)
    ep_dgs, sm_dgs = sm(ep_dg, y_dg)

    l100_ab = float(np.mean([r["last100_mean"] for r in att_base]))
    l100_ac = float(np.mean([r["last100_mean"] for r in att_ctrl]))
    l100_db = det_base[0]["last100_mean"]
    l100_dg = det_gate[0]["last100_mean"]

    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)

    ax.fill_between(ep_abs, sm_abs - sd_abs, sm_abs + sd_abs, color=COLOR_ATT_BASE, alpha=0.15, linewidth=0)
    ax.fill_between(ep_acs, sm_acs - sd_acs, sm_acs + sd_acs, color=COLOR_ATT_CTRL, alpha=0.15, linewidth=0)

    ax.plot(ep_abs, sm_abs, color=COLOR_ATT_BASE, lw=2.2,
            label=f"attached baseline (10 seeds)  last-100: {l100_ab:.1f}")
    ax.plot(ep_acs, sm_acs, color=COLOR_ATT_CTRL, lw=2.2,
            label=f"attached ctrl lr_only + soft τ (10 seeds)  last-100: {l100_ac:.1f}")
    ax.plot(ep_dbs, sm_dbs, color=COLOR_DET_BASE, lw=2.0, ls="--",
            label=f"detached W_z baseline (1 seed, B-only)  last-100: {l100_db:.1f}")
    ax.plot(ep_dgs, sm_dgs, color=COLOR_DET_GATE, lw=2.0, ls="--",
            label=f"detached W_z + conf. gate (1 seed)  last-100: {l100_dg:.1f}")

    ax.axhline(500, color="gray", lw=1.0, ls="--", alpha=0.6, label="max (500)")
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Episodic return", fontsize=12)
    ax.set_title(
        "W_z attached (backprop+Adam) vs detached (B-feedback only)  "
        "(ε-decay=3500, attached=soft target τ=0.005)",
        fontsize=11,
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 520)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_jpg, dpi=150, bbox_inches="tight", format="jpeg")
    plt.close(fig)
    print(f"Wrote {out_jpg}", flush=True)


if __name__ == "__main__":
    plot_comparison(FIG_DIR / "b_feedback_detached_vs_attached_ctrl_lr_decay3500.jpg")
