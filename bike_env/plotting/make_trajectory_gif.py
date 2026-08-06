#!/usr/bin/env python3
"""Animate a saved evaluation trajectory as a scrolling bike-env GIF."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, FancyBboxPatch

N_LANES = 3
VISIBILITY_RANGE = 25.0
WINDOW_WIDTH = 55.0


def load_episode(run_dir: Path, stage: str, episode: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    steps = pd.read_csv(run_dir / "evaluation_steps.csv")
    steps = steps[(steps["stage"] == stage) & (steps["episode"] == episode)].reset_index(drop=True)

    visible = pd.read_csv(run_dir / "evaluation_visible_holes.csv")
    visible = visible[(visible["stage"] == stage) & (visible["episode"] == episode)]

    events = pd.read_csv(run_dir / "evaluation_holes.csv")
    events = events[(events["stage"] == stage) & (events["episode"] == episode)]
    events = events[events["event"].isin(["avoided", "collision"])]

    return steps, visible, events


def _lane_color(lane: int) -> str:
    return ("#e8eef5", "#dde5f0", "#d2dceb")[lane % N_LANES]


def make_gif(
    run_dir: Path,
    output_path: Path,
    stage: str = "end",
    episode: int = 0,
    duration_sec: float = 10.0,
    fps: int = 30,
) -> Path:
    steps, visible_holes, events = load_episode(run_dir, stage, episode)
    if steps.empty:
        raise ValueError(f"No steps for stage={stage!r}, episode={episode}")

    n_frames = int(round(duration_sec * fps))
    frame_indices = np.linspace(0, len(steps) - 1, n_frames, dtype=int)

    event_lookup = {
        (round(row.distance, 3), int(row.lane)): row.event
        for _, row in events.iterrows()
    }
    triggered_events: set[tuple[float, int]] = set()

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=100)
    fig.patch.set_facecolor("#f7f7f7")

    trail_line, = ax.plot([], [], color="steelblue", linewidth=2.2, zorder=4)
    bike_marker = Circle((0, 0), 0.18, color="#1f77b4", ec="white", lw=1.5, zorder=6)
    ax.add_patch(bike_marker)
    hole_artists: list[tuple[plt.Artist, float, int]] = []
    event_artists: list[plt.Artist] = []

    ax.set_xlim(0, WINDOW_WIDTH)
    ax.set_ylim(-0.55, N_LANES - 0.45)
    ax.set_yticks(range(N_LANES))
    ax.set_xlabel("Longitudinal Distance")
    ax.set_ylabel("Lane")
    ax.set_title(f"Bike Environment — {stage.title()} Stage (eval episode {episode})")
    ax.grid(True, alpha=0.25)

    for lane in range(N_LANES):
        rect = FancyBboxPatch(
            (0, lane - 0.5),
            WINDOW_WIDTH,
            1.0,
            boxstyle="square,pad=0",
            facecolor=_lane_color(lane),
            edgecolor="#b0b8c4",
            linewidth=0.8,
            zorder=0,
        )
        ax.add_patch(rect)

    status_text = ax.text(
        0.02,
        0.97,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
    )

    def _clear_dynamic_artists() -> None:
        nonlocal hole_artists, event_artists
        for artist, _, _ in hole_artists:
            artist.remove()
        for artist in event_artists:
            artist.remove()
        hole_artists = []
        event_artists = []

    def _update(frame_idx: int):
        step_row = steps.iloc[frame_idx]
        bike_x = float(step_row["distance"])
        bike_y = float(step_row["lane_position"])
        view_start = max(0.0, bike_x - WINDOW_WIDTH * 0.35)
        view_end = view_start + WINDOW_WIDTH
        ax.set_xlim(view_start, view_end)

        history = steps.iloc[: frame_idx + 1]
        trail_line.set_data(history["distance"], history["lane_position"])
        bike_marker.center = (bike_x, bike_y)

        _clear_dynamic_artists()

        for _, hole in visible_holes.iterrows():
            rel = hole["distance"] - bike_x
            if not (0.0 <= rel <= VISIBILITY_RANGE):
                continue
            x = hole["distance"]
            y = int(hole["lane"])
            marker = ax.scatter(
                [x],
                [y],
                marker="x",
                s=55,
                color="black",
                alpha=0.55,
                zorder=3,
            )
            hole_artists.append((marker, x, y))

        for key, event_name in event_lookup.items():
            dist, lane = key
            if dist > bike_x or key in triggered_events:
                continue
            if dist < view_start:
                triggered_events.add(key)
                continue
            color = "green" if event_name == "avoided" else "red"
            marker = ax.scatter(
                [dist],
                [lane],
                s=90,
                color=color,
                zorder=5,
                edgecolors="white",
                linewidths=0.6,
            )
            event_artists.append(marker)
            if abs(dist - bike_x) < 0.6:
                triggered_events.add(key)

        status_text.set_text(
            f"distance={bike_x:.1f}  lane={int(step_row['lane'])}  "
            f"action={step_row['action_name']}"
        )
        artists = [trail_line, bike_marker, status_text]
        artists.extend(a for a, _, _ in hole_artists)
        artists.extend(event_artists)
        return artists

    anim = FuncAnimation(fig, _update, frames=frame_indices, interval=1000 / fps, blit=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(output_path), writer=PillowWriter(fps=fps))
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "code"
        / "training_results_v1"
        / "lanes_3__actions_3__falls_hole-collision__lambda_5__seed_1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent
        / "v1_plots"
        / "3lanes_3actions_8000ep"
        / "lanes_3__actions_3__falls_hole-collision__lambda_5__seed_1"
        / "04_end_trajectory_animation.gif",
    )
    parser.add_argument("--stage", default="end")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    out = make_gif(
        args.run_dir,
        args.output,
        stage=args.stage,
        episode=args.episode,
        duration_sec=args.duration,
        fps=args.fps,
    )
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
