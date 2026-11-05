#!/usr/bin/env python3
"""Combine stage-2 error bands and episode lengths in two panels."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt

from plot_stage2_error_band_compare import band_stats, collect_values_by_step
from plot_stage2_success_time_compare import success_times


DEFAULT_RUNS = [
    ("Vision-guided", Path("src/recordings/stage2_policy_replay/trajectory_data_009")),
    ("Vision-free", Path("src/recordings/stage2_policy_replay/trajectory_data_008")),
]

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 20,
        "axes.labelsize": 22,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 14,
        "mathtext.fontset": "stix",
    }
)


def apply_2d_style(ax: plt.Axes) -> None:
    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
        width=1.6,
        length=7,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(1.6)
    ax.grid(False)
    ymin, ymax = ax.get_ylim()
    if ymin < 0 < ymax:
        ax.axhline(
            0,
            color="0.35",
            linestyle="--",
            linewidth=1.6,
            alpha=0.75,
            zorder=0,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine stage-2 error-band and episode-length comparisons."
    )
    parser.add_argument(
        "--run",
        action="append",
        nargs=2,
        metavar=("LABEL", "DATA_DIR"),
        help="Run label and directory containing episode_*.csv. Can be used twice.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "src/recordings/stage2_policy_replay/"
            "stage2_error_completion_compare_combined.png"
        ),
    )
    parser.add_argument("--step-column", default="step")
    parser.add_argument("--error-column", default="pose_position_error")
    parser.add_argument("--alpha", type=float, default=0.18)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def plot_combined(
    runs: list[tuple[str, Path]],
    step_column: str,
    error_column: str,
    alpha: float,
    output_path: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 8.0))
    error_ax, time_ax = axes
    colors = ["tab:blue", "tab:green", "tab:orange", "tab:red"]

    for index, (label, data_dir) in enumerate(runs):
        csv_paths = sorted(data_dir.glob("episode_*.csv"))
        if not csv_paths:
            raise FileNotFoundError(f"No episode_*.csv files found in {data_dir}")

        color = colors[index % len(colors)]
        linestyle = "--" if index == 1 else "-"
        values_by_step = collect_values_by_step(csv_paths, step_column, error_column)
        steps, mins, maxs, means = band_stats(values_by_step)
        error_ax.fill_between(
            steps, mins, maxs, color=color, alpha=alpha, linewidth=0
        )
        error_ax.plot(
            steps,
            means,
            color=color,
            linestyle=linestyle,
            linewidth=4.4,
            label=label,
        )

        episodes, times = success_times(data_dir, step_column)
        time_ax.plot(
            episodes,
            times,
            color=color,
            linestyle=linestyle,
            marker="o",
            markersize=7,
            linewidth=4.4,
            label=label,
        )
        print(
            f"{label}: {len(csv_paths)} episodes, "
            f"step {steps[0]}..{steps[-1]}, episode length {min(times)}..{max(times)}"
        )

    error_ax.set_xlabel("Step")
    error_ax.set_ylabel("Error [m]")
    error_ax.legend(frameon=False, loc="best")

    time_ax.set_xlabel("Episode")
    time_ax.set_ylabel("Episode length [steps]", color="red")
    time_ax.set_xlim(0.5, 20.5)
    time_ax.set_xticks([1, 5, 10, 15, 20])
    time_ax.set_xticks(range(1, 21), minor=True)
    ymin, ymax = time_ax.get_ylim()
    time_ax.set_ylim(ymin, ymax + 0.3 * (ymax - ymin))
    time_ax.legend(frameon=False, loc="upper right")

    for ax in axes:
        apply_2d_style(ax)

    error_ax.text(
        0.5, -0.27, "(a)", transform=error_ax.transAxes, ha="center", va="top"
    )
    time_ax.text(
        0.5, -0.34, "(b)", transform=time_ax.transAxes, ha="center", va="top"
    )

    fig.tight_layout(h_pad=1.2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {output_path}")


def main() -> None:
    args = parse_args()
    runs = (
        [(label, Path(data_dir)) for label, data_dir in args.run]
        if args.run
        else DEFAULT_RUNS
    )
    plot_combined(
        runs,
        args.step_column,
        args.error_column,
        args.alpha,
        args.output,
        args.dpi,
    )


if __name__ == "__main__":
    main()
