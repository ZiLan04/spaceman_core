#!/usr/bin/env python3
"""Plot stage-1 episode trajectories from replay CSV files."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt


DEFAULT_DATA_DIR = Path(
    "src/recordings/stage1_policy_replay/"
    "video_ep_chunk_0001_20260724_154637.mp4"
)

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
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True, width=1.6, length=7)
    for spine in ax.spines.values():
        spine.set_linewidth(1.6)
    ax.grid(False)
    ymin, ymax = ax.get_ylim()
    if ymin < 0 < ymax:
        ax.axhline(0, color="0.35", linestyle="--", linewidth=1.6, alpha=0.75, zorder=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot base_x/base_y versus step for all stage-1 episodes."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Directory containing episode_*.csv files. Default: {DEFAULT_DATA_DIR}",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        default=["base_x", "base_y"],
        help="CSV columns to plot against step. Default: base_x base_y",
    )
    parser.add_argument(
        "--step-column",
        default="step",
        help="CSV column to use as the x axis. Default: step",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for PNG outputs. Default: <data-dir>/plots",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output image DPI. Default: 300",
    )
    return parser.parse_args()


def read_episode(csv_path: Path, step_column: str, value_column: str) -> tuple[list[float], list[float]]:
    steps: list[float] = []
    values: list[float] = []

    with csv_path.open(newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} is empty or has no header")

        missing = {step_column, value_column} - set(reader.fieldnames)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"{csv_path} is missing required column(s): {missing_text}")

        for row in reader:
            steps.append(float(row[step_column]))
            values.append(float(row[value_column]))

    return steps, values


def plot_column(
    csv_paths: list[Path],
    step_column: str,
    value_column: str,
    output_path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(9.4, 5.2))

    for index, csv_path in enumerate(csv_paths, start=1):
        steps, values = read_episode(csv_path, step_column, value_column)
        ax.plot(steps, values, linewidth=3.0, label=f"traj {index}")

    ax.set_xlabel("timesteps")
    ax.set_ylabel(f"{value_column} [m]")
    apply_2d_style(ax)
    ax.legend(fontsize=14, loc="upper left", bbox_to_anchor=(1.02, 1.0), ncol=2, frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    output_dir = args.output_dir or data_dir / "plots"

    csv_paths = sorted(data_dir.glob("episode_*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No episode_*.csv files found in {data_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for column in args.columns:
        output_path = output_dir / f"{column}_vs_{args.step_column}.png"
        plot_column(csv_paths, args.step_column, column, output_path, args.dpi)
        print(f"saved {output_path}")


if __name__ == "__main__":
    main()
