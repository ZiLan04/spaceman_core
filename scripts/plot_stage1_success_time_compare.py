#!/usr/bin/env python3
"""Plot stage-1 episode completion/success time for two replay result directories."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt


DEFAULT_RUNS = [
    (
        "Vision-guided",
        Path("src/recordings/stage1_policy_replay/video_ep_chunk_0001_20260724_154637.mp4"),
    ),
    ("Vision-free", Path("src/recordings/stage1_policy_replay/trajectory_data_005")),
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
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True, width=1.6, length=7)
    for spine in ax.spines.values():
        spine.set_linewidth(1.6)
    ax.grid(False)
    ymin, ymax = ax.get_ylim()
    if ymin < 0 < ymax:
        ax.axhline(0, color="0.35", linestyle="--", linewidth=1.6, alpha=0.75, zorder=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot final episode step as success/completion time for two stage-1 runs."
    )
    parser.add_argument(
        "--run",
        action="append",
        nargs=2,
        metavar=("LABEL", "DATA_DIR"),
        help=(
            "Run label and directory containing episode_*.csv. "
            "Can be used twice. Defaults to the stage1 error-band comparison runs."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/recordings/stage1_policy_replay/stage1_success_time_compare.png"),
        help=(
            "Output PNG path. Default: "
            "src/recordings/stage1_policy_replay/stage1_success_time_compare.png"
        ),
    )
    parser.add_argument("--step-column", default="step", help="Step column. Default: step")
    parser.add_argument("--dpi", type=int, default=300, help="Output image DPI. Default: 300")
    return parser.parse_args()


def final_step(csv_path: Path, step_column: str) -> int:
    last_row: dict[str, str] | None = None
    with csv_path.open(newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} is empty or has no header")
        if step_column not in reader.fieldnames:
            raise ValueError(f"{csv_path} is missing required column: {step_column}")

        for row in reader:
            last_row = row

    if last_row is None:
        raise ValueError(f"{csv_path} has no data rows")
    return int(float(last_row[step_column]))


def success_times(data_dir: Path, step_column: str) -> tuple[list[int], list[int]]:
    csv_paths = sorted(data_dir.glob("episode_*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No episode_*.csv files found in {data_dir}")

    episodes = list(range(1, len(csv_paths) + 1))
    times = [final_step(csv_path, step_column) for csv_path in csv_paths]
    return episodes, times


def plot_success_time_compare(
    runs: list[tuple[str, Path]],
    step_column: str,
    output_path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8))

    colors = ["tab:blue", "tab:green", "tab:orange", "tab:red"]
    for index, (label, data_dir) in enumerate(runs):
        episodes, times = success_times(data_dir, step_column)
        color = colors[index % len(colors)]
        linestyle = "--" if index == 1 else "-"
        ax.plot(
            episodes,
            times,
            color=color,
            linestyle=linestyle,
            marker="o",
            markersize=7,
            linewidth=4.4,
            label=label,
        )
        print(f"{label}: {len(episodes)} episodes, final step {min(times)}..{max(times)}")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Completion time")
    ax.set_xlim(0.5, 20.5)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xticks(range(1, 21), minor=True)
    apply_2d_style(ax)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    runs = (
        [(label, Path(data_dir)) for label, data_dir in args.run]
        if args.run
        else DEFAULT_RUNS
    )

    plot_success_time_compare(runs, args.step_column, args.output, args.dpi)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
