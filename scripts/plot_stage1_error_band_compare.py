#!/usr/bin/env python3
"""Plot stage-1 error bands for two replay result directories."""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
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
        description="Plot two stage-1 error curves with min-max bands."
    )
    parser.add_argument(
        "--run",
        action="append",
        nargs=2,
        metavar=("LABEL", "DATA_DIR"),
        help=(
            "Run label and directory containing episode_*.csv. "
            "Can be used twice. Defaults to video chunk and trajectory_data_005."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/recordings/stage1_policy_replay/stage1_dock_pos_error_band_compare.png"),
        help=(
            "Output PNG path. Default: "
            "src/recordings/stage1_policy_replay/stage1_dock_pos_error_band_compare.png"
        ),
    )
    parser.add_argument("--step-column", default="step", help="Step column. Default: step")
    parser.add_argument(
        "--error-column",
        default="dock_pos_error",
        help=(
            "Per-step error column. Default: dock_pos_error. "
            "Note: final_dock_pos_error is stored in summary.csv, not episode_*.csv."
        ),
    )
    parser.add_argument("--dpi", type=int, default=300, help="Output image DPI. Default: 300")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.18,
        help="Opacity of each min-max error band. Default: 0.18",
    )
    return parser.parse_args()


def collect_values_by_step(
    csv_paths: list[Path],
    step_column: str,
    error_column: str,
) -> dict[int, list[float]]:
    values_by_step: dict[int, list[float]] = defaultdict(list)

    for csv_path in csv_paths:
        with csv_path.open(newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None:
                raise ValueError(f"{csv_path} is empty or has no header")

            missing = {step_column, error_column} - set(reader.fieldnames)
            if missing:
                missing_text = ", ".join(sorted(missing))
                raise ValueError(f"{csv_path} is missing required column(s): {missing_text}")

            for row in reader:
                step = int(float(row[step_column]))
                values_by_step[step].append(float(row[error_column]))

    return values_by_step


def band_stats(values_by_step: dict[int, list[float]]) -> tuple[list[int], list[float], list[float], list[float]]:
    steps = sorted(values_by_step)
    mins = [min(values_by_step[step]) for step in steps]
    maxs = [max(values_by_step[step]) for step in steps]
    means = [
        sum(values_by_step[step]) / len(values_by_step[step])
        for step in steps
    ]
    return steps, mins, maxs, means


def plot_error_band_compare(
    runs: list[tuple[str, Path]],
    step_column: str,
    error_column: str,
    output_path: Path,
    dpi: int,
    alpha: float,
) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.2))

    colors = ["tab:blue", "tab:green", "tab:orange", "tab:red"]
    for index, (label, data_dir) in enumerate(runs):
        csv_paths = sorted(data_dir.glob("episode_*.csv"))
        if not csv_paths:
            raise FileNotFoundError(f"No episode_*.csv files found in {data_dir}")

        values_by_step = collect_values_by_step(csv_paths, step_column, error_column)
        steps, mins, maxs, means = band_stats(values_by_step)
        color = colors[index % len(colors)]
        linestyle = "--" if index == 1 else "-"
        ax.fill_between(steps, mins, maxs, color=color, alpha=alpha, linewidth=0)
        ax.plot(steps, means, color=color, linestyle=linestyle, linewidth=4.4, label=label)
        print(f"{label}: {len(csv_paths)} episodes, step {steps[0]}..{steps[-1]}")

    ax.set_xlabel("Step")
    ax.set_ylabel("Error [m]")
    apply_2d_style(ax)
    ax.legend(frameon=False)
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

    plot_error_band_compare(
        runs,
        args.step_column,
        args.error_column,
        args.output,
        args.dpi,
        args.alpha,
    )
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
