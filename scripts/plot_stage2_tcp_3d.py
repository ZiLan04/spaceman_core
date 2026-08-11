#!/usr/bin/env python3
"""Plot stage-2 TCP trajectories from replay CSV files in 3D."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt


DEFAULT_DATA_DIR = Path("src/recordings/stage2_policy_replay/trajectory_data_001")

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


def apply_3d_style(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", which="both", direction="in", width=1.6, length=7)
    ax.grid(False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot tcp_x/tcp_y/tcp_z trajectories for all stage-2 episodes."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Directory containing episode_*.csv files. Default: {DEFAULT_DATA_DIR}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Default: <data-dir>/plots/tcp_3d_trajectories.png",
    )
    parser.add_argument("--x-column", default="tcp_x", help="X-axis column. Default: tcp_x")
    parser.add_argument("--y-column", default="tcp_y", help="Y-axis column. Default: tcp_y")
    parser.add_argument("--z-column", default="tcp_z", help="Z-axis column. Default: tcp_z")
    parser.add_argument("--dpi", type=int, default=300, help="Output image DPI. Default: 300")
    parser.add_argument("--elev", type=float, default=22.0, help="3D view elevation. Default: 22")
    parser.add_argument("--azim", type=float, default=-58.0, help="3D view azimuth. Default: -58")
    return parser.parse_args()


def read_xyz(
    csv_path: Path,
    x_column: str,
    y_column: str,
    z_column: str,
) -> tuple[list[float], list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []

    with csv_path.open(newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} is empty or has no header")

        missing = {x_column, y_column, z_column} - set(reader.fieldnames)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"{csv_path} is missing required column(s): {missing_text}")

        for row in reader:
            xs.append(float(row[x_column]))
            ys.append(float(row[y_column]))
            zs.append(float(row[z_column]))

    return xs, ys, zs


def set_equal_axes(ax: plt.Axes, points: list[tuple[list[float], list[float], list[float]]]) -> None:
    all_x = [value for xs, _, _ in points for value in xs]
    all_y = [value for _, ys, _ in points for value in ys]
    all_z = [value for _, _, zs in points for value in zs]

    centers = [
        (min(all_x) + max(all_x)) / 2,
        (min(all_y) + max(all_y)) / 2,
        (min(all_z) + max(all_z)) / 2,
    ]
    radius = max(
        max(all_x) - min(all_x),
        max(all_y) - min(all_y),
        max(all_z) - min(all_z),
    ) / 2
    padding = radius * 0.08 if radius else 0.01
    radius += padding

    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def plot_trajectories(
    csv_paths: list[Path],
    x_column: str,
    y_column: str,
    z_column: str,
    output_path: Path,
    dpi: int,
    elev: float,
    azim: float,
) -> None:
    trajectories = [read_xyz(path, x_column, y_column, z_column) for path in csv_paths]

    fig = plt.figure(figsize=(10.8, 7.2))
    ax = fig.add_subplot(111, projection="3d")

    for index, (xs, ys, zs) in enumerate(trajectories, start=1):
        line = ax.plot(xs, ys, zs, linewidth=2.8, label=f"traj {index}")[0]
        ax.scatter(
            xs[0],
            ys[0],
            zs[0],
            marker="*",
            s=140,
            color=line.get_color(),
            edgecolors="black",
            linewidths=0.7,
        )

    ax.set_xlabel("X/m")
    ax.set_ylabel("Y/m")
    ax.set_zlabel("Z/m")
    ax.view_init(elev=elev, azim=azim)
    apply_3d_style(ax)
    set_equal_axes(ax, trajectories)
    ax.set_box_aspect((1, 1, 1))
    ax.legend(fontsize=14, loc="upper left", bbox_to_anchor=(1.08, 1.0), ncol=2, frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    output_path = args.output or data_dir / "plots" / "tcp_3d_trajectories.png"

    csv_paths = sorted(data_dir.glob("episode_*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No episode_*.csv files found in {data_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_trajectories(
        csv_paths,
        args.x_column,
        args.y_column,
        args.z_column,
        output_path,
        args.dpi,
        args.elev,
        args.azim,
    )
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
