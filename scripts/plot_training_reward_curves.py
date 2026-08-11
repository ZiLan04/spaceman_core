#!/usr/bin/env python3
"""Plot reward curves from TensorBoard event files without requiring TensorBoard."""

from __future__ import annotations

import argparse
import os
import struct
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory


DEFAULT_RUNS = [
    (
        "stage1 20260718",
        Path("src/outputs/rl/ppo_satellite_stage1_base/run_20260718_201139"),
    ),
    (
        "stage2 20260724",
        Path("src/outputs/rl/ppo_satellite_stage2_arm/run_20260724_214752"),
    ),
]
DEFAULT_TAG = "rollout/ep_rew_mean"

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


def build_event_class():
    file_proto = descriptor_pb2.FileDescriptorProto()
    file_proto.name = "minimal_tensorboard_event.proto"
    file_proto.package = "tensorboard_minimal"
    file_proto.syntax = "proto3"

    summary = file_proto.message_type.add()
    summary.name = "Summary"

    value = summary.nested_type.add()
    value.name = "Value"
    field = value.field.add()
    field.name = "tag"
    field.number = 1
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_STRING
    field = value.field.add()
    field.name = "simple_value"
    field.number = 2
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT

    field = summary.field.add()
    field.name = "value"
    field.number = 1
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
    field.type_name = ".tensorboard_minimal.Summary.Value"

    event = file_proto.message_type.add()
    event.name = "Event"
    field = event.field.add()
    field.name = "wall_time"
    field.number = 1
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE
    field = event.field.add()
    field.name = "step"
    field.number = 2
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_INT64
    field = event.field.add()
    field.name = "summary"
    field.number = 5
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
    field.type_name = ".tensorboard_minimal.Summary"

    pool = descriptor_pool.DescriptorPool()
    pool.Add(file_proto)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName("tensorboard_minimal.Event"))


EVENT_CLASS = build_event_class()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot reward scalar curves from stage1/stage2 TensorBoard event files."
    )
    parser.add_argument(
        "--tag",
        default=DEFAULT_TAG,
        help=f"Scalar tag to plot. Default: {DEFAULT_TAG}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/outputs/rl/training_reward_curves.png"),
        help="Output PNG path. Default: src/outputs/rl/training_reward_curves.png",
    )
    parser.add_argument(
        "--separate-output",
        type=Path,
        default=Path("src/outputs/rl/training_reward_curves_separate.png"),
        help=(
            "Output PNG path for separate subplots. "
            "Default: src/outputs/rl/training_reward_curves_separate.png"
        ),
    )
    parser.add_argument(
        "--single-output-dir",
        type=Path,
        default=Path("src/outputs/rl"),
        help="Directory for one PNG per run. Default: src/outputs/rl",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Output image DPI. Default: 300")
    parser.add_argument(
        "--list-tags",
        action="store_true",
        help="Print scalar tags found in the default runs and exit.",
    )
    return parser.parse_args()


def iter_tfrecord_payloads(event_path: Path):
    with event_path.open("rb") as file:
        while True:
            header = file.read(8)
            if not header:
                break
            if len(header) != 8:
                raise ValueError(f"Truncated TFRecord length header in {event_path}")

            (length,) = struct.unpack("<Q", header)
            file.read(4)  # masked CRC of length
            payload = file.read(length)
            file.read(4)  # masked CRC of payload

            if len(payload) != length:
                raise ValueError(f"Truncated TFRecord payload in {event_path}")
            yield payload


def event_files(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("tensorboard/**/events.out.tfevents.*"))


def scalar_points(run_dir: Path, tag: str) -> list[tuple[int, float]]:
    points: list[tuple[int, float]] = []
    for event_path in event_files(run_dir):
        for payload in iter_tfrecord_payloads(event_path):
            event = EVENT_CLASS.FromString(payload)
            for value in event.summary.value:
                if value.tag == tag:
                    points.append((int(event.step), float(value.simple_value)))

    points.sort(key=lambda item: item[0])
    return points


def scalar_tags(run_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event_path in event_files(run_dir):
        for payload in iter_tfrecord_payloads(event_path):
            event = EVENT_CLASS.FromString(payload)
            for value in event.summary.value:
                if value.tag:
                    counts[value.tag] = counts.get(value.tag, 0) + 1
    return counts


def load_run_points(tag: str) -> list[tuple[str, list[tuple[int, float]]]]:
    runs: list[tuple[str, list[tuple[int, float]]]] = []
    for label, run_dir in DEFAULT_RUNS:
        points = scalar_points(run_dir, tag)
        if not points:
            print(f"warning: no points for tag {tag!r} in {run_dir}")
            continue

        steps = [step for step, _ in points]
        print(f"{label}: {len(points)} points, step {steps[0]}..{steps[-1]}")
        runs.append((label, points))

    if not runs:
        raise ValueError(f"No data found for scalar tag {tag!r}")
    return runs


def plot_reward(runs: list[tuple[str, list[tuple[int, float]]]], output_path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    for label, points in runs:
        steps = [step for step, _ in points]
        rewards = [reward for _, reward in points]
        ax.plot(steps, rewards, linewidth=3.6, label=label)

    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean episode reward")
    apply_2d_style(ax)
    ax.legend(frameon=True)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {output_path}")


def plot_reward_separate(
    runs: list[tuple[str, list[tuple[int, float]]]],
    output_path: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(len(runs), 1, figsize=(7.2, 6.4), sharex=True)
    if len(runs) == 1:
        axes = [axes]

    panel_labels = ["(a)", "(b)", "(c)", "(d)"]
    for index, (ax, (label, points)) in enumerate(zip(axes, runs)):
        steps = [step for step, _ in points]
        rewards = [reward for _, reward in points]
        ax.plot(steps, rewards, linewidth=3.6, label=label)
        ax.set_ylabel("Reward")
        apply_2d_style(ax)
        ax.text(
            0.5,
            -0.58 if index == len(runs) - 1 else -0.24,
            panel_labels[index],
            transform=ax.transAxes,
            ha="center",
            va="top",
        )

    axes[-1].set_xlabel("Training step", labelpad=12)
    fig.tight_layout(h_pad=1.5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {output_path}")


def plot_reward_single_runs(
    runs: list[tuple[str, list[tuple[int, float]]]],
    output_dir: Path,
    dpi: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for label, points in runs:
        steps = [step for step, _ in points]
        rewards = [reward for _, reward in points]
        filename = label.replace(" ", "_") + "_reward_curve.png"
        output_path = output_dir / filename

        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        ax.plot(steps, rewards, linewidth=3.6, label=label)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Mean episode reward")
        apply_2d_style(ax)
        fig.tight_layout()
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {output_path}")


def main() -> None:
    args = parse_args()

    if args.list_tags:
        for label, run_dir in DEFAULT_RUNS:
            print(f"\n{label}: {run_dir}")
            for tag, count in sorted(scalar_tags(run_dir).items()):
                print(f"  {tag}: {count}")
        return

    runs = load_run_points(args.tag)
    plot_reward(runs, args.output, args.dpi)
    plot_reward_separate(runs, args.separate_output, args.dpi)
    plot_reward_single_runs(runs, args.single_output_dir, args.dpi)


if __name__ == "__main__":
    main()
