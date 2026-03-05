#!/usr/bin/env python3
"""Play Alex V1 velocity demo from a local checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import mjlab.tasks  # noqa: F401  # Registers tasks.
from mjlab.scripts.play import PlayConfig, run_play

DEFAULT_TASK = "Mjlab-Velocity-Flat-Alex-V1"
DEFAULT_CHECKPOINT = (
  "Mjlab-Velocity-Flat-Alex-V1/model.pt"
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Load an Alex checkpoint and play a demo rollout."
  )
  parser.add_argument("--task", default=DEFAULT_TASK, help="Task id to play.")
  parser.add_argument(
    "--checkpoint",
    default=DEFAULT_CHECKPOINT,
    help="Path to checkpoint (.pt) file.",
  )
  parser.add_argument(
    "--viewer",
    choices=("auto", "native", "viser"),
    default="auto",
    help="Viewer backend.",
  )
  parser.add_argument("--num-envs", type=int, default=None, help="Override num_envs.")
  parser.add_argument("--device", default=None, help='Torch device, e.g. "cuda:0".')
  parser.add_argument(
    "--video",
    action="store_true",
    help="Record a video to the checkpoint run directory.",
  )
  parser.add_argument(
    "--video-length",
    type=int,
    default=200,
    help="Video length in environment steps.",
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  checkpoint = Path(args.checkpoint).expanduser().resolve()
  if not checkpoint.exists():
    raise FileNotFoundError(f"Checkpoint file not found: {checkpoint}")

  run_play(
    args.task,
    PlayConfig(
      checkpoint_file=str(checkpoint),
      viewer=args.viewer,
      num_envs=args.num_envs,
      device=args.device,
      video=args.video,
      video_length=args.video_length,
    ),
  )


if __name__ == "__main__":
  main()
