"""
action_collect_video.py

Extracts human body keypoints from a video file using MediaPipe Pose
(Tasks API, mediapipe >= 0.10) and saves the sequence to a .npz file.

Usage:
    python action_collect_video.py <video_path> [output_path]
    python action_collect_video.py input.mov output.npz --model heavy

The pose landmarker model is downloaded automatically on first run.

Output .npz keys:
    keypoints       -- (N, 33, 4) normalized image-space landmarks (x, y, z, visibility)
    world_keypoints -- (N, 33, 4) metric world-space landmarks (x, y, z, visibility)
    frame_indices   -- (N,) frame numbers where a pose was detected
    landmark_names  -- (33,) string names for each landmark index
"""

import argparse
import sys
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks import python as mp_python

# 33 landmark names in order (Tasks API PoseLandmark enum)
LANDMARK_NAMES = [lm.name for lm in vision.PoseLandmark]

MODEL_URLS = {
    "lite":  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full":  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}
MODEL_FILES = {k: f"pose_landmarker_{k}.task" for k in MODEL_URLS}


def ensure_model(variant: str) -> str:
    model_path = Path(__file__).parent / MODEL_FILES[variant]
    if not model_path.exists():
        url = MODEL_URLS[variant]
        print(f"Downloading pose landmarker model ({variant}) ...")
        urllib.request.urlretrieve(url, model_path)
        print(f"  Saved to {model_path}")
    return str(model_path)


def extract_keypoints(video_path: str, output_path: str,
                      model_variant: str = "full",
                      min_pose_detection: float = 0.5,
                      min_pose_presence: float = 0.5,
                      min_tracking: float = 0.5,
                      num_poses: int = 1) -> None:

    model_path = ensure_model(model_variant)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open video '{video_path}'")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"Video: {video_path}  |  frames={total_frames}  fps={fps:.2f}")

    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=num_poses,
        min_pose_detection_confidence=min_pose_detection,
        min_pose_presence_confidence=min_pose_presence,
        min_tracking_confidence=min_tracking,
    )

    keypoints = []
    world_keypoints = []
    frame_indices = []

    with vision.PoseLandmarker.create_from_options(options) as detector:
        frame_idx = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(frame_idx * 1000 / fps)

            result = detector.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks:
                # Take the first detected person
                lms  = result.pose_landmarks[0]
                wlms = result.pose_world_landmarks[0]

                kp  = np.array([[l.x, l.y, l.z, l.visibility] for l in lms],  dtype=np.float32)
                wkp = np.array([[l.x, l.y, l.z, l.visibility] for l in wlms], dtype=np.float32)

                keypoints.append(kp)
                world_keypoints.append(wkp)
                frame_indices.append(frame_idx)

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"  processed {frame_idx}/{total_frames} frames, "
                      f"poses detected: {len(frame_indices)}")

    cap.release()

    if not keypoints:
        print("No poses detected in the video.")
        sys.exit(1)

    keypoints_arr       = np.stack(keypoints, axis=0)       # (N, 33, 4)
    world_keypoints_arr = np.stack(world_keypoints, axis=0) # (N, 33, 4)
    frame_indices_arr   = np.array(frame_indices, dtype=np.int32)

    np.savez_compressed(
        output_path,
        keypoints=keypoints_arr,
        world_keypoints=world_keypoints_arr,
        frame_indices=frame_indices_arr,
        landmark_names=np.array(LANDMARK_NAMES),
    )

    print(f"\nSaved {len(frame_indices)} frames to '{output_path}'")
    print(f"  keypoints shape      : {keypoints_arr.shape}")
    print(f"  world_keypoints shape: {world_keypoints_arr.shape}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe Pose keypoints from a video and save to .npz"
    )
    parser.add_argument("video", help="Path to the input video file")
    parser.add_argument("output", nargs="?",
                        help="Output .npz path (default: <video_stem>_keypoints.npz)")
    parser.add_argument("--model", choices=["lite", "full", "heavy"], default="full",
                        help="Model variant (default: full)")
    parser.add_argument("--min-detection", type=float, default=0.5)
    parser.add_argument("--min-presence",  type=float, default=0.5)
    parser.add_argument("--min-tracking",  type=float, default=0.5)
    parser.add_argument("--num-poses",     type=int,   default=1,
                        help="Max number of people to track (default: 1)")
    args = parser.parse_args()

    output_path = args.output or str(
        Path(args.video).parent / f"{Path(args.video).stem}_keypoints.npz"
    )

    extract_keypoints(
        video_path=args.video,
        output_path=output_path,
        model_variant=args.model,
        min_pose_detection=args.min_detection,
        min_pose_presence=args.min_presence,
        min_tracking=args.min_tracking,
        num_poses=args.num_poses,
    )


if __name__ == "__main__":
    main()
