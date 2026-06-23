"""Read a video and extract frames for the volleyball CV pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def get_video_info(video_path: Path) -> dict[str, float | int]:
    """Return basic video information from OpenCV."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0.0
    cap.release()

    return {
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "duration": duration,
    }


def extract_frames(video_path: Path, output_root: Path, target_fps: float) -> int:
    """Extract frames from a video at the requested FPS."""
    if target_fps <= 0:
        raise ValueError("--fps must be greater than 0")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        cap.release()
        raise ValueError(f"Could not read FPS from video: {video_path}")

    output_dir = output_root / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_interval = max(1, round(source_fps / target_fps))
    frame_index = 0
    saved_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index % frame_interval == 0:
            frame_name = f"frame_{saved_count:06d}.jpg"
            frame_path = output_dir / frame_name
            cv2.imwrite(str(frame_path), frame)
            saved_count += 1

        frame_index += 1

    cap.release()
    return saved_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read a volleyball video and extract frames."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input video path, for example data/raw/sample.mp4",
    )
    parser.add_argument(
        "--fps",
        required=True,
        type=float,
        help="Target frame extraction FPS, for example 10",
    )
    parser.add_argument(
        "--output-root",
        default=Path("data/frames"),
        type=Path,
        help="Root folder for extracted frames.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = args.input

    if not video_path.exists():
        raise SystemExit(f"Input video does not exist: {video_path}")
    if not video_path.is_file():
        raise SystemExit(f"Input path is not a file: {video_path}")

    try:
        info = get_video_info(video_path)
        saved_count = extract_frames(video_path, args.output_root, args.fps)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    output_dir = args.output_root / video_path.stem

    print(f"Input video: {video_path}")
    print(f"Source FPS: {info['fps']:.3f}")
    print(f"Total frames: {info['total_frames']}")
    print(f"Resolution: {info['width']}x{info['height']}")
    print(f"Duration: {info['duration']:.2f} seconds")
    print(f"Target FPS: {args.fps}")
    print(f"Saved frames: {saved_count}")
    print(f"Output folder: {output_dir}")


if __name__ == "__main__":
    main()
