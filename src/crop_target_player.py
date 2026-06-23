"""Crop one target player from a video using tracking CSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


Box = tuple[int, int, int, int]


def read_target_segments(segments_path: Path) -> dict[int, int]:
    """Return a frame-to-track-id lookup from a target segments CSV."""
    frame_to_track_id: dict[int, int] = {}

    with segments_path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"start_frame", "end_frame", "track_id"}
        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(
                "Target segments CSV must contain: start_frame, end_frame, track_id"
            )

        for row in reader:
            start_frame = int(row["start_frame"])
            end_frame = int(row["end_frame"])
            track_id = int(row["track_id"])

            for frame_number in range(start_frame, end_frame + 1):
                frame_to_track_id[frame_number] = track_id

    return frame_to_track_id


def read_tracking_boxes(tracks_path: Path) -> dict[tuple[int, int], Box]:
    """Return a lookup from (frame, track_id) to a bounding box."""
    boxes: dict[tuple[int, int], Box] = {}

    with tracks_path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"frame", "track_id", "x1", "y1", "x2", "y2"}
        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError("Tracking CSV must contain: frame, track_id, x1, y1, x2, y2")

        for row in reader:
            frame_number = int(row["frame"])
            track_id = int(row["track_id"])
            x1 = int(round(float(row["x1"])))
            y1 = int(round(float(row["y1"])))
            x2 = int(round(float(row["x2"])))
            y2 = int(round(float(row["y2"])))
            boxes[(frame_number, track_id)] = (x1, y1, x2, y2)

    return boxes


def choose_crop_size(
    frame_to_track_id: dict[int, int],
    tracking_boxes: dict[tuple[int, int], Box],
    padding: float,
) -> tuple[int, int]:
    """Choose a fixed crop size using the largest target box."""
    max_width = 1
    max_height = 1

    for frame_number, track_id in frame_to_track_id.items():
        box = tracking_boxes.get((frame_number, track_id))
        if box is None:
            continue

        x1, y1, x2, y2 = box
        max_width = max(max_width, x2 - x1)
        max_height = max(max_height, y2 - y1)

    crop_width = int(round(max_width * (1 + padding)))
    crop_height = int(round(max_height * (1 + padding)))

    crop_width = max(2, crop_width)
    crop_height = max(2, crop_height)
    if crop_width % 2 == 1:
        crop_width += 1
    if crop_height % 2 == 1:
        crop_height += 1

    return crop_width, crop_height


def crop_fixed_size(frame, box: Box, crop_width: int, crop_height: int):
    """Crop around a box center and return the box location inside the crop."""
    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    crop_x1 = center_x - crop_width // 2
    crop_y1 = center_y - crop_height // 2
    crop_x2 = crop_x1 + crop_width
    crop_y2 = crop_y1 + crop_height

    source_x1 = max(0, crop_x1)
    source_y1 = max(0, crop_y1)
    source_x2 = min(frame_width, crop_x2)
    source_y2 = min(frame_height, crop_y2)

    target_x1 = source_x1 - crop_x1
    target_y1 = source_y1 - crop_y1
    target_x2 = target_x1 + (source_x2 - source_x1)
    target_y2 = target_y1 + (source_y2 - source_y1)

    cropped = np.zeros((crop_height, crop_width, 3), dtype=frame.dtype)
    cropped[target_y1:target_y2, target_x1:target_x2] = frame[
        source_y1:source_y2, source_x1:source_x2
    ]

    target_box = (
        max(0, min(crop_width, x1 - crop_x1)),
        max(0, min(crop_height, y1 - crop_y1)),
        max(0, min(crop_width, x2 - crop_x1)),
        max(0, min(crop_height, y2 - crop_y1)),
    )
    return cropped, target_box


def draw_status(cropped_frame, frame_number: int, track_id: int | None, reused_previous: bool) -> None:
    """Draw frame status on the cropped debug frame."""
    if reused_previous:
        label = f"Frame {frame_number} | reused box"
    elif track_id is None:
        label = f"Frame {frame_number} | no target"
    else:
        label = f"Frame {frame_number} | ID {track_id}"

    cv2.rectangle(cropped_frame, (0, 0), (min(cropped_frame.shape[1], 420), 28), (0, 0, 0), -1)
    cv2.putText(
        cropped_frame,
        label,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def crop_target_player(
    input_path: Path,
    tracks_path: Path,
    segments_path: Path,
    output_video_path: Path,
    output_frames_dir: Path | None,
    output_metadata_path: Path,
    padding: float,
    save_frames: bool = True,
) -> tuple[int, int, int, int]:
    """Crop the target player for every frame in the input video."""
    frame_to_track_id = read_target_segments(segments_path)
    tracking_boxes = read_tracking_boxes(tracks_path)
    crop_width, crop_height = choose_crop_size(frame_to_track_id, tracking_boxes, padding)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    if save_frames and output_frames_dir is not None:
        output_frames_dir.mkdir(parents=True, exist_ok=True)
    output_metadata_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (crop_width, crop_height))
    if not writer.isOpened():
        cap.release()
        raise ValueError(f"Could not create output video: {output_video_path}")

    frame_number = 0
    saved_frames = 0
    reused_frames = 0
    last_box: Box | None = None

    metadata_file = output_metadata_path.open("w", newline="", encoding="utf-8")
    metadata_writer = csv.writer(metadata_file)
    metadata_writer.writerow(
        [
            "frame",
            "track_id",
            "reused_box",
            "crop_width",
            "crop_height",
            "target_x1",
            "target_y1",
            "target_x2",
            "target_y2",
        ]
    )

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            track_id = frame_to_track_id.get(frame_number)
            current_box = None
            reused_previous = False

            if track_id is not None:
                current_box = tracking_boxes.get((frame_number, track_id))

            if current_box is None and last_box is not None:
                current_box = last_box
                reused_previous = True
                reused_frames += 1

            if current_box is None:
                cropped = np.zeros((crop_height, crop_width, 3), dtype=frame.dtype)
                target_box = (0, 0, 0, 0)
            else:
                cropped, target_box = crop_fixed_size(frame, current_box, crop_width, crop_height)
                last_box = current_box

            draw_status(cropped, frame_number, track_id, reused_previous)
            writer.write(cropped)
            metadata_writer.writerow(
                [
                    frame_number,
                    track_id if track_id is not None else "",
                    int(reused_previous),
                    crop_width,
                    crop_height,
                    target_box[0],
                    target_box[1],
                    target_box[2],
                    target_box[3],
                ]
            )

            if save_frames and output_frames_dir is not None:
                frame_path = output_frames_dir / f"frame_{frame_number:06d}.jpg"
                cv2.imwrite(str(frame_path), cropped)

            saved_frames += 1
            frame_number += 1

            if frame_number % 50 == 0:
                print(f"Processed {frame_number}/{total_frames} frames...")

    finally:
        metadata_file.close()
        cap.release()
        writer.release()

    return saved_frames, reused_frames, crop_width, crop_height


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop one target volleyball player from a tracked video."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input clip path, for example data/clips/clip_001.mp4",
    )
    parser.add_argument(
        "--tracks",
        required=True,
        type=Path,
        help="Tracking CSV path from detect_track.py.",
    )
    parser.add_argument(
        "--target-segments",
        required=True,
        type=Path,
        help="CSV with start_frame,end_frame,track_id rows.",
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        help="Output cropped video path.",
    )
    parser.add_argument(
        "--output-frames",
        type=Path,
        help="Output folder for cropped frames.",
    )
    parser.add_argument(
        "--no-save-frames",
        action="store_true",
        help="Do not save cropped frame jpg files.",
    )
    parser.add_argument(
        "--output-metadata",
        type=Path,
        help="Output metadata CSV path for target boxes inside cropped frames.",
    )
    parser.add_argument(
        "--padding",
        default=0.15,
        type=float,
        help="Extra crop padding around the largest target box. Example: 0.15",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for path, label in [
        (args.input, "Input video"),
        (args.tracks, "Tracking CSV"),
        (args.target_segments, "Target segments CSV"),
    ]:
        if not path.exists():
            raise SystemExit(f"{label} does not exist: {path}")

    video_name = args.input.stem
    output_video = args.output_video or Path("outputs/debug_videos") / f"{video_name}_target_crop.mp4"
    output_frames = args.output_frames or Path("data/frames") / f"{video_name}_target_crop"
    output_metadata = (
        args.output_metadata
        or Path("data/tracks") / f"{video_name}_target_crop_metadata.csv"
    )

    try:
        saved_frames, reused_frames, crop_width, crop_height = crop_target_player(
            input_path=args.input,
            tracks_path=args.tracks,
            segments_path=args.target_segments,
            output_video_path=output_video,
            output_frames_dir=output_frames,
            output_metadata_path=output_metadata,
            padding=args.padding,
            save_frames=not args.no_save_frames,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    print(f"Input video: {args.input}")
    print(f"Tracking CSV: {args.tracks}")
    print(f"Target segments: {args.target_segments}")
    print(f"Output video: {output_video}")
    print(f"Output frames: {output_frames if not args.no_save_frames else 'disabled'}")
    print(f"Output metadata: {output_metadata}")
    print(f"Saved frames: {saved_frames}")
    print(f"Frames using previous box: {reused_frames}")
    print(f"Fixed crop size: {crop_width}x{crop_height}")


if __name__ == "__main__":
    main()
