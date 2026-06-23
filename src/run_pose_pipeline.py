"""Run the MVP pipeline from a clip to target pose CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.crop_target_player import crop_target_player
from src.detect_track import detect_and_track
from src.extract_pose import extract_pose


def build_target_segments(
    tracks_path: Path,
    target_ids: list[int],
    output_segments_path: Path,
) -> list[tuple[int, int, int]]:
    """Create target segments from selected YOLO track IDs."""
    target_id_set = set(target_ids)
    priority = {track_id: index for index, track_id in enumerate(target_ids)}
    frame_to_track_id: dict[int, int] = {}

    with tracks_path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            frame_number = int(row["frame"])
            track_id = int(row["track_id"])
            if track_id not in target_id_set:
                continue

            current_track_id = frame_to_track_id.get(frame_number)
            if current_track_id is None:
                frame_to_track_id[frame_number] = track_id
            elif priority[track_id] < priority[current_track_id]:
                frame_to_track_id[frame_number] = track_id

    segments: list[tuple[int, int, int]] = []
    for frame_number in sorted(frame_to_track_id):
        track_id = frame_to_track_id[frame_number]
        if not segments:
            segments.append((frame_number, frame_number, track_id))
            continue

        start_frame, end_frame, current_track_id = segments[-1]
        if frame_number == end_frame + 1 and track_id == current_track_id:
            segments[-1] = (start_frame, frame_number, current_track_id)
        else:
            segments.append((frame_number, frame_number, track_id))

    output_segments_path.parent.mkdir(parents=True, exist_ok=True)
    with output_segments_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["start_frame", "end_frame", "track_id"])
        for start_frame, end_frame, track_id in segments:
            writer.writerow([start_frame, end_frame, track_id])

    return segments


def parse_target_ids(text: str) -> list[int]:
    """Parse comma-separated track IDs."""
    target_ids = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not target_ids:
        raise argparse.ArgumentTypeError("Provide at least one target ID.")
    return target_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run tracking, target crop, and pose extraction for one clip."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input clip path, for example data/clips/clip_001.mp4",
    )
    parser.add_argument(
        "--target-ids",
        required=True,
        type=parse_target_ids,
        help="Comma-separated target track IDs, for example 2,61,206",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        help="Output name prefix. Defaults to {clip_name}_target_{ids}.",
    )
    parser.add_argument(
        "--yolo-model",
        default=Path("models/yolo/yolo11n.pt"),
        type=Path,
        help="YOLO .pt model path.",
    )
    parser.add_argument(
        "--pose-model",
        default=Path("models/pose_landmarker.task"),
        type=Path,
        help="MediaPipe Pose Landmarker .task model path.",
    )
    parser.add_argument(
        "--conf",
        default=0.25,
        type=float,
        help="YOLO detection confidence threshold.",
    )
    parser.add_argument(
        "--padding",
        default=0.15,
        type=float,
        help="Crop padding around the largest target box.",
    )
    parser.add_argument(
        "--num-poses",
        default=3,
        type=int,
        help="Maximum MediaPipe poses per frame.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input clip does not exist: {args.input}")
    if not args.yolo_model.exists():
        raise SystemExit(f"YOLO model does not exist: {args.yolo_model}")
    if not args.pose_model.exists():
        raise SystemExit(f"Pose model does not exist: {args.pose_model}")

    clip_name = args.input.stem
    ids_text = "_".join(str(track_id) for track_id in args.target_ids)
    run_name = args.run_name or f"{clip_name}_target_{ids_text}"

    tracks_path = Path("data/tracks") / f"{run_name}_tracks.csv"
    yolo_debug_video = Path("outputs/debug_videos") / f"{run_name}_tracks.mp4"
    segments_path = Path("data/tracks") / f"{run_name}_segments.csv"
    crop_video = Path("outputs/debug_videos") / f"{run_name}_crop.mp4"
    crop_metadata = Path("data/tracks") / f"{run_name}_crop_metadata.csv"
    pose_csv = Path("data/poses") / f"{run_name}_pose.csv"
    pose_video = Path("outputs/debug_videos") / f"{run_name}_pose.mp4"

    print("Step 1/4: YOLO detect and track")
    track_rows = detect_and_track(
        input_path=args.input,
        model_path=args.yolo_model,
        csv_path=tracks_path,
        debug_video_path=yolo_debug_video,
        confidence_threshold=args.conf,
    )

    print("Step 2/4: Build target segments")
    segments = build_target_segments(
        tracks_path=tracks_path,
        target_ids=args.target_ids,
        output_segments_path=segments_path,
    )
    if not segments:
        raise SystemExit(f"No rows found for target IDs: {args.target_ids}")

    print("Step 3/4: Crop target player without jpg frames")
    saved_frames, reused_frames, crop_width, crop_height = crop_target_player(
        input_path=args.input,
        tracks_path=tracks_path,
        segments_path=segments_path,
        output_video_path=crop_video,
        output_frames_dir=None,
        output_metadata_path=crop_metadata,
        padding=args.padding,
        save_frames=False,
    )

    print("Step 4/4: Extract pose CSV and annotated video")
    processed_frames, detected_frames = extract_pose(
        input_path=crop_video,
        output_csv_path=pose_csv,
        output_video_path=pose_video,
        model_path=args.pose_model,
        metadata_path=crop_metadata,
        num_poses=args.num_poses,
        draw_all_poses=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    print("Pipeline complete")
    print(f"Run name: {run_name}")
    print(f"Track rows: {track_rows}")
    print(f"Target segments: {segments_path}")
    print(f"Crop video: {crop_video}")
    print(f"Crop metadata: {crop_metadata}")
    print(f"Pose CSV: {pose_csv}")
    print(f"Pose annotated video: {pose_video}")
    print(f"Crop frames saved: disabled")
    print(f"Frames processed: {processed_frames}")
    print(f"Frames reused previous box: {reused_frames}")
    print(f"Crop size: {crop_width}x{crop_height}")
    print(f"Pose detected frames: {detected_frames}")
    print(f"Pose missing frames: {processed_frames - detected_frames}")
    print(f"Target IDs: {args.target_ids}")
    print(f"Cropped video frames: {saved_frames}")


if __name__ == "__main__":
    main()
