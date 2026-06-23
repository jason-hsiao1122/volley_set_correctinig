"""Detect and track volleyball players with YOLO."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import cv2


PERSON_CLASS_ID = 0


def configure_ultralytics() -> None:
    """Keep Ultralytics settings inside the project folder."""
    os.environ.setdefault("YOLO_CONFIG_DIR", "Ultralytics")


def load_yolo_model(model_path: Path):
    """Load a YOLO model after checking that the weight file exists."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"YOLO model not found: {model_path}\n"
            "Place a YOLO .pt file in models/yolo/ or pass --model path/to/model.pt."
        )

    configure_ultralytics()
    from ultralytics import YOLO

    return YOLO(str(model_path))


def create_csv_writer(csv_path: Path):
    """Create the tracking CSV and write its header row."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file = csv_path.open("w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(
        [
            "frame",
            "track_id",
            "class_id",
            "class_name",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
            "center_x",
            "center_y",
            "width",
            "height",
        ]
    )
    return csv_file, writer


def draw_player_box(frame, track_id: int, confidence: float, box: tuple[int, int, int, int]) -> None:
    """Draw one tracked player on the debug frame."""
    x1, y1, x2, y2 = box
    label = f"ID {track_id} person {confidence:.2f}"

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 80), 2)
    cv2.rectangle(frame, (x1, max(0, y1 - 24)), (x1 + 190, y1), (0, 220, 80), -1)
    cv2.putText(
        frame,
        label,
        (x1 + 4, max(16, y1 - 7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )


def detect_and_track(
    input_path: Path,
    model_path: Path,
    csv_path: Path,
    debug_video_path: Path,
    confidence_threshold: float,
) -> int:
    """Track people in a video and save CSV rows plus a debug video."""
    model = load_yolo_model(model_path)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    debug_video_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(debug_video_path), fourcc, fps, (width, height))
    if not video_writer.isOpened():
        cap.release()
        raise ValueError(f"Could not create debug video: {debug_video_path}")

    csv_file, csv_writer = create_csv_writer(csv_path)

    frame_index = 0
    saved_rows = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            results = model.track(
                frame,
                persist=True,
                classes=[PERSON_CLASS_ID],
                conf=confidence_threshold,
                verbose=False,
            )
            result = results[0]
            boxes = result.boxes

            if boxes is not None and boxes.id is not None:
                xyxy_values = boxes.xyxy.cpu().numpy()
                confidence_values = boxes.conf.cpu().numpy()
                class_values = boxes.cls.cpu().numpy().astype(int)
                track_ids = boxes.id.cpu().numpy().astype(int)

                for xyxy, confidence, class_id, track_id in zip(
                    xyxy_values, confidence_values, class_values, track_ids
                ):
                    x1, y1, x2, y2 = [int(round(value)) for value in xyxy]
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    box_width = x2 - x1
                    box_height = y2 - y1

                    csv_writer.writerow(
                        [
                            frame_index,
                            track_id,
                            class_id,
                            "person",
                            round(float(confidence), 6),
                            x1,
                            y1,
                            x2,
                            y2,
                            round(center_x, 2),
                            round(center_y, 2),
                            box_width,
                            box_height,
                        ]
                    )
                    saved_rows += 1
                    draw_player_box(frame, track_id, float(confidence), (x1, y1, x2, y2))

            video_writer.write(frame)
            frame_index += 1

            if frame_index % 50 == 0:
                print(f"Processed {frame_index}/{total_frames} frames...")

    finally:
        csv_file.close()
        cap.release()
        video_writer.release()

    return saved_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect and track volleyball players in a video clip."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input clip path, for example data/clips/clip_001.mp4",
    )
    parser.add_argument(
        "--model",
        default=Path("models/yolo/yolo11n.pt"),
        type=Path,
        help="Local YOLO .pt model path.",
    )
    parser.add_argument(
        "--conf",
        default=0.25,
        type=float,
        help="Detection confidence threshold.",
    )
    parser.add_argument(
        "--tracks-output",
        type=Path,
        help="Output CSV path. Defaults to data/tracks/{video_name}_tracks.csv",
    )
    parser.add_argument(
        "--debug-output",
        type=Path,
        help="Output debug video path. Defaults to outputs/debug_videos/{video_name}_tracks.mp4",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input video does not exist: {args.input}")
    if not args.input.is_file():
        raise SystemExit(f"Input path is not a file: {args.input}")

    video_name = args.input.stem
    tracks_output = args.tracks_output or Path("data/tracks") / f"{video_name}_tracks.csv"
    debug_output = args.debug_output or Path("outputs/debug_videos") / f"{video_name}_tracks.mp4"

    try:
        saved_rows = detect_and_track(
            input_path=args.input,
            model_path=args.model,
            csv_path=tracks_output,
            debug_video_path=debug_output,
            confidence_threshold=args.conf,
        )
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error

    print(f"Input video: {args.input}")
    print(f"YOLO model: {args.model}")
    print(f"Tracking CSV: {tracks_output}")
    print(f"Debug video: {debug_output}")
    print(f"Saved tracking rows: {saved_rows}")


if __name__ == "__main__":
    main()
