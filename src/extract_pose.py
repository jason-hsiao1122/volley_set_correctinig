"""Extract MediaPipe Pose landmarks from a target-player video."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


Box = tuple[int, int, int, int]

LANDMARK_NAMES = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]

POSE_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (0, 4),
    (4, 5),
    (5, 6),
    (3, 7),
    (6, 8),
    (9, 10),
    (11, 12),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (24, 26),
    (25, 27),
    (26, 28),
    (27, 29),
    (28, 30),
    (29, 31),
    (30, 32),
    (27, 31),
    (28, 32),
]


def build_csv_header() -> list[str]:
    """Create one CSV column set for every MediaPipe Pose landmark."""
    header = ["frame", "pose_detected"]
    for name in LANDMARK_NAMES:
        header.extend(
            [
                f"{name}_x",
                f"{name}_y",
                f"{name}_z",
                f"{name}_visibility",
            ]
        )
    return header


def landmark_row(frame_number: int, pose_landmarks) -> list[float | int | str]:
    """Convert MediaPipe landmarks into one CSV row."""
    if pose_landmarks is None:
        row: list[float | int | str] = [frame_number, 0]
        for _ in LANDMARK_NAMES:
            row.extend([math.nan, math.nan, math.nan, math.nan])
        return row

    row = [frame_number, 1]
    for landmark in pose_landmarks:
        row.extend(
            [
                round(landmark.x, 8),
                round(landmark.y, 8),
                round(landmark.z, 8),
                round(landmark.visibility, 8),
            ]
        )
    return row


def read_crop_metadata(metadata_path: Path) -> dict[int, Box]:
    """Read target boxes in cropped-frame coordinates."""
    boxes: dict[int, Box] = {}

    with metadata_path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"frame", "target_x1", "target_y1", "target_x2", "target_y2"}
        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(
                "Crop metadata CSV must contain: frame, target_x1, target_y1, target_x2, target_y2"
            )

        for row in reader:
            frame_number = int(row["frame"])
            box = (
                int(round(float(row["target_x1"]))),
                int(round(float(row["target_y1"]))),
                int(round(float(row["target_x2"]))),
                int(round(float(row["target_y2"]))),
            )
            if box[2] > box[0] and box[3] > box[1]:
                boxes[frame_number] = box

    return boxes


def pose_to_box(pose_landmarks, frame_width: int, frame_height: int) -> Box | None:
    """Convert visible pose landmarks into a pixel bounding box."""
    points = [
        (landmark.x * frame_width, landmark.y * frame_height)
        for landmark in pose_landmarks
        if landmark.visibility >= 0.2
    ]
    if not points:
        return None

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (
        int(round(max(0, min(xs)))),
        int(round(max(0, min(ys)))),
        int(round(min(frame_width, max(xs)))),
        int(round(min(frame_height, max(ys)))),
    )


def box_center(box: Box) -> tuple[float, float]:
    """Return the center of a pixel box."""
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def box_iou(first_box: Box, second_box: Box) -> float:
    """Return intersection-over-union for two boxes."""
    ax1, ay1, ax2, ay2 = first_box
    bx1, by1, bx2, by2 = second_box

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    intersection = inter_width * inter_height

    first_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    second_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = first_area + second_area - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def get_pose_center(pose_landmarks) -> tuple[float, float] | None:
    """Return a normalized body center for one detected pose."""
    visible_points = [
        landmark
        for landmark in pose_landmarks
        if landmark.visibility >= 0.2
    ]
    if not visible_points:
        return None

    # Prefer hip center because it is more stable than head or hands.
    left_hip = pose_landmarks[23]
    right_hip = pose_landmarks[24]
    if left_hip.visibility >= 0.2 and right_hip.visibility >= 0.2:
        return ((left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2)

    center_x = sum(landmark.x for landmark in visible_points) / len(visible_points)
    center_y = sum(landmark.y for landmark in visible_points) / len(visible_points)
    return center_x, center_y


def choose_center_pose(all_pose_landmarks) -> list | None:
    """Choose the detected pose closest to the center of the cropped frame."""
    return choose_closest_pose(all_pose_landmarks, previous_center=(0.5, 0.5))


def choose_closest_pose(
    all_pose_landmarks,
    previous_center: tuple[float, float] | None,
) -> list | None:
    """Choose the pose whose body center is closest to the previous target pose."""
    if not all_pose_landmarks:
        return None
    if previous_center is None:
        return choose_center_pose(all_pose_landmarks)

    best_pose = None
    best_distance = float("inf")
    previous_x, previous_y = previous_center

    for pose_landmarks in all_pose_landmarks:
        center = get_pose_center(pose_landmarks)
        if center is None:
            continue

        center_x, center_y = center
        distance = (center_x - previous_x) ** 2 + (center_y - previous_y) ** 2
        if distance < best_distance:
            best_distance = distance
            best_pose = pose_landmarks

    return best_pose


def choose_target_poses(all_frames_pose_landmarks: list) -> tuple[list, int | None]:
    """Choose one target pose per frame using a single-pose frame as the seed."""
    target_poses = [None] * len(all_frames_pose_landmarks)
    seed_frame = None

    for frame_number, frame_pose_landmarks in enumerate(all_frames_pose_landmarks):
        if len(frame_pose_landmarks or []) == 1:
            seed_frame = frame_number
            target_poses[frame_number] = frame_pose_landmarks[0]
            break

    if seed_frame is None:
        return target_poses, None

    previous_center = get_pose_center(target_poses[seed_frame])
    for frame_number in range(seed_frame + 1, len(all_frames_pose_landmarks)):
        target_pose = choose_closest_pose(
            all_frames_pose_landmarks[frame_number],
            previous_center,
        )
        target_poses[frame_number] = target_pose
        if target_pose is not None:
            previous_center = get_pose_center(target_pose)

    next_center = get_pose_center(target_poses[seed_frame])
    for frame_number in range(seed_frame - 1, -1, -1):
        target_pose = choose_closest_pose(
            all_frames_pose_landmarks[frame_number],
            next_center,
        )
        target_poses[frame_number] = target_pose
        if target_pose is not None:
            next_center = get_pose_center(target_pose)

    return target_poses, seed_frame


def choose_pose_by_target_box(
    all_pose_landmarks,
    target_box: Box | None,
    frame_width: int,
    frame_height: int,
) -> list | None:
    """Choose the pose that best matches the YOLO target box."""
    if not all_pose_landmarks or target_box is None:
        return None

    target_center_x, target_center_y = box_center(target_box)
    diagonal = (frame_width**2 + frame_height**2) ** 0.5
    best_pose = None
    best_score = -float("inf")

    for pose_landmarks in all_pose_landmarks:
        pose_box = pose_to_box(pose_landmarks, frame_width, frame_height)
        if pose_box is None:
            continue

        pose_center_x, pose_center_y = box_center(pose_box)
        center_distance = (
            (pose_center_x - target_center_x) ** 2
            + (pose_center_y - target_center_y) ** 2
        ) ** 0.5
        normalized_distance = center_distance / diagonal if diagonal > 0 else 1.0
        score = (box_iou(pose_box, target_box) * 5.0) - normalized_distance

        if score > best_score:
            best_score = score
            best_pose = pose_landmarks

    return best_pose


def choose_target_poses_by_metadata(
    all_frames_pose_landmarks: list,
    target_boxes: dict[int, Box],
    frame_width: int,
    frame_height: int,
) -> list:
    """Choose target poses using YOLO target boxes in cropped-frame coordinates."""
    target_poses = []

    for frame_number, frame_pose_landmarks in enumerate(all_frames_pose_landmarks):
        target_pose = choose_pose_by_target_box(
            frame_pose_landmarks,
            target_boxes.get(frame_number),
            frame_width,
            frame_height,
        )
        target_poses.append(target_pose)

    return target_poses


def draw_one_pose(frame, pose_landmarks, color: tuple[int, int, int]) -> None:
    """Draw one pose skeleton on the frame."""
    height, width = frame.shape[:2]
    points: list[tuple[int, int] | None] = []

    for landmark in pose_landmarks:
        if landmark.visibility < 0.2:
            points.append(None)
            continue
        x = int(round(landmark.x * width))
        y = int(round(landmark.y * height))
        points.append((x, y))

    for start_index, end_index in POSE_CONNECTIONS:
        start_point = points[start_index]
        end_point = points[end_index]
        if start_point is not None and end_point is not None:
            cv2.line(frame, start_point, end_point, color, 2)

    for point in points:
        if point is not None:
            cv2.circle(frame, point, 3, (0, 255, 255), -1)


def draw_pose(frame, pose_landmarks, all_pose_landmarks, draw_all_poses: bool) -> None:
    """Draw target pose or all detected poses with a small status label."""
    if draw_all_poses:
        pose_count = len(all_pose_landmarks or [])
        label = f"Poses: {pose_count}"
        color = (0, 220, 80) if pose_count else (0, 0, 255)
        pose_colors = [
            (0, 220, 80),
            (255, 80, 0),
            (255, 0, 255),
            (0, 180, 255),
            (180, 255, 0),
        ]
        for index, detected_pose in enumerate(all_pose_landmarks or []):
            draw_one_pose(frame, detected_pose, pose_colors[index % len(pose_colors)])
    elif pose_landmarks is None:
        label = "Pose: missing"
        color = (0, 0, 255)
    else:
        label = "Pose: detected"
        color = (0, 220, 80)
        draw_one_pose(frame, pose_landmarks, (0, 220, 80))

    cv2.rectangle(frame, (0, 0), (180, 28), (0, 0, 0), -1)
    cv2.putText(
        frame,
        label,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        1,
        cv2.LINE_AA,
    )


def extract_pose(
    input_path: Path,
    output_csv_path: Path,
    output_video_path: Path,
    model_path: Path,
    metadata_path: Path | None,
    num_poses: int,
    draw_all_poses: bool,
    min_detection_confidence: float,
    min_tracking_confidence: float,
) -> tuple[int, int]:
    """Run MediaPipe Pose on a video and save CSV plus annotated video."""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    target_boxes = read_crop_metadata(metadata_path) if metadata_path is not None else {}

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise ValueError(f"Could not create output video: {output_video_path}")

    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=num_poses,
        min_pose_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    pose_landmarker = vision.PoseLandmarker.create_from_options(options)

    try:
        frames = []
        all_frames_pose_landmarks = []
        processed_frames = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(round(processed_frames * 1000 / fps))
            result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)

            frames.append(frame)
            all_frames_pose_landmarks.append(result.pose_landmarks)

            processed_frames += 1
            if processed_frames % 50 == 0:
                print(f"Detected poses in {processed_frames}/{total_frames} frames...")

        if target_boxes:
            target_poses = choose_target_poses_by_metadata(
                all_frames_pose_landmarks,
                target_boxes,
                width,
                height,
            )
            print(f"Using crop metadata target boxes: {metadata_path}")
        else:
            target_poses, seed_frame = choose_target_poses(all_frames_pose_landmarks)
            if seed_frame is None:
                print("No crop metadata or single-pose seed frame found. Saving NaN landmarks for all frames.")
            else:
                print(f"Target pose seed frame: {seed_frame}")

        detected_frames = sum(pose is not None for pose in target_poses)

        with output_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer_csv = csv.writer(csv_file)
            writer_csv.writerow(build_csv_header())

            for frame_number, frame in enumerate(frames):
                pose_landmarks = target_poses[frame_number]
                writer_csv.writerow(landmark_row(frame_number, pose_landmarks))
                draw_pose(
                    frame,
                    pose_landmarks,
                    all_frames_pose_landmarks[frame_number],
                    draw_all_poses,
                )
                writer.write(frame)

                if (frame_number + 1) % 50 == 0:
                    print(f"Wrote {frame_number + 1}/{total_frames} frames...")
    finally:
        pose_landmarker.close()
        cap.release()
        writer.release()

    return processed_frames, detected_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe Pose landmarks from a target-player video."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input cropped target-player video path.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Output pose CSV path. Defaults to data/poses/{video_name}_pose.csv",
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        help="Output annotated video path. Defaults to outputs/debug_videos/{video_name}_pose.mp4",
    )
    parser.add_argument(
        "--model",
        default=Path("models/pose_landmarker.task"),
        type=Path,
        help="MediaPipe Pose Landmarker .task model path.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Crop metadata CSV from crop_target_player.py. Defaults to data/tracks/{video_name}_metadata.csv when it exists.",
    )
    parser.add_argument(
        "--num-poses",
        default=3,
        type=int,
        help="Maximum number of poses to detect before choosing the target pose.",
    )
    parser.add_argument(
        "--draw-all-poses",
        action="store_true",
        help="Draw every detected pose in the annotated video. CSV still saves the tracked target pose.",
    )
    parser.add_argument(
        "--min-detection-confidence",
        default=0.5,
        type=float,
        help="Minimum confidence for initial pose detection.",
    )
    parser.add_argument(
        "--min-tracking-confidence",
        default=0.5,
        type=float,
        help="Minimum confidence for landmark tracking.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input video does not exist: {args.input}")
    if not args.input.is_file():
        raise SystemExit(f"Input path is not a file: {args.input}")
    if not args.model.exists():
        raise SystemExit(f"Pose model does not exist: {args.model}")

    video_name = args.input.stem
    output_csv = args.output_csv or Path("data/poses") / f"{video_name}_pose.csv"
    output_video = (
        args.output_video
        or Path("outputs/debug_videos") / f"{video_name}_pose.mp4"
    )
    metadata_path = args.metadata
    if metadata_path is None:
        default_metadata = Path("data/tracks") / f"{video_name}_metadata.csv"
        if default_metadata.exists():
            metadata_path = default_metadata
    elif not metadata_path.exists():
        raise SystemExit(f"Crop metadata CSV does not exist: {metadata_path}")

    try:
        processed_frames, detected_frames = extract_pose(
            input_path=args.input,
            output_csv_path=output_csv,
            output_video_path=output_video,
            model_path=args.model,
            metadata_path=metadata_path,
            num_poses=args.num_poses,
            draw_all_poses=args.draw_all_poses,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    print(f"Input video: {args.input}")
    print(f"Pose CSV: {output_csv}")
    print(f"Annotated video: {output_video}")
    print(f"Processed frames: {processed_frames}")
    print(f"Pose detected frames: {detected_frames}")
    print(f"Pose missing frames: {processed_frames - detected_frames}")


if __name__ == "__main__":
    main()
