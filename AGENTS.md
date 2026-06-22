# AGENTS.md

## Project Goal

This project is a volleyball computer vision project.

The final goal is to:

1. Use international volleyball match videos as input.
2. Detect and track volleyball players and the ball.
3. Extract body pose landmarks from a selected player.
4. Convert pose landmarks into motion time-series features.
5. Train sequence models such as LSTM predictor or LSTM autoencoder.
6. Compare motion patterns between different videos and produce a similarity or performance score.

## Current MVP

The first MVP should only focus on:

1. Read a short volleyball video clip.
2. Detect and track players.
3. Select one target player.
4. Extract body pose landmarks using MediaPipe Pose.
5. Save intermediate results as CSV files.
6. Train a simple LSTM model to predict the next-frame pose.
7. Generate debug videos and plots.

Do not implement the full final system at once.

## Development Rules

* Use Python 3.11.
* Keep the code simple and beginner-friendly.
* Prefer readable code over complex architecture.
* Do not hard-code absolute Windows paths.
* Use command-line arguments for all scripts.
* Save intermediate results to files.
* Every computer vision step should generate a visual debug output.
* Do not modify files outside this project folder.
* Do not delete raw data, videos, trained models, or output files unless explicitly instructed.
* Ask for confirmation before installing new packages.
* Ask for confirmation before using internet access.
* Ask for confirmation before making large structural changes to the project.

## Folder Structure

Use this structure:

```text
volleyball_cv_project/
├── data/
│   ├── raw/              # original videos, do not commit
│   ├── clips/            # short clips
│   ├── frames/           # extracted frames
│   ├── tracks/           # YOLO tracking CSV
│   ├── poses/            # MediaPipe landmark CSV
│   └── features/         # training tensors or feature files
├── models/
│   ├── yolo/
│   ├── lstm_predictor/
│   └── lstm_autoencoder/
├── notebooks/
├── outputs/
│   ├── debug_videos/
│   ├── plots/
│   └── reports/
├── src/
├── README.md
├── requirements.txt
└── AGENTS.md
```

## Coding Style

* Write clear functions with meaningful names.
* Add short comments for beginner-level understanding.
* Avoid overly clever code.
* Prefer small modules over one large script.
* Print useful progress messages.
* Handle missing files and invalid inputs gracefully.
* Do not crash when pose landmarks are missing; save NaN values instead.

## Data Rules

* Raw videos should stay in `data/raw/`.
* Short clips should stay in `data/clips/`.
* Tracking results should stay in `data/tracks/`.
* Pose results should stay in `data/poses/`.
* Features should stay in `data/features/`.
* Debug videos should stay in `outputs/debug_videos/`.
* Plots should stay in `outputs/plots/`.

Do not commit large videos, model weights, or generated outputs.

## Recommended Pipeline

Build the project in this order:

1. `src/video_io.py`

   * Read video.
   * Extract frames.
   * Print video information.

2. `src/detect_track.py`

   * Use YOLO to detect and track players.
   * Save tracking CSV.
   * Save debug video with boxes and track IDs.

3. `src/extract_pose.py`

   * Use tracking CSV to crop the target player.
   * Run MediaPipe Pose.
   * Save landmark CSV.
   * Save debug video with skeleton overlay.

4. `src/build_features.py`

   * Normalize pose coordinates.
   * Compute velocity features.
   * Build sequence tensors.

5. `src/train_lstm_predictor.py`

   * Train a simple LSTM model.
   * Predict next-frame pose.
   * Save checkpoint and loss plot.

6. `src/visualize_prediction.py`

   * Overlay real skeleton and predicted skeleton.
   * Save prediction debug video.

7. `src/train_autoencoder.py`

   * Train LSTM autoencoder only after the predictor pipeline works.

8. `src/score_motion.py`

   * Compare different videos only after pose extraction and model training are stable.

## Model Rules

For the first version:

* Use body landmarks only.
* Do not use hand or finger landmarks until the body pose pipeline is stable.
* Use 2D pose coordinates first.
* Normalize pose coordinates using hip center as origin.
* Normalize body scale using shoulder width or torso length.
* Start with a simple LSTM predictor before building an LSTM autoencoder.

## Prompting Rules for Codex

When modifying code:

1. Explain what files will be changed before editing.
2. Modify only the files related to the current task.
3. After editing, summarize:

   * Files changed
   * How to run the script
   * Expected outputs
   * Possible failure points

Do not rewrite the whole project unless explicitly instructed.

## Safety Rules

* Do not delete files unless explicitly instructed.
* Do not run commands outside the project folder unless approved.
* Do not use full system access unless approved.
* Do not upload private files or videos anywhere.
* Do not expose API keys, tokens, or personal paths.
* Do not add cloud services unless explicitly requested.

## First Task Recommendation

If the user has not specified a task, start with:

Create `src/video_io.py` that:

* reads an input video,
* extracts frames at a target FPS,
* saves frames to `data/frames/{video_name}/`,
* prints FPS, total frames, duration, and saved frame count,
* provides this CLI example:

```bash
python -m src.video_io --input data/raw/sample.mp4 --fps 10
```
