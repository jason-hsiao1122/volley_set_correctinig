# Volleyball Computer Vision Project

## 專案目標

本專案使用排球比賽影片建立電腦視覺與動作分析流程：

1. 使用 YOLO 偵測並追蹤球員與排球。
2. 使用 MediaPipe 擷取目標球員的身體姿態 landmarks。
3. 將姿態資料轉換成時間序列 features。
4. 訓練 LSTM predictor 與 LSTM autoencoder。
5. 比較不同影片中的動作，輸出分數與視覺化結果。

目前先完成小型 MVP，不一次實作完整系統。

## 資料夾說明

- `data/raw/`：原始影片。
- `data/clips/`：裁切後的短影片。
- `data/frames/`：從影片擷取的影格。
- `data/tracks/`：YOLO 追蹤結果。
- `data/poses/`：MediaPipe 姿態 landmarks。
- `data/features/`：模型訓練使用的 features。
- `src/`：各階段的 Python scripts。
- `notebooks/`：除錯與實驗 notebooks。
- `models/`：模型權重與 checkpoints。
- `outputs/debug_videos/`：含標記的除錯影片。
- `outputs/plots/`：訓練與分析圖表。
- `outputs/reports/`：動作比較報告。

## 建議執行順序

1. `video_io.py`
2. `detect_track.py`
3. `extract_pose.py`
4. `build_features.py`
5. `train_lstm_predictor.py`
6. `visualize.py`
7. `train_lstm_autoencoder.py`
8. `score_motion.py`

## 初始安裝指令

建議使用 Python 3.11 建立虛擬環境：

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

以上指令僅供手動執行；初始化專案時不會自動下載套件。

## MVP 開發路線

1. 讀取短影片並擷取影格。
2. 偵測與追蹤球員，輸出 tracking CSV 與 debug video。
3. 選取目標球員並擷取姿態，輸出 landmark CSV 與 skeleton video。
4. 正規化姿態並建立時間序列 features。
5. 訓練簡單的 next-frame LSTM predictor。
6. 視覺化真實與預測姿態。
7. Predictor 流程穩定後，再加入 autoencoder 與動作評分。
