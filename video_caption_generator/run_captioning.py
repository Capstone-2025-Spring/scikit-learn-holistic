from datetime import datetime

import numpy as np
from classifier.ensemble.predict import predict_tree  # ← 당신의 앙상블 파일에서 import
from utils.feature_extractor import extract_features_from_landmarks
from utils.label_map import label_map


def run_captioning_from_json(json_data):
    def format_relative(ms, base_ms):
        total_seconds = int((ms - base_ms) / 1000)
        mins = total_seconds // 60
        secs = total_seconds % 60
        return f"{mins:02}:{secs:02}"

    holistic_frames = json_data.get("holisticData", [])
    predictions = []
    timestamps = []

    WINDOW_SIZE = 10
    feature_buffer = []

    for frame in holistic_frames:
        landmarks = frame.get("results")
        if landmarks and len(landmarks) == 33:
            ts = frame["timestamp"]
            feature = extract_features_from_landmarks(landmarks)
            feature_buffer.append(feature)

            if len(feature_buffer) >= WINDOW_SIZE:
                window = feature_buffer[-WINDOW_SIZE:]
                x_flat = np.concatenate(window).reshape(1, -1)

                # 앙상블 모델로 예측
                pred = predict_tree(x_flat)
                predictions.append(pred)
                timestamps.append(ts)

    # 구간별 캡션 생성
    caption_lines = []
    if predictions:
        base_time = timestamps[0]
        start_idx = 0
        for i in range(1, len(predictions)):
            if predictions[i] != predictions[i - 1]:
                t1 = format_relative(timestamps[start_idx], base_time)
                t2 = format_relative(timestamps[i - 1], base_time)
                label_id = predictions[i - 1]
                label_name = label_map.get(label_id, f"Label {label_id}")
                caption_lines.append(f"[{t1}~{t2}] {label_name}")
                start_idx = i

        # 마지막 구간
        t1 = format_relative(timestamps[start_idx], base_time)
        t2 = format_relative(timestamps[-1], base_time)
        label_id = predictions[-1]
        label_name = label_map.get(label_id, f"Label {label_id}")
        caption_lines.append(f"[{t1}~{t2}] {label_name}")
    return caption_lines
