from datetime import datetime

import numpy as np
from utils.feature_extractor import extract_features_from_landmarks
from utils.label_map import label_map

from classifier.ensemble.predict import predict_tree  # ← 당신의 앙상블 파일에서 import


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

                pred = predict_tree(x_flat)
                predictions.append(pred)
                timestamps.append(ts)

    # ✅ 캡션 구간 병합 포함
    caption_lines = []
    if predictions:
        base_time = timestamps[0]
        i = 0
        last_label = None
        last_start = None
        last_end = None

        while i < len(predictions):
            label = predictions[i]
            start_ts = timestamps[i]
            j = i + 1
            while j < len(predictions) and predictions[j] == label:
                j += 1
            end_ts = timestamps[j - 1]

            # 너무 짧은 구간은 버림
            if end_ts - start_ts < 500:
                i = j
                continue

            # ✅ 이전 구간과 병합 조건: label 같고 시간 gap이 1초 이내
            if (
                last_label == label
                and start_ts - last_end <= 1000
            ):
                last_end = end_ts  # 병합 확장
            else:
                # 기존 구간 저장
                if last_label is not None:
                    t1 = format_relative(last_start, base_time)
                    t2 = format_relative(last_end, base_time)
                    caption_lines.append(f"[{t1}:{t2}] {last_label}")

                # 새로운 구간 시작
                last_label = label
                last_start = start_ts
                last_end = end_ts

            i = j

        # 마지막 구간 저장
        if last_label is not None:
            t1 = format_relative(last_start, base_time)
            t2 = format_relative(last_end, base_time)
            caption_lines.append(f"[{t1}:{t2}] {last_label}")

    return caption_lines
