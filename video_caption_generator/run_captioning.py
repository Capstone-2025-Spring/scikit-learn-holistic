import importlib.util
import os
from datetime import datetime

import numpy as np
import torch
from utils.feature_extractor import extract_features_from_landmarks
from utils.label_map import label_map

##########################################################
# === 모델 로딩 (초기화 시 1회만) ===
MODEL_PATH = "model/v1.pt"
CLASSIFIER_PATH = "classifier/v1_classifier.py"
##########################################################

def load_classifier(module_path):
    spec = importlib.util.spec_from_file_location("ClassifierModule", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Classifier

checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
Classifier = load_classifier(CLASSIFIER_PATH)
model = Classifier(input_dim=checkpoint['input_dim'], num_classes=checkpoint['num_classes'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

def run_captioning_from_json(json_data):
    from datetime import timedelta

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
                input_feature = np.concatenate(window).reshape(1, -1)
                input_tensor = torch.tensor(input_feature, dtype=torch.float32)

                with torch.no_grad():
                    pred = model(input_tensor).argmax().item()

                predictions.append(pred)
                timestamps.append(ts)

    caption_lines = []
    if predictions:
        base_time = timestamps[0]  # 기준 시간 설정
        start_idx = 0
        for i in range(1, len(predictions)):
            if predictions[i] != predictions[i - 1]:
                t1 = format_relative(timestamps[start_idx], base_time)
                t2 = format_relative(timestamps[i-1], base_time)
                label_id = predictions[i - 1]
                caption_lines.append(f"[{t1}:{t2}] {label_id}")
                start_idx = i
        t1 = format_relative(timestamps[start_idx], base_time)
        t2 = format_relative(timestamps[-1], base_time)
        label_id = predictions[-1]
        caption_lines.append(f"[{t1}:{t2}] {label_id}")

    return caption_lines

  
def format_ms(ms):
    return datetime.fromtimestamp(ms / 1000).strftime("%H:%M:%S")