# 📁 util/process_tensor_json.py

import json
import os
from datetime import datetime

import numpy as np

from util.feature_extractor import extract_features_from_landmarks


def process_tensor_json(json_path, label, output_dir="processed_tensor", window_size=10, trim_frame=60):
    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, "r") as f:
        data = json.load(f)

    holistic_frames = data.get("holisticData", [])
    features = []

    for frame in holistic_frames:
        landmarks = frame.get("results")
        if landmarks and len(landmarks) == 33:
            f = extract_features_from_landmarks(landmarks)
            features.append(f)

    features = features[trim_frame:-trim_frame]

    X_tensor, y = [], []
    for i in range(len(features) - window_size + 1):
        window = features[i:i+window_size]
        X_tensor.append(np.stack(window))  # (10, 83)
        y.append(label)

    X_tensor = np.array(X_tensor)         # (N, 10, 83)
    y = np.array(y)                       # (N,)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    prefix = f"{output_dir}/label{label}_{timestamp}"

    np.save(f"{prefix}_X_tensor.npy", X_tensor)
    np.save(f"{prefix}_y.npy", y)

    print(f"✅ Saved to {prefix}_X_tensor.npy / y.npy")
    print(f"Samples: {X_tensor.shape[0]}, Feature shape: {X_tensor.shape[1:]}")

    return X_tensor.shape[0], X_tensor.shape[1:]
