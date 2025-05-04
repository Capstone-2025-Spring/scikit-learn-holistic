import json
import os
import sys
from datetime import datetime

import numpy as np

from util.feature_extractor import extract_features_from_landmarks  # ✅ import


def process_pose_json(json_path, label, output_dir="processed_data", window_size=10, trim_frame=60):
    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, "r") as f:
        data = json.load(f)

    holistic_frames = data.get("holisticData", [])
    features = []

    for frame in holistic_frames:
        landmarks = frame.get("results")
        if landmarks and len(landmarks) == 33:
            f=extract_features_from_landmarks(landmarks)
            features.append(f)

    features = features[trim_frame:-trim_frame]  # 앞뒤 프레임 자르기

    X, y = [], []
    for i in range(len(features) - window_size + 1):
        window = features[i:i+window_size]
        X.append(np.concatenate(window))
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    prefix = f"{output_dir}/label{label}_{timestamp}"

    np.save(f"{prefix}_X.npy", X)
    np.save(f"{prefix}_y.npy", y)

    print(f"✅ Saved to {prefix}_X.npy / y.npy")
    print(f"Samples: {X.shape[0]}, Feature size: {X.shape[1]}")

    return X.shape[0], X.shape[1]


# 기존 코드 생략...

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python process_pose_json.py [json_path] [label]")
        sys.exit(1)

    json_path = sys.argv[1]
    label = int(sys.argv[2])

    process_pose_json(json_path, label)
