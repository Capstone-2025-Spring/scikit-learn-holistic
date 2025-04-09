import numpy as np


def extract_features_from_landmarks(landmarks):
    lm = np.array([[l["x"], l["y"], l["z"]] for l in landmarks])
    center = lm[23]  # left hip 기준 정규화
    lm -= center
    scale = np.linalg.norm(lm[11] - lm[12]) + 1e-6
    lm /= scale

    features = []
    features.extend(lm.flatten())  # 33 * 3 = 99차원

    def distance(i, j):
        return np.linalg.norm(lm[i] - lm[j])

    pair_indices = [
        (11, 13), (13, 15), (12, 14), (14, 16),
        (15, 0), (16, 0), (15, 11), (16, 12),
        (11, 12), (23, 24), (11, 23), (12, 24),
        (25, 27), (26, 28), (27, 31), (28, 32), (15, 16),
    ]
    features.extend([distance(i, j) for i, j in pair_indices])

    def angle(a, b, c):
        ba = lm[a] - lm[b]
        bc = lm[c] - lm[b]
        cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return np.arccos(np.clip(cos, -1.0, 1.0))

    angle_triplets = [
        (11, 13, 15), (12, 14, 16), (13, 11, 23), (14, 12, 24),
        (23, 25, 27), (24, 26, 28), (11, 23, 24)
    ]
    features.extend([angle(a, b, c) for a, b, c in angle_triplets])

    nose = lm[0]
    shoulder_center = (lm[11] + lm[12]) / 2
    torso_vector = np.array([0, -1, 0])
    neck_vector = nose - shoulder_center
    cos = np.dot(neck_vector, torso_vector) / (np.linalg.norm(neck_vector) * np.linalg.norm(torso_vector) + 1e-6)
    head_tilt = np.arccos(np.clip(cos, -1.0, 1.0))
    features.append(head_tilt)

    shoulder_vec = lm[11] - lm[12]
    shoulder_slope = np.arctan2(shoulder_vec[1], shoulder_vec[0])
    features.append(shoulder_slope)

    visibility = np.array([l["visibility"] for l in landmarks])
    features.append(np.mean(visibility))
    features.append(np.min(visibility))
    features.append(np.max(visibility))

    return np.array(features)
