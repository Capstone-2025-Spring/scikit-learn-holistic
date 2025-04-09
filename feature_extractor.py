import numpy as np


def extract_features_from_landmarks(landmarks):
    lm = np.array([[l["x"], l["y"], l["z"]] for l in landmarks])
    center = lm[23]  # left hip 기준 정규화
    lm -= center
    scale = np.linalg.norm(lm[11] - lm[12]) + 1e-6  # 어깨 폭
    lm /= scale

    features = []

    # 1. Flattened normalized coordinates (33 x 3 = 99)
    features.extend(lm.flatten())

    # 2. 유의미한 거리 기반 피처
    def distance(i, j):
        return np.linalg.norm(lm[i] - lm[j])

    pair_indices = [
        (11, 13), (13, 15),  # 왼팔
        (12, 14), (14, 16),  # 오른팔
        (15, 0), (16, 0),    # 손목-머리
        (15, 11), (16, 12),  # 손목-어깨
        (11, 12),            # 어깨너비
        (23, 24),            # 힙 너비
        (11, 23), (12, 24),  # 어깨-힙
        (25, 27), (26, 28),  # 무릎-발목
        (27, 31), (28, 32),  # 발목-발끝
        (15, 16),            # 양손 사이 거리
    ]
    features.extend([distance(i, j) for i, j in pair_indices])

    # 3. 관절 각도 기반 피처
    def angle(a, b, c):
        ba = lm[a] - lm[b]
        bc = lm[c] - lm[b]
        cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return np.arccos(np.clip(cos, -1.0, 1.0))

    angle_triplets = [
        (11, 13, 15),  # 왼팔
        (12, 14, 16),  # 오른팔
        (13, 11, 23),  # 왼쪽 어깨 각도
        (14, 12, 24),  # 오른쪽 어깨 각도
        (23, 25, 27),  # 왼쪽 다리
        (24, 26, 28),  # 오른쪽 다리
        (11, 23, 24),  # 상체-골반 기울기
    ]
    features.extend([angle(a, b, c) for a, b, c in angle_triplets])

    # 4. 고개 숙임 각도
    nose = lm[0]
    shoulder_center = (lm[11] + lm[12]) / 2
    torso_vector = np.array([0, -1, 0])  # 수직 기준
    neck_vector = nose - shoulder_center
    cos = np.dot(neck_vector, torso_vector) / (np.linalg.norm(neck_vector) * np.linalg.norm(torso_vector) + 1e-6)
    head_tilt = np.arccos(np.clip(cos, -1.0, 1.0))
    features.append(head_tilt)

    # 5. 어깨 기울기
    shoulder_vec = lm[11] - lm[12]
    shoulder_slope = np.arctan2(shoulder_vec[1], shoulder_vec[0])
    features.append(shoulder_slope)

    # 6. visibility 통계
    visibility = np.array([l["visibility"] for l in landmarks])
    features.append(np.mean(visibility))
    features.append(np.min(visibility))
    features.append(np.max(visibility))

    return np.array(features)
