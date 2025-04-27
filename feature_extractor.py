import numpy as np


def extract_features_from_landmarks(landmarks):
    lm = np.array([[l["x"], l["y"], l["z"]] for l in landmarks])

    # Centering: 좌우 힙 중앙 기준
    center = (lm[23] + lm[24]) / 2
    lm -= center

    # Scaling: 어깨-골반 거리 기반
    left_shoulder = lm[11]
    right_shoulder = lm[12]
    left_hip = lm[23]
    right_hip = lm[24]
    shoulders_center = (left_shoulder + right_shoulder) / 2
    hips_center = (left_hip + right_hip) / 2
    torso_size = np.linalg.norm(shoulders_center - hips_center)

    max_dist = np.max(np.linalg.norm(lm, axis=1))
    pose_size = max(torso_size * 2.5, max_dist)
    pose_size = pose_size + 1e-6

    lm /= pose_size
    lm *= 100

    features = []

    # Pairwise difference feature
    def get_distance(idx1, idx2):
        return lm[idx2] - lm[idx1]

    def get_avg(idx1, idx2):
        return (lm[idx1] + lm[idx2]) * 0.5

    features.extend([
        get_avg(23, 24) - get_avg(11, 12),

        get_distance(11, 13),
        get_distance(12, 14),
        get_distance(13, 15),
        get_distance(14, 16),

        get_distance(23, 25),
        get_distance(24, 26),
        get_distance(25, 27),
        get_distance(26, 28),

        get_distance(11, 15),
        get_distance(12, 16),

        get_distance(23, 27),
        get_distance(24, 28),

        get_distance(23, 15),
        get_distance(24, 16),

        get_distance(11, 27),
        get_distance(12, 28),

        get_distance(13, 14),
        get_distance(25, 26),
        get_distance(15, 16),
        get_distance(27, 28),

        # 추가 크로스 연결
        get_distance(11, 12),
        get_distance(11, 14),
        get_distance(12, 13),
    ])

    # Head-Neck-Torso alignment
    nose = lm[0]
    neck_center = (lm[11] + lm[12]) / 2
    neck_vector = neck_center - nose
    y_axis = np.array([0, -1, 0])
    cos_neck = np.dot(neck_vector, y_axis) / (np.linalg.norm(neck_vector) * np.linalg.norm(y_axis) + 1e-6)
    head_neck_tilt = np.arccos(np.clip(cos_neck, -1.0, 1.0))
    features.append(np.array([head_neck_tilt]))

    # Torso tilt
    torso_vector = shoulders_center - hips_center
    cos_torso = np.dot(torso_vector, y_axis) / (np.linalg.norm(torso_vector) * np.linalg.norm(y_axis) + 1e-6)
    torso_tilt = np.arccos(np.clip(cos_torso, -1.0, 1.0))
    features.append(np.array([torso_tilt]))

    # Arm length ratio
    left_arm = np.linalg.norm(lm[11] - lm[13]) + np.linalg.norm(lm[13] - lm[15])
    right_arm = np.linalg.norm(lm[12] - lm[14]) + np.linalg.norm(lm[14] - lm[16])
    arm_ratio = left_arm / (right_arm + 1e-6)
    features.append(np.array([arm_ratio]))

    # Leg length ratio
    left_leg = np.linalg.norm(lm[23] - lm[25]) + np.linalg.norm(lm[25] - lm[27])
    right_leg = np.linalg.norm(lm[24] - lm[26]) + np.linalg.norm(lm[26] - lm[28])
    leg_ratio = left_leg / (right_leg + 1e-6)
    features.append(np.array([leg_ratio]))

    # Hand to Foot distance
    left_hand_foot = np.linalg.norm(lm[15] - lm[27])
    right_hand_foot = np.linalg.norm(lm[16] - lm[28])
    features.append(np.array([left_hand_foot, right_hand_foot]))

    # Hand vs Shoulder height difference
    left_hand_shoulder_height = lm[15][1] - lm[11][1]
    right_hand_shoulder_height = lm[16][1] - lm[12][1]
    features.append(np.array([left_hand_shoulder_height, right_hand_shoulder_height]))

    # Center shift (neck vs torso)
    neck_y = lm[0][1]
    torso_y = (lm[23][1] + lm[24][1]) / 2
    center_shift = neck_y - torso_y
    features.append(np.array([center_shift]))

    # Elbow and Knee bend angle differences
    def joint_angle(a, b, c):
        ba = lm[a] - lm[b]
        bc = lm[c] - lm[b]
        cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return np.arccos(np.clip(cos, -1.0, 1.0))

    left_elbow_angle = joint_angle(11, 13, 15)
    right_elbow_angle = joint_angle(12, 14, 16)
    elbow_angle_diff = left_elbow_angle - right_elbow_angle

    left_knee_angle = joint_angle(23, 25, 27)
    right_knee_angle = joint_angle(24, 26, 28)
    knee_angle_diff = left_knee_angle - right_knee_angle

    features.append(np.array([elbow_angle_diff, knee_angle_diff]))

    features = np.concatenate(features)

    return features
   