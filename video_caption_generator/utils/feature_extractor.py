import numpy as np


def extract_features_from_landmarks(landmarks):
    lm = np.array([[l["x"], l["y"], l["z"]] for l in landmarks])

    # Centering
    center = (lm[23] + lm[24]) / 2
    lm -= center

    # Scaling
    shoulders_center = (lm[11] + lm[12]) / 2
    hips_center = (lm[23] + lm[24]) / 2
    torso_size = np.linalg.norm(shoulders_center - hips_center)
    max_dist = np.max(np.linalg.norm(lm, axis=1))
    pose_size = max(torso_size * 2.5, max_dist) + 1e-6
    lm /= pose_size
    lm *= 100

    features = []

    def get_distance(idx1, idx2):
        return lm[idx2] - lm[idx1]

    def get_avg(idx1, idx2):
        return (lm[idx1] + lm[idx2]) * 0.5

    features.extend([
        get_avg(23, 24) - get_avg(11, 12),
        get_distance(11, 13), get_distance(12, 14),
        get_distance(13, 15), get_distance(14, 16),
        get_distance(23, 25), get_distance(24, 26),
        get_distance(25, 27), get_distance(26, 28),
        get_distance(11, 15), get_distance(12, 16),
        get_distance(23, 27), get_distance(24, 28),
        get_distance(23, 15), get_distance(24, 16),
        get_distance(11, 27), get_distance(12, 28),
        get_distance(13, 14), get_distance(25, 26),
        get_distance(15, 16), get_distance(27, 28),
        get_distance(11, 12), get_distance(11, 14),
        get_distance(12, 13),
    ])

    def joint_angle(a, b, c):
        ba = lm[a] - lm[b]
        bc = lm[c] - lm[b]
        cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return np.arccos(np.clip(cos, -1.0, 1.0))

    angle_joints = [
        (11, 13, 15), (12, 14, 16),  # elbows
        (23, 25, 27), (24, 26, 28),  # knees
        (11, 13, 23), (12, 14, 24),
        (0, 11, 12), (0, (11+12)//2, (23+24)//2),
        (11, 15, 23), (12, 16, 24),
        (11, 23, 25), (12, 24, 26),
        (0, (11+12)//2, 23), (0, (11+12)//2, 24),
        (23, 25, 27), (24, 26, 28),
    ]

    for a, b, c in angle_joints:
        angle = joint_angle(a, b, c)
        features.append(np.array([angle]))

    # Head-Neck tilt
    nose = lm[0]
    neck_center = (lm[11] + lm[12]) / 2
    y_axis = np.array([0, -1, 0])
    neck_vector = neck_center - nose
    cos_neck = np.dot(neck_vector, y_axis) / (np.linalg.norm(neck_vector) * np.linalg.norm(y_axis) + 1e-6)
    head_neck_tilt = np.arccos(np.clip(cos_neck, -1.0, 1.0))
    features.append(np.array([head_neck_tilt]))

    # Torso tilt
    torso_vector = shoulders_center - hips_center
    cos_torso = np.dot(torso_vector, y_axis) / (np.linalg.norm(torso_vector) * np.linalg.norm(y_axis) + 1e-6)
    torso_tilt = np.arccos(np.clip(cos_torso, -1.0, 1.0))
    features.append(np.array([torso_tilt]))

    # Limb ratios
    left_arm = np.linalg.norm(lm[11] - lm[13]) + np.linalg.norm(lm[13] - lm[15])
    right_arm = np.linalg.norm(lm[12] - lm[14]) + np.linalg.norm(lm[14] - lm[16])
    features.append(np.array([left_arm / (right_arm + 1e-6)]))

    left_leg = np.linalg.norm(lm[23] - lm[25]) + np.linalg.norm(lm[25] - lm[27])
    right_leg = np.linalg.norm(lm[24] - lm[26]) + np.linalg.norm(lm[26] - lm[28])
    features.append(np.array([left_leg / (right_leg + 1e-6)]))

    # Hand to foot distance
    features.append(np.array([
        np.linalg.norm(lm[15] - lm[27]),
        np.linalg.norm(lm[16] - lm[28])
    ]))

    # Hand vs Shoulder height
    features.append(np.array([
        lm[15][1] - lm[11][1],
        lm[16][1] - lm[12][1]
    ]))

    # Neck vs torso Y shift
    neck_y = lm[0][1]
    torso_y = (lm[23][1] + lm[24][1]) / 2
    features.append(np.array([neck_y - torso_y]))

    # Elbow & Knee angle differences
    left_elbow_angle = joint_angle(11, 13, 15)
    right_elbow_angle = joint_angle(12, 14, 16)
    left_knee_angle = joint_angle(23, 25, 27)
    right_knee_angle = joint_angle(24, 26, 28)
    features.append(np.array([
        left_elbow_angle - right_elbow_angle,
        left_knee_angle - right_knee_angle
    ]))
    # === Face vs Torso angle ===
    eye_center = (lm[2] + lm[5]) / 2
    face_vec = lm[0] - eye_center  # nose - eye center
    torso_vec = ((lm[11] + lm[12]) / 2) - ((lm[23] + lm[24]) / 2)

    cos_face_torso = np.dot(face_vec, torso_vec) / (np.linalg.norm(face_vec) * np.linalg.norm(torso_vec) + 1e-6)
    face_torso_angle = np.arccos(np.clip(cos_face_torso, -1.0, 1.0))
    features.append(np.array([face_torso_angle]))

    # === Face forward z-direction (얼굴 방향이 얼마나 전방/하방향인가) ===
    face_z_dir = face_vec[2]  # 양수면 정면, 음수면 아래
    features.append(np.array([face_z_dir]))

    # === Nose below neck (고개 아래로 떨어진 정도) ===
    neck_y = ((lm[11][1] + lm[12][1]) / 2)
    nose_y = lm[0][1]
    head_drop_y = nose_y - neck_y  # 아래로 내려가면 +값
    features.append(np.array([head_drop_y]))

    # === Eye vs Shoulder height ===
    eye_center_y = ((lm[2][1] + lm[5][1]) / 2)
    shoulder_center_y = ((lm[11][1] + lm[12][1]) / 2)
    eye_shoulder_y_diff = eye_center_y - shoulder_center_y
    features.append(np.array([eye_shoulder_y_diff]))

    return np.concatenate(features)
