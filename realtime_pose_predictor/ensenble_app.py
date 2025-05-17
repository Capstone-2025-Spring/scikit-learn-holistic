import os
import sys
from collections import deque

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import joblib
import mediapipe as mp
import numpy as np
import torch
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget
from utils.feature_extractor import extract_features_from_landmarks

from classifier.cnn.v1_cnn import CNNClassifier as CNN
from classifier.mlp.v1_classifier import Classifier as MLP

# === device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 모델 로더 ===
def load_mlp(path):
    state = torch.load(path, map_location=device)
    model = MLP(input_dim=state["input_dim"], num_classes=state["num_classes"])
    model.load_state_dict(state["model_state_dict"])
    model.to(device).eval()
    return model

def load_cnn(path):
    state = torch.load(path, map_location=device)
    model = CNN(input_channels=state["input_channels"], num_classes=state["num_classes"])
    model.load_state_dict(state["model_state_dict"])
    model.to(device).eval()
    return model

def soft_vote(models, x_tensor):
    logits = [model(x_tensor).detach().cpu().numpy() for model in models]
    return np.mean(logits, axis=0)

# === 모델 로딩 ===
step1 = load_mlp("classifier/ensemble/step1/step1_0.pt")
step2_mlps = [load_mlp(f"classifier/ensemble/step2/step2_{i}.pt") for i in range(3)]
step2_xgb = joblib.load("classifier/ensemble/step2/step2_3.pkl")
step2a_mlps = [load_mlp(f"classifier/ensemble/step2A/step2A_{i}.pt") for i in range(3)]
step2a_cnn = load_cnn("classifier/ensemble/step2A/step2A_3.pt")
step2a_xgb = joblib.load("classifier/ensemble/step2A/step2A_4.pkl")

step2b_mlps = [load_mlp(f"classifier/ensemble/step2B/step2B_{i}.pt") for i in range(2)]
step2b_xgb = joblib.load("classifier/ensemble/step2B/step2B_2.pkl")

# === 예측 함수 ===
def predict_tree(frame_buffer: list) -> int:
    features_seq = np.stack([
        extract_features_from_landmarks(f) for f in frame_buffer
    ])  # shape: (10, 103)

    x_flat = features_seq.flatten().reshape(1, -1)  # (1, 1030)
    x_cnn = features_seq.reshape(1, 1, 10, 103)     # (1, 1, 10, 103)

    x_tensor = torch.tensor(x_flat, dtype=torch.float32).to(device)
    x_seq_tensor = torch.tensor(x_cnn, dtype=torch.float32).to(device)

    # Step 1: 뒤돌기 판별
    pred_step1 = step1(x_tensor).argmax(dim=1).item()
    if pred_step1 == 1:
        return 2

    # Step 2: 상위 분기
    mlp_logits = soft_vote(step2_mlps, x_tensor)

    xgb_logits = step2_xgb.predict_proba(x_flat)
    step2_logits = np.mean([mlp_logits, xgb_logits], axis=0)
    pred_step2 = np.argmax(step2_logits)

    if pred_step2 == 1:
        # Step 2A: 서 vs 고개숙
        mlp_logits = soft_vote(step2a_mlps, x_tensor)
        cnn_logits = step2a_cnn(x_seq_tensor).detach().cpu().numpy()
        xgb_logits = step2a_xgb.predict_proba(x_flat)
        step2a_logits = np.mean([mlp_logits, cnn_logits, xgb_logits], axis=0)
        return 0 if np.argmax(step2a_logits) == 1 else 4
    else:
        # Step 2B: 손 vs 팔짱
        mlp_logits = soft_vote(step2b_mlps, x_tensor)
        xgb_logits = step2b_xgb.predict_proba(x_flat)
        step2b_logits = np.mean([mlp_logits, xgb_logits], axis=0)
        return 1 if np.argmax(step2b_logits) == 1 else 3


# === MediaPipe ===
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

label_map = {
    0: "서있음",
    1: "손 머리에 대기",
    2: "뒤돌기",
    3: "팔짱끼기",
    4: "시선이 아래를 향함",
}


# === PyQt5 App ===
class EnsemblePoseApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ensemble 실시간 Pose 예측기")
        self.resize(800, 600)

        self.video_label = QLabel()
        self.label_output = QLabel("데이터 수집 중...")
        self.label_output.setStyleSheet("font-size: 24px; font-weight: bold; color: #0055ff")

        self.start_button = QPushButton("녹화 시작")
        self.start_button.clicked.connect(self.toggle_capture)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.label_output)
        layout.addWidget(self.start_button)
        self.setLayout(layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.cap = None
        self.frame_buffer = deque(maxlen=10)
        self.is_capturing = False

    def toggle_capture(self):
        if not self.is_capturing:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("❌ 카메라 열기 실패")
                return
            self.start_button.setText("중단")
            self.is_capturing = True
            self.timer.start(30)
        else:
            self.stop_capture()

    def stop_capture(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.start_button.setText("녹화 시작")
        self.label_output.setText("데이터 수집 중...")
        self.frame_buffer.clear()
        self.is_capturing = False

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            pose_data = [
                {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
                for lm in results.pose_landmarks.landmark
            ]
            self.frame_buffer.append(pose_data)

            if len(self.frame_buffer) == 10:
                pred = predict_tree(self.frame_buffer)
                label = label_map.get(pred, str(pred))
                self.label_output.setText(f"예측: {label}")

        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(img))

    def closeEvent(self, event):
        self.stop_capture()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = EnsemblePoseApp()
    window.show()
    sys.exit(app.exec_())
