# app.py

import os
import sys
from collections import deque

import cv2
import joblib
import mediapipe as mp
import numpy as np
import torch
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

# === 추가: utils 폴더 import 위해 경로 추가 (필요 시) ===
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# === utils 폴더에서 import ===
from utils.feature_extractor import extract_features_from_landmarks
from utils.mlp_classifier import MLPClassifier

# === MediaPipe 초기화 ===
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# === 라벨 매핑 ===
label_map = {
    0: "서있음",
    1: "손 머리에 대기",
    2: "뒤돌기",
    3: "팔짱끼기",
}

class PoseApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("실시간 Pose 데이터 수집기")
        self.resize(800, 600)

        self.video_label = QLabel()
        self.label_output = QLabel("데이터 수집 중...")
        self.label_output.setStyleSheet("font-size: 24px; font-weight: bold; color: #0055ff")

        self.model_selector = QComboBox()
        self.model_selector.addItems(self.load_model_list())

        self.start_button = QPushButton("녹화 시작")
        self.start_button.clicked.connect(self.toggle_capture)

        self.save_button = QPushButton("데이터 저장")
        self.save_button.clicked.connect(self.save_data)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.label_output)
        layout.addWidget(self.model_selector)
        layout.addWidget(self.start_button)
        layout.addWidget(self.save_button)
        self.setLayout(layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.cap = None
        self.frame_buffer = deque(maxlen=10)
        self.collected_features = []
        self.collected_labels = []
        self.is_capturing = False
        self.model = None
        self.model_type = None  # 'sklearn' / 'pytorch' / 'pytorch-deferred'
        self.defer_model_creation = None  # pytorch-deferred 일 때 사용

    def load_model_list(self):
        model_dir = "model"
        if not os.path.exists(model_dir):
            return []
        return [f for f in os.listdir(model_dir) if f.endswith(".pkl") or f.endswith(".pt")]

    def toggle_capture(self):
        if not self.is_capturing:
            model_name = self.model_selector.currentText()
            model_path = os.path.join("model", model_name)

            if model_name.endswith(".pkl"):
                self.model = joblib.load(model_path)
                self.model_type = 'sklearn'
                print(f"✅ Sklearn 모델 {model_name} 로딩 완료")

            elif model_name.endswith(".pt"):
                checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    input_dim = checkpoint.get('input_dim')
                    num_classes = checkpoint.get('num_classes', 4)
                    state_dict = checkpoint['model_state_dict']
                else:
                    input_dim = None
                    num_classes = 4
                    state_dict = checkpoint

                if input_dim is None:
                    print("⚠️ input_dim 정보가 없습니다. feature shape에서 자동 추정합니다.")
                    self.defer_model_creation = {
                        "state_dict": state_dict,
                        "num_classes": num_classes
                    }
                    self.model = None
                    self.model_type = 'pytorch-deferred'
                else:
                    self.model = MLPClassifier(input_dim=input_dim, num_classes=num_classes)
                    self.model.load_state_dict(state_dict)
                    self.model.eval()
                    self.model_type = 'pytorch'
                    print(f"✅ PyTorch 모델 {model_name} 로딩 완료 (input_dim={input_dim}, num_classes={num_classes})")

            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("❌ 카메라를 열 수 없습니다.")
                return

            self.start_button.setText("녹화 중단")
            self.is_capturing = True
            self.timer.start(30)
        else:
            self.stop_capture()

    def stop_capture(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.start_button.setText("녹화 시작")
        self.is_capturing = False
        self.label_output.setText("데이터 수집 중...")
        self.frame_buffer.clear()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            pose_data = [{
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility
            } for lm in results.pose_landmarks.landmark]
            self.frame_buffer.append(pose_data)

            if len(self.frame_buffer) == 10:
                feature = np.concatenate([
                    extract_features_from_landmarks(f) for f in self.frame_buffer
                ])
                feature = feature.reshape(1, -1)
                self.collected_features.append(feature.squeeze())

                if self.model_type == 'pytorch-deferred' and self.model is None:
                    input_dim = feature.shape[1]
                    print(f"🛠️ feature input_dim={input_dim} 기반으로 MLPClassifier 새로 생성")
                    self.model = MLPClassifier(input_dim=input_dim, num_classes=self.defer_model_creation["num_classes"])
                    self.model.load_state_dict(self.defer_model_creation["state_dict"])
                    self.model.eval()
                    self.model_type = 'pytorch'

                if self.model:
                    if self.model_type == 'sklearn':
                        pred_label = self.model.predict(feature)[0]
                    elif self.model_type == 'pytorch':
                        input_tensor = torch.tensor(feature, dtype=torch.float32)
                        with torch.no_grad():
                            pred_label = self.model(input_tensor).argmax().item()

                    self.collected_labels.append(pred_label)
                    label_text = label_map.get(pred_label, str(pred_label))
                    self.label_output.setText(f"현재 예측: {label_text} ({len(self.collected_features)}개 수집)")

        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_img))

    def save_data(self):
        if not self.collected_features:
            print("수집된 데이터가 없습니다.")
            return

        X_save = np.array(self.collected_features)
        y_save = np.array(self.collected_labels)

        save_dir = "stream_saved"
        os.makedirs(save_dir, exist_ok=True)

        np.save(os.path.join(save_dir, "X_stream.npy"), X_save)
        np.save(os.path.join(save_dir, "y_stream.npy"), y_save)

        print(f"✅ {len(X_save)}개 feature와 예측 라벨을 {save_dir}/ 에 저장했습니다.")

    def closeEvent(self, event):
        self.stop_capture()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PoseApp()
    window.show()
    sys.exit(app.exec_())
