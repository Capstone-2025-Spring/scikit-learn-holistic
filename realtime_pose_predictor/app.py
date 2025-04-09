import os
import sys
from collections import Counter, deque

import cv2
import joblib
import mediapipe as mp
import numpy as np
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
from utils.feature_extractor import extract_features_from_landmarks

# === 라벨 매핑 ===
label_map = {
    0: "서있음",
    1: "교안",
    2: "뒤돌기"
}

# === MediaPipe 초기화 ===
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

class PoseApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("실시간 Pose 예측기")
        self.resize(800, 600)

        self.video_label = QLabel()
        self.label_output = QLabel("예측 라벨: -")
        self.label_output.setStyleSheet("font-size: 36px; font-weight: bold; color: #0055ff")

        self.model_selector = QComboBox()
        self.model_selector.addItems(self.load_model_list())

        self.start_button = QPushButton("예측 시작")
        self.start_button.clicked.connect(self.toggle_prediction)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.label_output)
        layout.addWidget(self.model_selector)
        layout.addWidget(self.start_button)
        self.setLayout(layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.cap = None
        self.frame_buffer = deque(maxlen=10)
        self.pred_history = deque(maxlen=5)
        self.model = None
        self.is_predicting = False

    def load_model_list(self):
        model_dir = "model"
        if not os.path.exists(model_dir):
            return []
        return [f for f in os.listdir(model_dir) if f.endswith(".pkl")]

    def toggle_prediction(self):
        if not self.is_predicting:
            model_name = self.model_selector.currentText()
            model_path = os.path.join("model", model_name)
            self.model = joblib.load(model_path)
            self.cap = cv2.VideoCapture(0)
            self.start_button.setText("예측 중단")
            self.is_predicting = True
            self.timer.start(30)
        else:
            self.stop_prediction()

    def stop_prediction(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.start_button.setText("예측 시작")
        self.is_predicting = False
        self.label_output.setText("예측 라벨: -")
        self.frame_buffer.clear()
        self.pred_history.clear()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            pose_data = []
            for lm in results.pose_landmarks.landmark:
                pose_data.append({
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                    "visibility": lm.visibility
                })
            self.frame_buffer.append(pose_data)

            if len(self.frame_buffer) == 10 and self.model:
                feature = np.concatenate([
                    extract_features_from_landmarks(f) for f in self.frame_buffer
                ])
                feature = feature.reshape(1, -1)
                pred = self.model.predict(feature)[0]
                self.pred_history.append(pred)

                # ✅ Smoothing: 최근 5개 중 가장 많은 라벨 표시
                most_common = Counter(self.pred_history).most_common(1)[0][0]
                self.label_output.setText(f"예측 라벨: {label_map.get(most_common, str(most_common))}")

        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_img))

    def closeEvent(self, event):
        self.stop_prediction()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PoseApp()
    window.show()
    sys.exit(app.exec_())   