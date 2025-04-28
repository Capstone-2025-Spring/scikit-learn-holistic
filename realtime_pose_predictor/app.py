# app.py (최종 버전)

import importlib.util
import os
import sys
from collections import deque

import cv2
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

# === Feature Extractor Import === #
from utils.feature_extractor import extract_features_from_landmarks

# === utils 폴더 import 위해 경로 추가 === #
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# === MediaPipe 초기화 === #
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# === 라벨 매핑 === #
label_map = {
    0: "서있음",
    1: "손 머리에 대기",
    2: "뒤돌기",
    3: "팔짱끼기",
}


# === Classifier Loader === #
def dynamic_load_classifier(module_path):
    spec = importlib.util.spec_from_file_location("ClassifierModule", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Classifier  # 무조건 'Classifier'라는 이름의 클래스 로딩

class PoseApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("실시간 Pose 예측기")
        self.resize(800, 600)

        self.video_label = QLabel()
        self.label_output = QLabel("데이터 수집 중...")
        self.label_output.setStyleSheet("font-size: 24px; font-weight: bold; color: #0055ff")

        self.model_selector = QComboBox()
        self.model_selector.addItems(self.load_model_list())

        self.start_button = QPushButton("녹화 시작")
        self.start_button.clicked.connect(self.toggle_capture)

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
        self.model = None
        self.is_capturing = False

        os.makedirs("model", exist_ok=True)  # model 폴더 자동 생성

    def load_model_list(self):
        model_dir = "model"
        if not os.path.exists(model_dir):
            return []
        return sorted([f for f in os.listdir(model_dir) if f.endswith(".pt")])

    def toggle_capture(self):
        if not self.is_capturing:
            model_name = self.model_selector.currentText()
            model_path = os.path.join("model", model_name)
            model_base_name = model_name.replace(".pt", "")
            classifier_path = os.path.join("classifier", f"{model_base_name}_classifier.py")

            if not os.path.exists(classifier_path):
                raise FileNotFoundError(f"❌ 매칭되는 모델 정의 파일을 찾을 수 없습니다: {classifier_path}")

            Classifier = dynamic_load_classifier(classifier_path)

            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            input_dim = checkpoint.get('input_dim')
            num_classes = checkpoint.get('num_classes', 4)
            state_dict = checkpoint['model_state_dict']

            if input_dim is None:
                raise ValueError(f"❌ {model_name} 파일에 input_dim 정보가 없습니다.")

            self.model = Classifier(input_dim=input_dim, num_classes=num_classes)
            self.model.load_state_dict(state_dict)
            self.model.eval()

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
            pose_data = [
                {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
                for lm in results.pose_landmarks.landmark
            ]
            self.frame_buffer.append(pose_data)

            if len(self.frame_buffer) == 10:
                feature = np.concatenate([
                    extract_features_from_landmarks(f) for f in self.frame_buffer
                ])
                feature = feature.reshape(1, -1)

                if self.model:
                    input_tensor = torch.tensor(feature, dtype=torch.float32)
                    with torch.no_grad():
                        pred_label = self.model(input_tensor).argmax().item()

                    label_text = label_map.get(pred_label, str(pred_label))
                    self.label_output.setText(f"현재 예측: {label_text}")

        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_img))

    def closeEvent(self, event):
        self.stop_capture()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PoseApp()
    window.show()
    sys.exit(app.exec_())
    sys.exit(app.exec_())
    sys.exit(app.exec_())
