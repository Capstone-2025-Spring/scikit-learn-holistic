# train_mlp_classifier.py (리팩토링 및 디렉토리 구조 개선)

import importlib.util
import os
import shutil
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# === 모델 정의 파일 경로 설정 === #
model_def_file = "mlp/v1_classifier.py"
model_base_name = "mlp/v1"

# === 디렉토리 생성 === #
os.makedirs("classifier/mlp", exist_ok=True)
os.makedirs("model/mlp", exist_ok=True)
os.makedirs("realtime_pose_predictor/model/mlp", exist_ok=True)
os.makedirs("realtime_pose_predictor/classifier/mlp", exist_ok=True)
os.makedirs("report", exist_ok=True)

# === 모델 정의 파일 로드 === #
def load_classifier(file_path):
    spec = importlib.util.spec_from_file_location("ClassifierModule", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Classifier

Classifier = load_classifier(os.path.join("classifier", model_def_file))

# === 모델 디렉토리 초기화 === #
def clear_directory(dir_path):
    if os.path.exists(dir_path):
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        os.makedirs(dir_path, exist_ok=True)

clear_directory("model/mlp")

# === 데이터 로딩 및 전처리 === #
X = np.load("dataset/X_total.npy")
y = np.load("dataset/y_total.npy")

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)

# === 텐서 변환 === #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Classifier(input_dim=X.shape[1], num_classes=len(np.unique(y)), dropout_rate=0.3).to(device)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)
y_valid_tensor = torch.tensor(y_valid, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# === 옵티마이저, 손실함수, 스케줄러 === #
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)

# === 학습 === #
early_stopping_patience = 10
no_improve_epochs = 0
best_valid_f1 = 0
best_state_dict = None

for epoch in range(100):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        valid_preds = model(X_valid_tensor.to(device))
        valid_preds_label = valid_preds.argmax(dim=1).cpu().numpy()
        valid_f1 = f1_score(y_valid, valid_preds_label, average="macro")

    scheduler.step(valid_f1)

    if valid_f1 > best_valid_f1:
        best_valid_f1 = valid_f1
        best_state_dict = model.state_dict()
        no_improve_epochs = 0
    else:
        no_improve_epochs += 1

    if no_improve_epochs >= early_stopping_patience:
        print(f"\nEarly stopping triggered at epoch {epoch}")
        break

# === 모델 저장 === #
model_save_dict = {
    'input_dim': X.shape[1],
    'num_classes': len(np.unique(y)),
    'model_state_dict': best_state_dict
}

torch.save(model_save_dict, f"model/mlp/v1.pt")
torch.save(model_save_dict, f"realtime_pose_predictor/model/mlp/v1.pt")
shutil.copy("classifier/mlp/v1_classifier.py", "realtime_pose_predictor/classifier/mlp/v1_classifier.py")

# === 테스트 평가 === #
checkpoint = torch.load("model/mlp/v1.pt")
model = Classifier(input_dim=checkpoint['input_dim'], num_classes=checkpoint['num_classes'], dropout_rate=0.3).to(device)
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()
with torch.no_grad():
    y_test_pred = model(X_test_tensor.to(device)).argmax(dim=1).cpu().numpy()

report_text = classification_report(y_test, y_test_pred)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

with open(f"report/{timestamp}_mlp_report.txt", "w") as f:
    f.write(f"Model: {model_base_name}_classifier\n")
    f.write(f"Best Valid F1 Macro: {best_valid_f1:.4f}\n")
    f.write(f"Test F1 Macro: {f1_score(y_test, y_test_pred, average='macro'):.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report_text)

print(f"\n✅ 학습 및 저장 완료! (model/mlp/v1.pt)")
print(f"✅ 학습 및 저장 완료! (realtime_pose_predictor/model/mlp/v1.pt)")
print(f"✅ 모델 구조 복사 완료! (realtime_pose_predictor/classifier/mlp/v1_classifier.py)")
print(f"✅ 성능 리포트 저장 완료! (report/{timestamp}_mlp_report.txt)")
