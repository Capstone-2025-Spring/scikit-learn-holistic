import os

# train_step1.py 맨 위에 추가
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from classifier.mlp.v1_classifier import Classifier  # MLP 모델

# === 데이터 로딩 === #
X = np.load("dataset/X_total.npy")  # shape: (N, 990)
y = np.load("dataset/y_total.npy")  # shape: (N,)

# === Step 1 라벨 변환: 2(뒤돌기) → 1, 나머지 → 0 === #
y_binary = (y == 2).astype(int)
print("라벨 분포:", np.bincount(y_binary))

# === train/val split === #
X_train, X_val, y_train, y_val = train_test_split(
    X, y_binary, test_size=0.2, stratify=y_binary, random_state=42
)
print("train ratio:", np.mean(y_train), "val ratio:", np.mean(y_val))
# === DataLoader === #
train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                         torch.tensor(y_train, dtype=torch.long))
val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                       torch.tensor(y_val, dtype=torch.long))

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_X = torch.tensor(X_val, dtype=torch.float32)
val_y = torch.tensor(y_val, dtype=torch.long)

# === 모델 === #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Classifier(input_dim=1030, num_classes=2, dropout_rate=0.3).to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

# === 학습 === #
best_acc = 0
best_state = None

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
        val_preds = model(val_X.to(device)).argmax(dim=1).cpu()
        acc = (val_preds == val_y).float().mean().item()
        print(f"[Epoch {epoch}] val_acc = {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_state = model.state_dict()

# === 저장 === #
save_path = "classifier/ensemble/step1"
os.makedirs(save_path, exist_ok=True)
torch.save({
    "input_dim": 1030,
    "num_classes": 2,
    "model_state_dict": best_state
}, f"{save_path}/step1_0.pt")
print("✅ step1_0.pt 저장 완료")
# === 예측 수행 ===
val_preds = model(val_X.to(device)).argmax(dim=1).cpu().numpy()
val_true = val_y.cpu().numpy()

