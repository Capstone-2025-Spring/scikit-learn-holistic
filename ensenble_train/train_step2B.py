import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

from classifier.mlp.v1_classifier import Classifier as MLP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 데이터 로딩 ===
X = np.load("dataset/X_total.npy")
y = np.load("dataset/y_total.npy")

# Step 2B: 손(1) vs 팔짱(3)
mask = (y == 1) | (y == 3)
X = X[mask]
y = (y[mask] == 1).astype(int)  # 손 = 1, 팔짱 = 0

# === MLP 2개 ===
for i in range(2):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=3000 + i
    )

    train_loader = DataLoader(TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    ), batch_size=64, shuffle=True)

    val_X = torch.tensor(X_val, dtype=torch.float32).to(device)
    val_y = torch.tensor(y_val, dtype=torch.long)

    model = MLP(input_dim=1030, num_classes=2, dropout_rate=0.3).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    best_state = None

    for epoch in range(50):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = model(val_X).argmax(dim=1).cpu()
            acc = (preds == val_y.cpu()).float().mean().item()
            if acc > best_acc:
                best_acc = acc
                best_state = model.state_dict()
        print(f"MLP {i} Epoch {epoch} acc = {acc:.4f}")

    os.makedirs("classifier/ensemble/step2B", exist_ok=True)
    torch.save({
        "input_dim": 1030,
        "num_classes": 2,
        "model_state_dict": best_state
    }, f"classifier/ensemble/step2B/step2B_{i}.pt")

# === XGBoost ===
xgb = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05)
xgb.fit(X, y)
joblib.dump(xgb, "classifier/ensemble/step2B/step2B_2.pkl")

print("✅ Step 2B 학습 완료 (MLP 2 + XGB 1)")
