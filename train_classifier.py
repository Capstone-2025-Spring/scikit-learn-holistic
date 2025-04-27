import os
import sys

import joblib
import numpy as np
import pandas as pd

# ==== PyTorch 관련 추가 ====
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

# === Load dataset === #
X = np.load("dataset/X_total.npy")
y = np.load("dataset/y_total.npy")

# === Split into train/valid/test === #
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)
# train: 60%, valid: 20%, test: 20%

# === Define models === #
models = {
    "svm": SVC(kernel="rbf", probability=True),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "adaboost": AdaBoostClassifier(n_estimators=100, random_state=42),
    "xgboost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    "mlp": None  # 🔥 PyTorch MLP는 따로 처리할 예정
}

# === Define MLP === #
class MLPClassifier(nn.Module):
    def __init__(self, input_dim=90, num_classes=4):
        super(MLPClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# === Create output folders === #
os.makedirs("model", exist_ok=True)
os.makedirs("pkl", exist_ok=True)
os.makedirs("report", exist_ok=True)

# === Cross-validation and evaluation === #
results = {}

for name, model in models.items():
    print(f"\n🔍 Training model: {name.upper()}")

    if name != "mlp":
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_cv_train, X_cv_val = X_train[train_idx], X_train[val_idx]
            y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]
            model.fit(X_cv_train, y_cv_train)
            y_pred = model.predict(X_cv_val)
            score = f1_score(y_cv_val, y_pred, average="macro")
            cv_scores.append(score)

        model.fit(X_train, y_train)  # retrain on all training data
        joblib.dump(model, f"model/{name}_model.pkl")
        joblib.dump(model, f"pkl/{name}_model.pkl")

        y_test_pred = model.predict(X_test)
        report_text = classification_report(y_test, y_test_pred)
        report_dict = classification_report(y_test, y_test_pred, output_dict=True)

        results[name] = {
            "cv_f1_macro_mean": np.mean(cv_scores),
            "test_f1_macro": f1_score(y_test, y_test_pred, average="macro"),
            "test_report": report_dict
        }

        with open(f"report/{name}_report.txt", "w") as f:
            f.write(f"Model: {name}\n")
            f.write(f"CV F1 Macro: {np.mean(cv_scores):.4f}\n")
            f.write(f"Test F1 Macro: {results[name]['test_f1_macro']:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report_text)

    else:
        # 🔥 PyTorch MLP 학습
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MLPClassifier(input_dim=X.shape[1], num_classes=len(np.unique(y))).to(device)

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
        X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32).to(device)
        y_valid_tensor = torch.tensor(y_valid, dtype=torch.long).to(device)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        best_valid_f1 = 0
        best_state_dict = None

        for epoch in range(50):  # 50 epoch
            model.train()
            for xb, yb in train_loader:
                optimizer.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()

            # Validation check
            model.eval()
            with torch.no_grad():
                valid_preds = model(X_valid_tensor)
                valid_preds_label = valid_preds.argmax(dim=1).cpu().numpy()
                valid_f1 = f1_score(y_valid, valid_preds_label, average="macro")
                if valid_f1 > best_valid_f1:
                    best_valid_f1 = valid_f1
                    best_state_dict = model.state_dict()

        # ✅ Best 모델 저장: 구조 정보까지 같이 저장
        torch.save({
            'input_dim': X.shape[1],
            'num_classes': len(np.unique(y)),
            'model_state_dict': best_state_dict
        }, "model/mlp_model.pt")

        # === 모델 불러서 Test 세트 평가 ===
        checkpoint = torch.load("model/mlp_model.pt")
        model = MLPClassifier(input_dim=checkpoint['input_dim'], num_classes=checkpoint['num_classes']).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])

        model.eval()
        with torch.no_grad():
            y_test_pred = model(X_test_tensor).argmax(dim=1).cpu().numpy()

        report_text = classification_report(y_test, y_test_pred)
        report_dict = classification_report(y_test, y_test_pred, output_dict=True)

        results[name] = {
            "cv_f1_macro_mean": best_valid_f1,
            "test_f1_macro": f1_score(y_test, y_test_pred, average="macro"),
            "test_report": report_dict
        }

        with open(f"report/{name}_report.txt", "w") as f:
            f.write(f"Model: {name}\n")
            f.write(f"Best Valid F1 Macro: {best_valid_f1:.4f}\n")
            f.write(f"Test F1 Macro: {results[name]['test_f1_macro']:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report_text)

# === Save summary === #
summary = pd.DataFrame({
    model: {
        "CV F1 Macro": f"{res['cv_f1_macro_mean']:.4f}",
        "Test F1 Macro": f"{res['test_f1_macro']:.4f}"
    } for model, res in results.items()
}).T

summary.to_csv("report/performance_summary.csv")
print("\n📊 [완료] 성능 요약이 report/performance_summary.csv 에 저장되었습니다.")
