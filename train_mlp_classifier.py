# train_mlp_classifier.py

import os
import shutil
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


# === Utility: Clear directory === #
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

# === Clear model and report directory === #
clear_directory("model")
os.makedirs("report", exist_ok=True)

# === Load dataset === #
X = np.load("dataset/X_total.npy")
y = np.load("dataset/y_total.npy")

# === Split into train/valid/test === #
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)
# train: 60%, valid: 20%, test: 20%

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

# === Prepare for training === #
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

# === Train === #
for epoch in range(50):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        valid_preds = model(X_valid_tensor)
        valid_preds_label = valid_preds.argmax(dim=1).cpu().numpy()
        valid_f1 = f1_score(y_valid, valid_preds_label, average="macro")
        if valid_f1 > best_valid_f1:
            best_valid_f1 = valid_f1
            best_state_dict = model.state_dict()

# === Save best model === #
torch.save({
    'input_dim': X.shape[1],
    'num_classes': len(np.unique(y)),
    'model_state_dict': best_state_dict
}, "model/mlp_model.pt")

# === Test evaluation === #
checkpoint = torch.load("model/mlp_model.pt")
model = MLPClassifier(input_dim=checkpoint['input_dim'], num_classes=checkpoint['num_classes']).to(device)
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()
with torch.no_grad():
    y_test_pred = model(X_test_tensor).argmax(dim=1).cpu().numpy()

report_text = classification_report(y_test, y_test_pred)
report_dict = classification_report(y_test, y_test_pred, output_dict=True)

# === Save report with timestamp === #
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

with open(f"report/{timestamp}_mlp_report.txt", "w") as f:
    f.write("Model: MLPClassifier\n")
    f.write(f"Best Valid F1 Macro: {best_valid_f1:.4f}\n")
    f.write(f"Test F1 Macro: {f1_score(y_test, y_test_pred, average='macro'):.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report_text)

print("\n✅ MLP 학습 및 저장 완료! (model/mlp_model.pt)")
print(f"✅ 성능 리포트 저장 완료! (report/{timestamp}_mlp_report.txt)")
print("\n✅ MLP 학습 및 저장 완료! (model/mlp_model.pt)")
print(f"✅ 성능 리포트 저장 완료! (report/{timestamp}_mlp_report.txt)")
