import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

from classifier.cnn.cnn_classifier import CNNClassifier


# ====== 1. Custom Dataset ======
class PoseDataset(Dataset):
    def __init__(self, x_path, y_path):
        self.x = np.load(x_path)  # shape: (N, 10, 83)
        self.y = np.load(y_path)  # shape: (N,)

        self.x = self.x[:, np.newaxis, :, :]  # reshape to (N, 1, 10, 83)
        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# ====== 2. Evaluation Function ======
def evaluate(model, valid_loader, criterion, device):
    model.eval()
    val_loss, correct = 0, 0

    with torch.no_grad():
        for x_batch, y_batch in valid_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            val_loss += criterion(outputs, y_batch).item()
            correct += (outputs.argmax(dim=1) == y_batch).sum().item()

    val_acc = correct / len(valid_loader.dataset)
    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}\n")
    return val_loss, val_acc


# ====== 3. Training Function ======
def train(model, train_loader, valid_loader, criterion, optimizer, device, epochs=20, patience=3):
    model.to(device)
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    best_acc = 0
    best_val_acc = 0

    for epoch in range(epochs):
        model.train()
        total_loss, correct = 0, 0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == y_batch).sum().item()

        acc = correct / len(train_loader.dataset)
        print(f"[Epoch {epoch+1}] Train Loss: {total_loss:.4f}, Accuracy: {acc:.4f}")

        val_loss, val_acc = evaluate(model, valid_loader, criterion, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            best_acc = acc
            best_val_acc = val_acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"⏹ Early stopping triggered after {epoch+1} epochs!")
                break

    return best_model_state, best_acc, best_val_acc


# ====== 4. Main ======
if __name__ == "__main__":
    # Settings
    x_path = "dataset/X_tensor.npy"
    y_path = "dataset/y_tensor.npy"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    epochs = 3
    lr = 1e-4

    # Dataset
    dataset = PoseDataset(x_path, y_path)
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

    # Model, Loss, Optimizer
    model = CNNClassifier(input_channels=1, num_classes=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train
    best_model_state, best_acc, best_val_acc = train(model, train_loader, valid_loader, criterion, optimizer, device, epochs)

    # Save Model
    os.makedirs("pkl", exist_ok=True)
    torch.save(model.state_dict(), "cnn_classifier.pth")
    print("✅ Model saved to pkl/cnn_classifier.pth")

    # Save Report
    os.makedirs("report/cnn", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_path = f"report/cnn/{timestamp}_cnn_report.txt"
    with open(report_path, "w") as f:
        f.write(f"CNN Classifier Training Report\n")
        f.write(f"Train Accuracy (best): {best_acc:.4f}\n")
        f.write(f"Validation Accuracy (best): {best_val_acc:.4f}\n")
    print(f"✅ 성능 리포트 저장 완료! ({report_path})")
