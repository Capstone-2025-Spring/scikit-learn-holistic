import os
import sys

import numpy as np
import torch
from xgboost import XGBClassifier

# === 경로 설정 ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # predict.py 기준
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "../..")))  # video_caption_generator 루트 추가

from classifier.cnn.cnn import CNNClassifier as CNN
from classifier.mlp.mlp import Classifier as MLP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_mlp(rel_path, input_dim=1030, num_classes=2):
    path = os.path.join(BASE_DIR, rel_path)
    state = torch.load(path, map_location=device)
    model = MLP(input_dim=input_dim, num_classes=num_classes)
    model.load_state_dict(state["model_state_dict"])
    model.to(device).eval()
    return model


def load_cnn(rel_path, input_channels=1, num_classes=2):
    path = os.path.join(BASE_DIR, rel_path)
    state = torch.load(path, map_location=device)
    model = CNN(input_channels=input_channels, num_classes=num_classes)
    model.load_state_dict(state["model_state_dict"])
    model.to(device).eval()
    return model


def load_xgb(rel_path):
    path = os.path.join(BASE_DIR, rel_path)
    model = XGBClassifier()
    model.load_model(path)
    return model


# === Load all models === #
step1 = load_mlp("step1/step1_0.pt")

step2_mlps = [load_mlp(f"step2/step2_{i}.pt") for i in range(3)]
step2_xgb = load_xgb("step2/step2_3.json")

step2a_mlps = [load_mlp(f"step2A/step2A_{i}.pt") for i in range(3)]
step2a_cnn = load_cnn("step2A/step2A_3.pt")
step2a_xgb = load_xgb("step2A/step2A_4.json")

step2b_mlps = [load_mlp(f"step2B/step2B_{i}.pt") for i in range(2)]
step2b_xgb = load_xgb("step2B/step2B_2.json")


def soft_vote(models, x_tensor):
    logits = [model(x_tensor).detach().cpu().numpy() for model in models]
    return np.mean(logits, axis=0)


def predict_tree(x_flat: np.ndarray) -> int:
    x_tensor = torch.tensor(x_flat, dtype=torch.float32).to(device)
    x_seq_tensor = x_tensor.view(1, 10, 103)  # NOTE: adjust if needed
    x_seq_cnn = x_seq_tensor.unsqueeze(1)  # (1, 1, 10, 103)

    # Step 1
    pred_step1 = step1(x_tensor).argmax(dim=1).item()
    if pred_step1 == 1:
        return 2  # 뒤돌기

    # Step 2
    mlp_logits = soft_vote(step2_mlps, x_tensor)
    xgb_logits = step2_xgb.predict_proba(x_flat)
    step2_logits = np.mean([mlp_logits, xgb_logits], axis=0)
    pred_step2 = np.argmax(step2_logits)

    if pred_step2 == 1:
        # Step 2A
        mlp_logits = soft_vote(step2a_mlps, x_tensor)
        xgb_logits = step2a_xgb.predict_proba(x_flat)
        step2a_logits = np.mean([mlp_logits, xgb_logits], axis=0)
        return 0 if np.argmax(step2a_logits) == 1 else 4
    else:
        # Step 2B
        mlp_logits = soft_vote(step2b_mlps, x_tensor)
        xgb_logits = step2b_xgb.predict_proba(x_flat)
        step2b_logits = np.mean([mlp_logits, xgb_logits], axis=0)
        return 1 if np.argmax(step2b_logits) == 1 else 3
