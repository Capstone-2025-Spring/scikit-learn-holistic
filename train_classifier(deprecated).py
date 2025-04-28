# train_classifier.py

import os
import shutil
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import SVC
from xgboost import XGBClassifier


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

# === Clear model directory === #
clear_directory("model")

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
    "xgboost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
}

# === Create output folders === #
os.makedirs("model", exist_ok=True)
os.makedirs("pkl", exist_ok=True)
os.makedirs("report", exist_ok=True)

# === Train and evaluate === #
results = {}

for name, model in models.items():
    print(f"\n🔍 Training model: {name.upper()}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_cv_train, X_cv_val = X_train[train_idx], X_train[val_idx]
        y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]
        model.fit(X_cv_train, y_cv_train)
        y_pred = model.predict(X_cv_val)
        score = f1_score(y_cv_val, y_pred, average="macro")
        cv_scores.append(score)

    model.fit(X_train, y_train)
    joblib.dump(model, f"model/{name}_model.pkl")

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

# === Save performance summary with timestamp === #
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
summary = pd.DataFrame({
    model: {
        "CV F1 Macro": f"{res['cv_f1_macro_mean']:.4f}",
        "Test F1 Macro": f"{res['test_f1_macro']:.4f}"
    } for model, res in results.items()
}).T

summary.to_csv(f"report/{timestamp}_performance_summary.csv")
print(f"\n📊 [완료] 성능 요약이 report/{timestamp}_performance_summary.csv 에 저장되었습니다.")
summary.to_csv(f"report/{timestamp}_performance_summary.csv")
print(f"\n📊 [완료] 성능 요약이 report/{timestamp}_performance_summary.csv 에 저장되었습니다.")
