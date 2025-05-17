import importlib.util
import os

import numpy as np
import torch

# === 모델 정의 파일 경로 === #
model_def_path = "classifier/mlp/v1_classifier.py"
model_dir = "model/ensemble"
model_paths = [os.path.join(model_dir, f"v1_model_{i}.pt") for i in range(3)]

# === Classifier 불러오기 === #
def load_classifier(file_path):
    spec = importlib.util.spec_from_file_location("ClassifierModule", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Classifier

Classifier = load_classifier(model_def_path)

# === 모델 로딩 === #
models = []
for path in model_paths:
    checkpoint = torch.load(path, map_location=torch.device("cpu"))
    model = Classifier(
        input_dim=checkpoint["input_dim"],
        num_classes=checkpoint["num_classes"],
        dropout_rate=0.3
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    models.append(model)

print("✅ 앙상블 모델 3개 로딩 완료")

# === 예측 함수 (Soft Voting) === #
def ensemble_predict(input_vector: np.ndarray) -> int:
    """
    input_vector: shape (1, input_dim)
    return: int (예측 클래스)
    """
    input_tensor = torch.tensor(input_vector, dtype=torch.float32)

    with torch.no_grad():
        logits_sum = None
        for model in models:
            logits = model(input_tensor)
            if logits_sum is None:
                logits_sum = logits
            else:
                logits_sum += logits

        avg_logits = logits_sum / len(models)
        pred_class = avg_logits.argmax(dim=1).item()
        return pred_class

# === 예시 사용 === #
if __name__ == "__main__":
    test_input = np.random.rand(1, 830).astype(np.float32)  # 예시 입력
    pred = ensemble_predict(test_input)
    print(f"🔮 최종 예측 클래스: {pred}")
