import glob
import os

import numpy as np


def merge_tensor_npy_files(input_dir="processed_tensor", output_dir="dataset", output_x="X_tensor.npy", output_y="y_tensor.npy"):
    x_files = sorted(glob.glob(os.path.join(input_dir, "*_X_tensor.npy")))
    y_files = sorted(glob.glob(os.path.join(input_dir, "*_y.npy")))

    if not x_files or not y_files:
        print("❌ 병합할 텐서 .npy 파일이 없습니다.")
        return

    print(f"📁 Tensor X 파일 수: {len(x_files)}")
    print(f"📁 Tensor y 파일 수: {len(y_files)}")

    X = np.concatenate([np.load(f) for f in x_files], axis=0)  # (N, 10, 83)
    y = np.concatenate([np.load(f) for f in y_files], axis=0)  # (N,)

    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, output_x), X)
    np.save(os.path.join(output_dir, output_y), y)

    print("✅ 텐서 병합 완료!")
    print(f"🔸 {output_dir}/{output_x} → {X.shape}")
    print(f"🔸 {output_dir}/{output_y} → {y.shape}")


if __name__ == "__main__":
    merge_tensor_npy_files()
