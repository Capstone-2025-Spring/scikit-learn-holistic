import glob
import os

import numpy as np


def merge_npy_files(input_dir="processed_data", output_dir="dataset", output_x="X_total.npy", output_y="y_total.npy"):
    x_files = sorted(glob.glob(os.path.join(input_dir, "*_X.npy")))
    y_files = sorted(glob.glob(os.path.join(input_dir, "*_y.npy")))

    if not x_files or not y_files:
        print("❌ 병합할 .npy 파일이 없습니다.")
        return

    print(f"📁 X 파일 수: {len(x_files)}")
    print(f"📁 y 파일 수: {len(y_files)}")

    X = np.concatenate([np.load(f) for f in x_files], axis=0)
    y = np.concatenate([np.load(f) for f in y_files], axis=0)

    os.makedirs(output_dir, exist_ok=True)  # ✅ 저장 폴더 생성

    np.save(os.path.join(output_dir, output_x), X)
    np.save(os.path.join(output_dir, output_y), y)

    print("✅ 병합 완료!")
    print(f"🔸 {output_dir}/{output_x} → {X.shape}")
    print(f"🔸 {output_dir}/{output_y} → {y.shape}")

if __name__ == "__main__":
    merge_npy_files()
