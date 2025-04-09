import glob
import os

from merge_npy_dataset import merge_npy_files
from process_pose_json import process_pose_json

# JSON 파일 목록과 각 파일의 라벨 정의
json_files = [
    ("holistic_json/holistic_label00_standing.json", 0),
    ("holistic_json/holistic_label01_reading.json", 1),
    ("holistic_json/holistic_label02_behind.json", 2),
]

def run_all_processing():
    for path, label in json_files:
        if os.path.exists(path):
            print(f"📥 Processing {path} with label {label}")
            process_pose_json(path, label)
        else:
            print(f"❌ 파일 없음: {path}")

    # 모든 npy 생성 후 병합
    print("\n🔀 병합 시작...")
    merge_npy_files()
    print("✅ 전체 파이프라인 완료!")

if __name__ == "__main__":
    run_all_processing()
