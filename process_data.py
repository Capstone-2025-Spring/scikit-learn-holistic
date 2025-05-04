import glob
import os

from util.merge_npy_dataset import merge_npy_files
from util.merge_tensor_npy_files import merge_tensor_npy_files
from util.process_pose_json import process_pose_json
from util.process_tensor_json import process_tensor_json

# JSON 파일 목록과 각 파일의 라벨 정의
json_files = [
    ("holistic_json/holistic_label02_behind.json", 2),
    ("holistic_json/holistic_back_02.json",2),
    ("holistic_json/holistic_crossarm_03.json",3),
    ("holistic_json/holistic_label00_standing_1.json",0),
    ("holistic_json/holistic_label00_standing.json",0),
    ("holistic_json/holistic_label03_crossarm_1.json",3),
    ("holistic_json/holistic_label03_crossarm_2.json",3),
    ("holistic_json/holistic_stand_01.json",0),
    ("holistic_json/handsonHead.json",1),
]

def run_all_processing():
    for path, label in json_files:
        if os.path.exists(path):
            print(f"📥 Processing {path} with label {label}")
            process_pose_json(path, label)
            process_tensor_json(path, label)        # ✅ 추가: (N, 10, 83)
        else:
            print(f"❌ 파일 없음: {path}")

    # 모든 npy 생성 후 병합
    print("\n🔀 병합 시작...")
    merge_npy_files()
    merge_tensor_npy_files()
    print("✅ 전체 파이프라인 완료!")

if __name__ == "__main__":
    run_all_processing()
