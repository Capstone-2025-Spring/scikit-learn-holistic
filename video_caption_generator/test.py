import json
import os

from run_captioning import run_captioning_from_json

# 예시 Holistic JSON 데이터 (timestamp + landmarks 33개)
# 실제로는 33개 전체를 넣어야 하나, 여기선 더미 3개만 사용
# test_data.json은 {"holisticData": [...]} 구조를 가져야 함
with open("holistic.json", "r") as f:
    holistic_json = json.load(f)

        
def save_caption_to_file(captions, output_path="output/captions.txt"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for line in captions:
            f.write(line + "\n")

    print(f"✅ 캡션 저장 완료: {output_path}")
    
        
if __name__ == "__main__":
    captions = run_captioning_from_json(holistic_json)
    print("=== 캡션 결과 ===")
    save_caption_to_file(captions)