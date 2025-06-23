#  Edu-Mate AI 모델 개발 레파지토리

이 저장소는 Edu-Mate 시스템에서 사용되는 **행동 인식 모델**과 **동작 캡션 생성 모델**의 학습 및 추론 코드를 포함합니다.  
MediaPipe로 추출한 포즈 데이터를 기반으로, 수업 중 지양해야 할 행동을 탐지하고 이를 기반으로 실시간 피드백을 제공합니다.

---

##  주요 기능

- Holistic Keypoint JSON 데이터 전처리
- 다양한 ML/DL 기반 동작 분류기 학습 및 앙상블
- 실시간 예측 앱 (`cnn_app.py`, `mlp_app.py`, `ensemble_app.py`)
- t-SNE 기반 시각화 및 리포트 자동 저장
- JSON 기반 캡션 생성기 (강의 행동 자동 요약)

---

##  디렉토리 구조

```bash
EduMate-AI/
├── ensemble_train/              # 앙상블 학습 구성
│   ├── train_step1.py
│   ├── train_step2.py
│   ├── train_step2A.py
│   └── train_step2B.py

├── holistic_json/              # 학습용 포즈 JSON (수동 수집)
├── model/                      # 학습된 모델 가중치 저장
│   ├── cnn/
│   ├── ensemble/
│   └── mlp/

├── report/                     # 학습 리포트 및 평가 지표
├── pkl/                        # scikit-learn 기반 모델들 (.pkl)
├── util/                       # 데이터 처리 및 특징 추출 유틸
│   └── feature_extractor.py ...

├── realtime_pose_predictor/    # 실시간 예측 앱
│   ├── cnn_app.py
│   ├── mlp_app.py
│   ├── ensemble_app.py
│   ├── model/                  # 실시간 추론용 모델 로드
│   ├── classifier/
│   ├── utils/
│   └── tsne.py

├── video_caption_generator/    # 행동 JSON → 캡션 문자열 생성
│   ├── run_captioning.py
│   ├── classifier/             # 내부 분류 모델 구조
│   ├── output/
│   └── utils/


```
## 1. 데이터 전처리
```
python util/process_pose_json.py
```
## 2. 모델 학습 (MLP or CNN or 앙상블)
```
python ensemble_train/train_step1.py      # 1차 판별: 뒤돌기 탐지
python ensemble_train/train_step2.py      # 전체 클래스 분류 팔을 사용하거나 사용하지 않음
python ensemble_train/train_step2A.py     # 소분류 1 : 팔짱을 끼거나 손을 머리위로 올림
python ensemble_train/train_step2A.py     # 소분류 2 : 고개를 숙이거나, 가만히 서있음
```
## 3. 실시간 예측 실행
```
python realtime_pose_predictor/mlp_app.py # 웹캠을 이용한 실시간 예측 PyQt 기반 어플리케이션
```
## 4. 캡션 생성 테스트
```
python video_caption_generator/run_captioning.py # 백엔드에 적용되는 json 데이터 기반 캡션 생성 API
```
## 사용된 분류기들
```
MLPClassifier (PyTorch 기반)
CNNClassifier
SVM, Random Forest, XGBoost, AdaBoost (Scikit-learn 기반)
```
## 앙상블 구조: Soft Voting 기반

