import numpy as np

# 파일 경로
file_path = "dataset\X_total.npy"  # 파일 경로는 예시입니다

# 데이터 불러오기
X = np.load(file_path)

# 차원 확인
print("Shape:", X.shape)  # 예: (491, 770)
print("Samples:", X.shape[0])
print("Feature dimension:", X.shape[1])
print(X[0][-10:])     
# 예시 샘플 보기
print("First sample:", X[0])        # 첫 번째 샘플 전체
print("First 10 features:", X[0][:10])  # 첫 샘플의 앞 10개 feature
