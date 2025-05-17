# utils/mlp_classifier.py (강화버전)

import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, input_dim=103, num_classes=4, dropout_rate=0.3):
        """
        강화된 MLP Classifier 모델
        - input_dim: 입력 feature 차원
        - num_classes: 출력 클래스 수
        - dropout_rate: dropout 비율
        """
        super(Classifier, self).__init__()

        # Sequential 네트워크 정의
        self.net = nn.Sequential(
            # 첫 번째 Linear + BatchNorm + ReLU + Dropout
            nn.Linear(input_dim, 512),          # 입력 -> 512 hidden units
            nn.BatchNorm1d(512),                 # 배치 정규화로 학습 안정화
            nn.ReLU(),                           # 비선형 활성화 함수
            nn.Dropout(dropout_rate),             # 과적합 방지를 위한 드롭아웃

            # 두 번째 Linear + BatchNorm + ReLU + Dropout
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # 세 번째 Linear + BatchNorm + ReLU + Dropout
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # 네 번째 Linear + BatchNorm + ReLU + Dropout
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # 출력층 (num_classes로 매핑)
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """
        순전파 정의
        :param x: 입력 텐서(batch_size, input_dim)
        :return: 출력 로짓(batch_size, num_classes)
        """
        return self.net(x)

