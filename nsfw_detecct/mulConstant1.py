from concrete import fhe
import torch
from torch import nn
from torchvision import models

# Custom ResNet50 모델 정의
class CustomResNet50(nn.Module):
    def __init__(self):
        super(CustomResNet50, self).__init__()
        original_model = models.resnet50()
        self.conv1 = original_model.conv1
        self.bn1 = original_model.bn1
        self.relu = original_model.relu
        self.maxpool = original_model.maxpool
        self.layer1 = original_model.layer1
        self.layer2 = original_model.layer2
        self.layer3 = original_model.layer3
        self.layer4 = original_model.layer4
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 5)  # 클래스 수에 맞게 조정
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# FHE 모듈로 변환하기 위한 Concrete FHE 래퍼 클래스 정의
@fhe.module()
class FHECustomResNet50:
    def __init__(self, model):
        self.model = model

    @fhe.function({"x": "encrypted"})
    def predict(self, x):
        return self.model(x)

# 모델 인스턴스 생성
original_model = CustomResNet50()
# 사전 학습된 가중치 로드
state_dict = torch.load('ResNet50_nsfw_model.pth', map_location=torch.device('cpu'))
original_model.load_state_dict(state_dict, strict=False)
original_model.eval()

# FHE 모델 인스턴스 생성
fhe_model = FHECustomResNet50(original_model)
