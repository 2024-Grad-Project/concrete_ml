from concrete import fhe
import torch
from torch import nn
from torchvision import models
import numpy as np

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
            nn.Dropout(0.2),
            nn.Linear(512, 10),
            nn.Sigmoid()
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
        
        # Flatten and apply fully connected layers
        x = torch.flatten(x, 1)
        activations = self.fc(x)
        
        # Return the approximate argmax
        return self.get_approximate_argmax(activations)

    def get_approximate_argmax(self, activations, scale_factor=100):
        exponents = torch.exp(activations * scale_factor)
        index_weights = torch.arange(10).view(1, -1).float()
        weighted_sum = (exponents * index_weights).sum(dim=1)
        normalization_factor = exponents.sum(dim=1)
        return (weighted_sum / normalization_factor).round()

# FHE 모듈 정의
@fhe.module()
class FHECustomResNet50:
    @fhe.function({"x": "encrypted"})
    def model_inference(x):
        model = CustomResNet50()
        return model.forward(x)

    @fhe.function({"y": "encrypted"})
    def multiply_with_random(y):
        random_scalar = np.random.uniform(1, 10)
        return y * random_scalar

# 입력셋 정의 및 컴파일
inputset = [np.random.randint(0, 256, (3, 224, 224)).astype(np.float32) for _ in range(20)]
FHEModel = FHECustomResNet50.compile({"model_inference": inputset, "multiply_with_random": inputset})

# 키셋 생성
FHEModel.keygen()

# 암호화된 상태에서 예측 및 랜덤 상수 곱하기
x = np.random.randint(0, 256, (3, 224, 224)).astype(np.float32)
x_enc = FHEModel.model_inference.encrypt(x)
pred_enc = FHEModel.model_inference.run(x_enc)
final_result_enc = FHEModel.multiply_with_random.run(pred_enc)

# 결과 복호화 및 출력
final_result = FHEModel.multiply_with_random.decrypt(final_result_enc)
print("Encrypted inference result multiplied by random scalar:", final_result)
