from concrete import fhe
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
from torch.fx import symbolic_trace
import os

# 이미지 전처리 정의
test_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

import torch.nn.functional as F


# CustomResNet50 정의
class CustomResNet50(nn.Module):
    def __init__(self):
        super(CustomResNet50, self).__init__()
        original_model = models.resnet50()
        self.conv1_weight = original_model.conv1.weight
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
        self.register_buffer("index_weights", torch.arange(10).view(1, -1).float())

    def forward(self, x):
        x = F.conv2d(x, self.conv1_weight, bias=None, stride=(2, 2), padding=(3, 3))
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        N, C, H, W = x.shape
        x = x.view(N, C, H * W)
        x = x.sum(dim=-1) / (H * W)
        
        activations = self.fc(x)
        indices = self.get_approximate_argmax(activations, scale_factor=10)
        
        return indices

    def get_approximate_argmax(self, activations, scale_factor=100):
        exponents = torch.exp(activations * scale_factor)
        weighted_sum = (exponents * self.index_weights).sum(dim=1)
        normalization_factor = exponents.sum(dim=1)
        approximate_indices = (weighted_sum / normalization_factor).float()
        return approximate_indices.round()


import torch.fx
from concrete.ml.quantization import QuantizedArray
# Mean 연산을 행렬 곱과 스케일링으로 대체하는 변환 클래스 정의
class MeanToMatMulTransform(torch.fx.Transformer):
    def call_method(self, target: str, args, kwargs):
        # mean 연산을 발견하면 대체
        if target == 'mean':
            print(f"Replacing 'mean' with scaling at node {target}")
            
            # 입력 텐서를 펼친 후, H * W로 나눠 스케일링
            x = args[0]
            H, W = x.shape[2], x.shape[3]
            num_elements = H * W

            # 평균을 내는 대신 행렬 곱과 스케일링 적용
            x = x.flatten(2)  # (N, C, H*W)
            x = x.sum(dim=-1)  # H * W 차원을 sum으로 합침
            x = x / num_elements  # H * W 요소로 나눠 평균 근사
            return x  # 변환된 결과 반환
        return super().call_method(target, args, kwargs)
    
# 모델 인스턴스 생성 및 state_dict 로드
model = CustomResNet50()
state_dict = torch.load('ResNet50_nsfw_model.pth', map_location=torch.device('cpu'))
model.load_state_dict(state_dict, strict=False)
model.eval()

# Symbolic Trace 및 Mean 연산 변환
traced = symbolic_trace(model)
transformed_model = MeanToMatMulTransform(traced).transform()

# Counter 클래스 정의
@fhe.module()
class Counter:
    @fhe.function({"image": "encrypted"})
    def inc(image):
        indices = transformed_model(image)  # transformed_model 사용
        return indices

    @fhe.function({"x": "encrypted", "y": "clear"})
    def dec(x, y):
        return (x * y) % 20

# 이미지 로드 및 전처리
def load_image_as_tensor(image_path):
    image = Image.open(image_path).convert("RGB")
    return test_transforms(image).unsqueeze(0)

# FHE 환경에서 Counter 실행
image_path = './images/2.jpg'
image_tensor = load_image_as_tensor(image_path)
n_bits = 8  # 원하는 양자화 비트 수
quantized_image = QuantizedArray(n_bits, image_tensor.numpy())
quantized_image_values = quantized_image.qvalues
CounterFhe = Counter.compile({"inc": [quantized_image_values], "dec": [(4, 3)]})
x_enc = CounterFhe.inc.encrypt(quantized_image_values)
index = CounterFhe.inc.run(x_enc)
result = CounterFhe.dec.run(index, 10)

print("NSFW Index:", index)
print("Result after multiplying with constant:", result)
