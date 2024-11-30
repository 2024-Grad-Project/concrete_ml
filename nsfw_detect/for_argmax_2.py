import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
from torch.autograd import Variable
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from torch.fx import symbolic_trace

start = time.time()
data_dir = 'images/'

# 이미지 전처리 변환
test_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class CustomResNet50(nn.Module):
    def __init__(self):
        super(CustomResNet50, self).__init__()
        original_model = models.resnet50(pretrained=True) 
        self.conv1 = original_model.conv1
        self.bn1 = original_model.bn1
        self.relu = original_model.relu
        self.maxpool = original_model.maxpool
        self.layer1 = original_model.layer1
        self.layer2 = original_model.layer2
        self.layer3 = original_model.layer3
        self.layer4 = original_model.layer4

        # Fully Connected Layer without Softmax
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 10)  # 10개 클래스
        )

        # 10개 클래스에 대한 인덱스 가중치
        self.index_weights = nn.Parameter(torch.arange(10).float().reshape(1, -1))
        
    def forward(self, x):
        x = self.conv1(x)
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
        
        logits = self.fc(x)
        indices = (logits * self.index_weights).sum(dim=1)  # Softmax 없이 인덱스 계산

        return indices

# 모델 인스턴스 생성 및 사전 학습된 가중치 로드
model = CustomResNet50()
state_dict = torch.load('ResNet50_nsfw_model.pth', map_location=torch.device('cpu'))
model.load_state_dict(state_dict, strict=False)
model.eval()

# Symbolic Trace로 모델 추적
traced = symbolic_trace(model)

# Custom 변환 클래스 정의
class MeanToMatMulTransform(torch.fx.Transformer):
    def call_method(self, target: str, args, kwargs):
        if target == 'mean':
            x = args[0]
            H, W = x.shape[2], x.shape[3]
            num_elements = H * W
            x = x.flatten(2).sum(dim=-1) / num_elements
            return x
        return super().call_method(target, args, kwargs)

# Mean 연산을 행렬 곱과 스케일링으로 대체
transformed_model = MeanToMatMulTransform(traced).transform()

# 예측 함수 정의
def predict_image(image):
    image_tensor = test_transforms(image).float().unsqueeze_(0)
    input = Variable(image_tensor)
    output = transformed_model(input)
    index = output.data.numpy().argmax()
    return index

# Concrete-ML로 모델 컴파일 코드
from concrete.ml.torch.compile import compile_torch_model
from concrete.fhe import Configuration

config = Configuration(
    enable_unsafe_features=True,
    use_insecure_key_cache=True,
    insecure_key_cache_location="~/.cml_keycache"
)

quantized_module = compile_torch_model(
    transformed_model,
    input4,
    import_qat=False,
    configuration=config,
    artifacts=None,
    show_mlir=True,
    n_bits=7,
    rounding_threshold_bits={"n_bits": 7, "method": "approximate"},
    p_error=0.05,
    global_p_error=None,
    verbose=False,
    inputs_encryption_status=None,
    reduce_sum_copy=False,
    device="cpu"
)

"""
q_module = compile_torch_model(
    torch_model=net, 
    torch_inputset=x_train2, 
    import_qat=False,
    configuration = config,
    artifacts = None,
    show_mlir=True,
    n_bits=6,  # n_bits 값을 줄여보세요.
    rounding_threshold_bits={"n_bits": 6, "method":"APPROXIMATE"},  # 기본 값을 사용해 볼 수 있습니다.
    p_error=0.05,  # 오류 허용 값을 비활성화
    global_p_error = None,
    verbose=False,
    inputs_encryption_status = None,
    reduce_sum_copy=False,
    device = "cpu"


"""
from concrete.ml.deployment import FHEModelDev, FHEModelClient, FHEModelServer
print("here is after compile")


fhe_directory = '/home/giuk/fhe_client_server_files_nsfw_1/' # 자기 자신에 맞게 파일명 바꾸기
dev = FHEModelDev(path_dir=fhe_directory, model=quantized_module)
dev.save() #여기가 이제 deploy 생성코드
# 컴파일된 모델로 추론
output_fhe = quantized_module.forward(iamage_input.numpy())


print(output_fhe)
fhe_end = time.time()
fhe_time = fhe_end - start
print(f"FHE execution completed in {fhe_time:.4f} seconds")