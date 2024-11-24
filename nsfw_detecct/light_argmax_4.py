import torch
from torch import nn
from torchvision import models
from torchvision import transforms
from PIL import Image
from torch.autograd import Variable
import matplotlib.pyplot as plt
import os
from torch.fx import symbolic_trace
import time
start = time.time()
# 이미지 디렉토리
data_dir = 'images/'

# 이미지 전처리 변환
test_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])







import torch.fx

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

        # softmax 대신 sigmoid를 사용
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 10),
            nn.Sigmoid()  # softmax 대신 sigmoid 사용
        )

        # 클래스 수에 맞게 index_weights를 미리 생성하여 모델에 등록 (여기서는 10으로 고정)
        self.register_buffer("index_weights", torch.arange(10).view(1, -1).float())

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
        
        # sigmoid를 통과한 값 얻기
        activations = self.fc(x)
        
        # 최대 활성화 값을 가진 인덱스를 근사 계산
        indices = self.get_approximate_argmax(activations, scale_factor=10)
        
        return indices  # INT64 -> FLOAT 변환 추가

    def get_approximate_argmax(self, activations, scale_factor=100):
        # 큰 값을 더 강조하기 위해 활성화 값을 scale_factor 배로 확장하고 지수화
        exponents = torch.exp(activations * scale_factor)

        # 전체 index_weights를 사용하여 weighted_sum 계산
        weighted_sum = (exponents * self.index_weights).sum(dim=1)

        # 지수의 합으로 정규화하여 인덱스를 근사
        normalization_factor = exponents.sum(dim=1)
        approximate_indices = (weighted_sum / normalization_factor).float()  # INT64 -> FLOAT 변환

        return approximate_indices.round()

# 모델 인스턴스 생성
model = CustomResNet50()

# 사전 학습된 가중치 로드
state_dict = torch.load('ResNet50_nsfw_model.pth', map_location=torch.device('cpu'))
model.load_state_dict(state_dict, strict=False)
model.eval()

# Symbolic Trace로 모델 추적
traced = symbolic_trace(model)

# Mean 연산을 행렬 곱과 스케일링으로 대체
transformed_model = MeanToMatMulTransform(traced).transform()


# Concrete-ML로 모델 컴파일


testimage = Image.open('./images/2.jpg')
image_tensor3 = test_transforms(testimage).float()
image_tensor3 = image_tensor3.unsqueeze_(0)

iamage_input = Variable(image_tensor3)


from concrete.fhe import Configuration
config = Configuration(
    enable_unsafe_features=True,
    use_insecure_key_cache=True,
    insecure_key_cache_location="~/.cml_keycache"
)

from concrete.ml.torch.compile import compile_torch_model
print("Now, we start compile")
quantized_module = compile_torch_model(
    transformed_model,  # 변환된 모델 사용
    iamage_input ,  # 입력 텐서
    import_qat=False,
    configuration = config,
    artifacts = None,
    show_mlir=True,
    n_bits = 8,  # 양자화 비트 수
    rounding_threshold_bits= {"n_bits": 8, "method": "approximate"},
    p_error=0.05,  # 오류 허용 값을 비활성화
    global_p_error = None,
    verbose= False,
    inputs_encryption_status = None,
    reduce_sum_copy= False,
    device = "cpu"
)



from concrete.ml.deployment import FHEModelDev, FHEModelClient, FHEModelServer
print("here is after compile")
#fhe_directory = '/home/giuk/fhe_client_server_files_nsfw_1/' # 자기 자신에 맞게 파일명 바꾸기
#dev = FHEModelDev(path_dir=fhe_directory, model=quantized_module)
#dev.save() #여기가 이제 deploy 생성코드
# 컴파일된 모델로 추론
output_fhe = quantized_module.forward(iamage_input.numpy())


print(output_fhe)
fhe_end = time.time()
fhe_time = fhe_end - start
print(f"FHE execution completed in {fhe_time:.4f} seconds")