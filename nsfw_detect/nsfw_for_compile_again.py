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

# Custom ResNet50 모델 정의
class CustomResNet50(nn.Module):
    def __init__(self):
        super(CustomResNet50, self).__init__()
        # Pretrained ResNet50 모델 로드
        original_model = models.resnet50()
        self.conv1 = original_model.conv1
        self.bn1 = original_model.bn1
        self.relu = original_model.relu
        self.maxpool = original_model.maxpool
        self.layer1 = original_model.layer1
        self.layer2 = original_model.layer2
        self.layer3 = original_model.layer3
        self.layer4 = original_model.layer4
        # Fully connected layer 수정 (클래스 수에 따라 출력 차원 조정)
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 10)  # 출력 차원을 5로 수정
        )
class CustomResNet50(nn.Module):
    def __init__(self):
        super(CustomResNet50, self).__init__()
        # Pretrained ResNet50 모델 로드
        original_model = models.resnet50()
        self.conv1 = original_model.conv1
        self.bn1 = original_model.bn1
        self.relu = original_model.relu
        self.maxpool = original_model.maxpool
        self.layer1 = original_model.layer1
        self.layer2 = original_model.layer2
        self.layer3 = original_model.layer3
        self.layer4 = original_model.layer4
        # Fully connected layer 수정
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 10)  # 출력 차원
        )
        # 각 클래스에 대한 one-hot 벡터를 미리 생성
        self.class_weights = nn.Parameter(torch.eye(10), requires_grad=False)

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

        x = self.fc(x)
        # 각 클래스와의 내적을 계산하여 가중치 부여
        # 가장 높은 값을 가진 클래스의 인덱스에 가장 큰 가중치가 부여됨
        weights = torch.matmul(x, self.class_weights.T)  # (batch_size, n_classes)
        # 가중치가 가장 큰 클래스의 인덱스를 근사
        # 각 클래스에 대한 가중치를 정규화하여 반환
        normalized_weights = weights / weights.sum(dim=1, keepdim=True)
        # 각 클래스의 인덱스와 가중치를 곱하여 합산
        predicted_index = (normalized_weights * torch.arange(10).float()).sum(dim=1)
        return predicted_index.round()  # 반올림하여 정수 인덱스 반환

# Custom 변환 클래스 정의
class MeanToMatMulTransform(torch.fx.Transformer):
    def call_method(self, target: str, args, kwargs):
        if target == 'mean':
            print(f"Replacing 'mean' with matrix multiplication and scaling at node {target}")
            
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

# 예측 함수 정의
def predict_image(image):
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)

    input = Variable(image_tensor)
    output = transformed_model(input)
    print("output")
    print(output)
    index = output.data.numpy().argmax()
    print("index")
    print(index)
    return index

# 모델 클래스 정의
classes = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']

# 이미지 로드 및 예측
entries = os.listdir(data_dir)

fig = plt.figure(figsize=(10, 10))
i = 0
print(entries)
for entry in entries:
    i += 1
    print("entry")
    print(entry)
    image = Image.open(data_dir + entry)

    # 예측
    index = predict_image(image)

    sub = fig.add_subplot(1, len(entries), i)
    sub.set_title(classes[index])
    plt.axis('off')
    plt.imshow(image)
plt.show()

# Concrete-ML로 모델 컴파일
from concrete.ml.torch.compile import compile_torch_model

testimage2 = Image.open('./images/2.jpg')
image_tensor4 = test_transforms(testimage2).float()
image_tensor4 = image_tensor4.unsqueeze_(0)

input4 = Variable(image_tensor4)


testimage = Image.open('./images/1.jpg')
image_tensor3 = test_transforms(testimage).float()
image_tensor3 = image_tensor3.unsqueeze_(0)

input3 = Variable(image_tensor3)


# 이미지 전처리 변환 정의 (ResNet과 같은 이미지 전처리)
test_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# JPEG 이미지를 불러와서 텐서로 변환하는 함수
def load_image_as_tensor(image_path):
    image = Image.open(image_path).convert("RGB")  # 이미지 불러오기
    image_tensor = test_transforms(image).float()  # 전처리 및 텐서 변환
    image_tensor = image_tensor.unsqueeze_(0)  # 배치 차원 추가 (N, C, H, W)
    return image_tensor

# 이미지 경로 설정
image_path = './images/20.jpg'

# 이미지를 torch_input으로 대체
torch_input = load_image_as_tensor(image_path)


from concrete.ml.quantization import QuantizedArray
n_bits = 6
quantized_input3 = QuantizedArray(n_bits, input3.numpy())
quantized_input3_long = torch.tensor(quantized_input3.qvalues, dtype=torch.long)
# 모델 컴파일




from concrete.ml.quantization import QuantizedArray
import numpy as np

# 이미지 로드 및 전처리
image_path = './images/20.jpg'
torch_input = load_image_as_tensor(image_path)

# torch 텐서를 numpy 배열로 변환
numpy_input = torch_input.numpy()

# 입력 양자화
n_bits = 7  # compile_torch_model에서 사용한 n_bits와 일치해야 함
quantized_input = QuantizedArray(n_bits, numpy_input)

#################################################################################


import matplotlib.pyplot as plt

from PIL import Image

img1 = Image.open('./images/1.jpg')
img2 = Image.open('./images/2.jpg')
img3 = Image.open('./images/3.jpg')
img4 = Image.open('./images/4.jpg')
img5 = Image.open('./images/5.jpg')


# 이미지 크기 조정 (예: 28x28 크기로 변환)
img1 = img1.resize((64, 64))
img2 = img2.resize((64, 64))
img3 = img3.resize((64, 64))
img4 = img4.resize((64, 64))
img5 = img5.resize((64, 64))




# 이미지 데이터를 numpy 배열로 변환 (0~255 값)
img_array1 = np.array(img1)
img_array2 = np.array(img2)
img_array3 = np.array(img3)
img_array4 = np.array(img4)
img_array5 = np.array(img5)


# 0-255 범위를 0-1 범위로 정규화
img_array1 = img_array1 / 255.0
img_array2 = img_array2 / 255.0
img_array3 = img_array3 / 255.0
img_array4 = img_array4 / 255.0
img_array5 = img_array5 / 255.0


def combine_vectors(*vectors):
    """여러 개의 벡터를 2차원 배열로 결합합니다."""
    return np.vstack(vectors)

# 이미지를 1차원 벡터로 펼치기 (28x28 이미지라면 784개의 값으로 변환)
img_flat1 = img_array1.flatten().astype(np.float32)
img_flat2 = img_array2.flatten().astype(np.float32)
img_flat3 = img_array3.flatten().astype(np.float32)
img_flat4 = img_array4.flatten().astype(np.float32)
img_flat5 = img_array5.flatten().astype(np.float32)

print("img_flat1")
print(img_flat1)
print(img_flat2)
result_img = combine_vectors(img_flat1, img_flat2, img_flat3, img_flat4, img_flat5)
ydd=np.array([0, 1, 0, 0, 1])
print(result_img)
print("ydd")
print(ydd)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    result_img, ydd, test_size=0.25, shuffle=True, random_state=42
)#X, y
x_train2 = x_train.reshape(-1, 3, 64, 64)
x_test = x_test.reshape(-1, 3, 64, 64)


##################################################################################
# 양자화된 모델이 예상하는 형식으로 변환
quantized_input_values = np.array(quantized_input.qvalues, dtype=np.int64)
from concrete.fhe import Configuration
config = Configuration(
    enable_unsafe_features=True,
    use_insecure_key_cache=True,
    insecure_key_cache_location="~/.cml_keycache"
)


quantized_module = compile_torch_model(
    transformed_model,  # 변환된 모델 사용
    input4 ,  # 입력 텐서
    import_qat=False,
    configuration = config,
    artifacts = None,
    show_mlir=True,
    n_bits = 7,  # 양자화 비트 수
    rounding_threshold_bits= {"n_bits": 7, "method": "approximate"},
    p_error=0.05,  # 오류 허용 값을 비활성화
    global_p_error = None,
    verbose= False,
    inputs_encryption_status = None,
    reduce_sum_copy= False,
    device = "cpu"
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
#fhe_directory = '/home/giuk/fhe_client_server_files_nsfw_1/' # 자기 자신에 맞게 파일명 바꾸기
#dev = FHEModelDev(path_dir=fhe_directory, model=quantized_module)
#dev.save() #여기가 이제 deploy 생성코드
# 컴파일된 모델로 추론
output_fhe = quantized_module.forward(input3.numpy())
print(output_fhe)
fhe_end = time.time()
fhe_time = fhe_end - start
print(f"FHE execution completed in {fhe_time:.4f} seconds")