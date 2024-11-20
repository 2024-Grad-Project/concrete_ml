import time
from concrete.ml.deployment import FHEModelDev, FHEModelClient, FHEModelServer
from torchvision import transforms
import numpy as np
import torch
import torch.utils
#from concrete.compiler import check_gpu_available
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from concrete.ml.torch.compile import compile_torch_model

# And some helpers for visualization.

#:wq
# %matplotlib inline

import matplotlib.pyplot as plt

from PIL import Image



from PIL import Image
import numpy as np


class TinyCNN(nn.Module):
    """A very small CNN to classify the sklearn digits data-set."""

    def __init__(self, n_classes) -> None:
        """Construct the CNN with a configurable number of classes."""
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, stride=2, padding=1)  # 64x64
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)  # 32x32
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)  # 16x16
        self.conv4 = nn.Conv2d(64, 128, 3, stride=2, padding=1)  # 8x8

        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, n_classes)

    def forward(self, x):
        """Run inference on the tiny CNN."""
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    

    torch.manual_seed(42)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch

class ImprovedCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, n_classes)
        self.dropout = nn.Dropout(0.5)
        self.register_buffer("index_weights", torch.arange(n_classes).view(1, -1).float())  # index_weights 등록

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        activations = self.fc2(x)
        print("here is in forward")
        # activations를 사용하여 근사 index 반환
        return self.get_approximate_argmax(activations)

    def get_approximate_argmax(self, activations, scale_factor=10):
        exponents = torch.exp(activations * scale_factor)
        weighted_sum = (exponents * self.index_weights).sum(dim=1)
        normalization_factor = exponents.sum(dim=1)
        approximate_indices = (weighted_sum / normalization_factor)
        dd = approximate_indices.round()+1
        return dd



# 모델 인스턴스 생성
net = ImprovedCNN(2)


def image_to_tensor(image_path):
    # 이미지 로드
    image = Image.open(image_path).convert("RGB")
    
    # 이미지 전처리 (크기 조정 및 텐서로 변환)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),  # [0, 1] 범위의 Tensor로 변환 (C, H, W)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 일반적인 이미지넷 평균 및 표준편차
    ])
    
    tensor = transform(image)  # (C, H, W)
    tensor = tensor.unsqueeze(0)  # (1, 3, 224, 224)로 변환 (배치 크기 추가)
    
    return tensor

image_path = './images/image5.jpg'
sample_input = image_to_tensor(image_path)


n_bits = 7

use_gpu_if_available = False
devicew = "cpu"# "cuda" if use_gpu_if_available and check_gpu_available() else "cpu"
from concrete.fhe import Configuration

config = Configuration(
    enable_unsafe_features=True,
    use_insecure_key_cache=True,
    insecure_key_cache_location="~/.cml_keycache"
)
{"x": "encrypted", "y": "encrypted"}
q_module = compile_torch_model(
    torch_model=net, 
    torch_inputset=sample_input, 
    import_qat=False,
    configuration = config,
    artifacts = None,
    show_mlir=True,
    n_bits=7,  # n_bits 값을 줄여보세요.
    rounding_threshold_bits={"n_bits": 6, "method":"APPROXIMATE"},  # 기본 값을 사용해 볼 수 있습니다.
    p_error=0.05,  # 오류 허용 값을 비활성화
    global_p_error = None,
    verbose=False,
    inputs_encryption_status = None,
    reduce_sum_copy=False,
    device = "cpu"
)



def image_to_tensor(image_path):
    # 이미지 로드
    image = Image.open(image_path).convert("RGB")
    
    # 이미지 전처리 (크기 조정 및 텐서로 변환)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),  # [0, 1] 범위의 Tensor로 변환 (C, H, W)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 일반적인 이미지넷 평균 및 표준편차
    ])
    
    tensor = transform(image)  # (C, H, W)
    tensor = tensor.unsqueeze(0)  # (1, 3, 224, 224)로 변환 (배치 크기 추가)
    
    return tensor

image_path = './images/image5.jpg'
sample_input = image_to_tensor(image_path)


np_array = sample_input.numpy()
print("here is before forward")
output = q_module.forward(np_array)
print(output)
print("here is after forward")
#print(concrete.ml.__version__)
#fhe_directory = '/home/giuk/zama_fhe_directory/test_7/' # 자기 자신에 맞게 파일명 바꾸기
#dev = FHEModelDev(path_dir=fhe_directory, model=q_module)
#dev.save() #여기가 이제 deploy 생성코드


#fhe_directory = '/home/giuk/fhe_client_server_files_128/'
#dev = FHEModelDev(path_dir=fhe_directory, model=q_module)
#dev.save()
""""""
"""def compile_torch_model(
    torch_model: torch.nn.Module,
    torch_inputset: Dataset,
    import_qat: bool = False,
    configuration: Optional[Configuration] = None,
    artifacts: Optional[DebugArtifacts] = None,
    show_mlir: bool = False,
    n_bits: Union[int, Dict[str, int]] = MAX_BITWIDTH_BACKWARD_COMPATIBLE,
    rounding_threshold_bits: Union[None, int, Dict[str, Union[str, int]]] = None,
    p_error: Optional[float] = None,
    global_p_error: Optional[float] = None,
    verbose: bool = False,
    inputs_encryption_status: Optional[Sequence[str]] = None,
    reduce_sum_copy: bool = False,
) -> QuantizedModule:"""









def image_to_tensor(image_path):
    # 이미지 로드
    image = Image.open(image_path).convert("RGB")
    
    # 이미지 전처리 (크기 조정 및 텐서로 변환)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),  # [0, 1] 범위의 Tensor로 변환 (C, H, W)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 일반적인 이미지넷 평균 및 표준편차
    ])
    
    tensor = transform(image)  # (C, H, W)
    tensor = tensor.unsqueeze(0)  # (1, 3, 224, 224)로 변환 (배치 크기 추가)
    
    return tensor

image_path = './images/image5.jpg'
sample_input = image_to_tensor(image_path)


np_array = sample_input.numpy()



'''
# Setup the client
client = FHEModelClient(path_dir=fhe_directory, key_dir="/home/giuk/keys_client/")
serialized_evaluation_keys = client.get_serialized_evaluation_keys()



print("here is before encryption")

encrypted_data = client.quantize_encrypt_serialize(np_array)

print("here is after encryption")
# Setup the server
server = FHEModelServer(path_dir=fhe_directory)
server.load()
print("here is after server.load")
# Server processes the encrypted data
total_fhe_time = 0
start = time.time()
encrypted_result = server.run(encrypted_data, serialized_evaluation_keys)
print("here is after server.run")
fhe_end = time.time()


fhe_time = fhe_end - start
total_fhe_time += fhe_time
print(f"  FHE execution completed in {fhe_time:.4f} seconds")
# Client decrypts the result


dec_start = time.time()
result = client.deserialize_decrypt_dequantize(encrypted_result)
dec_end = time.time()
dec_time = dec_end - dec_start
print(f"  DEC execution completed in {dec_time:.4f} seconds")
print(result)

print("here is after result")
'''

