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




fhe_directory = '/home/giuk/fhe_client_server_files_128/'

# Setup the client
client = FHEModelClient(path_dir=fhe_directory, key_dir="/home/giuk/keys_client/")
serialized_evaluation_keys = client.get_serialized_evaluation_keys()

# Client pre-processes new data
X_new = np.random.rand(1, 20)



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


