import torch
import time
from torch import nn
from torch.autograd import Variable
from concrete.ml.deployment import FHEModelDev, FHEModelClient, FHEModelServer
#여기서부터 deploy _ client & server
fhe_directory = '/home/giuk/fhe_client_server_files_nsfw_2/' # 자기 자신에 맞게 파일명 바꾸기
# Setup the client
client = FHEModelClient(path_dir=fhe_directory, key_dir="/home/giuk/keys_client_nsfw_17_Oct/")
serialized_evaluation_keys = client.get_serialized_evaluation_keys()

from PIL import Image
from torchvision import models
from torchvision import transforms


def image_to_tensor(image_path):
    # 이미지 로드
    image = Image.open(image_path).convert("RGB")
    
    # 이미지 전처리 (크기 조정 및 텐서로 변환)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),  # [0, 1] 범위의 Tensor로 변환 (C, H, W)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 일반적인 이미지넷 평균 및 표준편차
    ])
    
    tensor = transform(image)  # (C, H, W)
    tensor = tensor.unsqueeze(0)  # (1, 3, 224, 224)로 변환 (배치 크기 추가)
    
    return tensor

image_path = './images/5.jpg'
sample_input = image_to_tensor(image_path)


np_array = sample_input.numpy()

test_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

image_path_for_check = './images/2.jpg'
testimage = Image.open(image_path_for_check)
image_tensor3 = test_transforms(testimage).float()
image_tensor3 = image_tensor3.unsqueeze_(0)

input3 = Variable(image_tensor3)

print("here is before encryption")

encrypted_data = client.quantize_encrypt_serialize(input3.numpy())

print("here is after encryption")
# Setup the server
server = FHEModelServer(path_dir=fhe_directory)
server.load()
print("here is after server.load")
# Server processes the encrypted data
total_fhe_time = 0
start = time.time()
print("Processing the image.(FHE Computation)")
print(image_path_for_check)
encrypted_result = server.run(encrypted_data, serialized_evaluation_keys)
print("here is after server.run(FHE Computation)")
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

