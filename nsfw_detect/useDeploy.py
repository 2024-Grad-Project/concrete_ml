import torch
import time
import random
from torch import nn
from torch.autograd import Variable
from concrete.ml.deployment import FHEModelDev, FHEModelClient, FHEModelServer
#여기서부터 deploy _ client & server
#fhe_client_server_files_nsfw_3_gpu = gpu machine with n_bits = 6 and image999
#fhe_client_server_files_nsfw_1 = cpu machine with n_bits = 7 and image2
#fhe_client_server_files_nsfw_2 = cpu machine with n_bits = 6 and image2
fhe_directory = '/home/giuk/fhe_client_server_files_nsfw_1/' # 자기 자신에 맞게 파일명 바꾸기
# Setup the client
client = FHEModelClient(path_dir=fhe_directory, key_dir="/home/giuk/keys_client_nsfw_4_Nov/")
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

image_path = './images/20.jpg'
sample_input = image_to_tensor(image_path)


np_array = sample_input.numpy()

test_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

image_path_for_check = './images/20.jpg'
testimage = Image.open(image_path_for_check)
image_tensor3 = test_transforms(testimage).float()
image_tensor3 = image_tensor3.unsqueeze_(0)

input_image = Variable(image_tensor3)

print("here is before encryption")

randConstant = random.randint(1, 1000)
print(randConstant)


# Client encrypts the result
encrypted_data = client.quantize_encrypt_serialize(input_image.numpy())

# Setup the server
server = FHEModelServer(path_dir=fhe_directory)
server.load()

# Server processes the encrypted data
print(image_path_for_check)
encrypted_result = server.run(encrypted_data, serialized_evaluation_keys)

# Server multiplies the ciphertext by a random constant.
randConstant = random.randint(1, 1000)
encrypted_result_with_constant = encrypted_result * randConstant

print(type(encrypted_result))
print(encrypted_result.fhe.Value)

# Client decrypts the result
result = client.deserialize_decrypt_dequantize(encrypted_result_with_constant)
print(result)


