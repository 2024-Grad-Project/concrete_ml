import torch
import time
from torch import nn
from torch.autograd import Variable
from concrete.ml.deployment import FHEModelDev, FHEModelClient, FHEModelServer
#여기서부터 deploy _ client & server
fhe_directory = '/home/giuk/zama_fhe_directory/test_3/' # 자기 자신에 맞게 파일명 바꾸기
# Setup the client
client = FHEModelClient(path_dir=fhe_directory, key_dir="/home/giuk/keys_client_cnn_13_Nov_999/")
serialized_evaluation_keys = client.get_serialized_evaluation_keys()

from PIL import Image
from torchvision import models
from torchvision import transforms



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
# 상수 값을 원본 텐서의 크기와 일치시키기
constant_value = 10  # 원하는 상수 값
constant_tensor = torch.full_like(sample_input, constant_value, dtype=torch.int16)  # sample_input과 동일한 크기와 타입으로 상수 텐서 생성

# 상수 값을 numpy 배열로 변환 후 암호화
encrypted_constant = client.quantize_encrypt_serialize(constant_tensor.numpy())
print("Constant encrypted")
print("here is after encryption")
# Setup the server
server = FHEModelServer(path_dir=fhe_directory)
server.load()
print("here is after server.load")
# Server processes the encrypted data

encrypted_result = server.run(encrypted_data, encrypted_constant,serialized_evaluation_keys)
print("here is after server.run")

# Client decrypts the result
print("This is the decryption result without multiplying the ciphertext by a constant.")
result = client.deserialize_decrypt_dequantize(encrypted_result)
print(result)
num = 12345
byte_length = 4  # 바이트 길이 설정
byte_data = num.to_bytes(byte_length, byteorder="big")
print("This is the decryption result after multiplying the ciphertext by a constant.")
encrypted_result_with_token = encrypted_result + byte_data
#print("encyrpted+1",encrypted_result_with_token)
print(result)
result_with_token = client.deserialize_decrypt_dequantize(encrypted_result_with_token)
print("decrypted+1")
print(result_with_token)

#






################################################################
#from concrete import fhe

#def add(x, y):
#    return x * y

#compiler = fhe.Compiler(add, {"x": "encrypted", "y":"clear"})

#inputset = [(1, 2), (1, 4), (1, 6), (1, 9), (1, 3), (1, 4), (1, 5), (1, 7), (1, 11), (1, 3)]

#print(f"Compilation...")
#circuit = compiler.compile(inputset)


################################################################





#encrypted_result333 = circuit.run(encrypted_result, 250)


