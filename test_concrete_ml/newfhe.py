from concrete import fhe
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SimpleFHEModel(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.fc1 = nn.Linear(128 * 128 * 3, 512)  # Flatten된 입력을 받기 위한 Fully Connected Layer
        self.fc2 = nn.Linear(512, n_classes)

    def forward(self, x):
        # 이미 Flatten된 입력으로 가정
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.argmax(dim=1).float()  # argmax 후 float 변환

@fhe.module()
class FHEModule:
    @fhe.function({"x": "encrypted"})
    def model_inference(x):
        model = SimpleFHEModel(2)
        return model.forward(x)

    @fhe.function({"y": "encrypted"})
    def multiply_random(y):
        random_scalar = np.random.uniform(1, 10)  # 1~10 범위의 랜덤 상수
        return y * random_scalar

# 평탄화된 입력 셋 정의 및 컴파일
inputset = [np.random.randint(0, 256, (128 * 128 * 3)).astype(np.float32) for _ in range(100)]
FHEModuleCompiled = FHEModule.compile({"model_inference": inputset, "multiply_random": inputset})

# 키셋 생성
FHEModuleCompiled.keygen()

# 모델 인퍼런스 암호화 실행 및 랜덤 상수 곱하기
x_enc = FHEModuleCompiled.model_inference.encrypt(np.random.randint(0, 256, (128 * 128 * 3)).astype(np.float32))
result_enc = FHEModuleCompiled.model_inference.run(x_enc)
final_result_enc = FHEModuleCompiled.multiply_random.run(result_enc)

# 결과 복호화 및 출력
final_result = FHEModuleCompiled.multiply_random.decrypt(final_result_enc)
print("Encrypted inference result multiplied by random scalar:", final_result)
