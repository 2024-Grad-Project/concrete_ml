import time

# import numpy as np
# import torch
# import torch.utils
# #from concrete.compiler import check_gpu_available
# from sklearn.datasets import load_digits
# from sklearn.model_selection import train_test_split

# from tqdm import tqdm



# And some helpers for visualization.

#:wq
# %matplotlib inline

# import matplotlib.pyplot as plt

from PIL import Image
"""
img1 = Image.open('./images/image1.jpg')
img2 = Image.open('./images/image2.jpg')
img3 = Image.open('./images/image3.jpg')
img4 = Image.open('./images/image4.jpg')
img5 = Image.open('./images/image5.jpg')
img6 = Image.open('./images/image6.jpg')
img7 = Image.open('./images/image7.jpg')
img8 = Image.open('./images/image8.jpg')
img9 = Image.open('./images/image9.jpg')
img10 = Image.open('./images/image5.jpg')
img11 = Image.open('./images/image7.jpg')


# 이미지 크기 조정 (예: 28x28 크기로 변환)
img1 = img1.resize((64, 64))
img2 = img2.resize((64, 64))
img3 = img3.resize((64, 64))
img4 = img4.resize((64, 64))
img5 = img5.resize((64, 64))
img6 = img6.resize((64, 64))
img7 = img7.resize((64, 64))
img8 = img8.resize((64, 64))
img9 = img9.resize((64, 64))
img10 = img10.resize((64, 64))
img11 = img11.resize((64, 64))



# 이미지 데이터를 numpy 배열로 변환 (0~255 값)
img_array1 = np.array(img1)
img_array2 = np.array(img2)
img_array3 = np.array(img3)
img_array4 = np.array(img4)
img_array5 = np.array(img5)
img_array6 = np.array(img6)
img_array7 = np.array(img7)
img_array8 = np.array(img8)
img_array9 = np.array(img9)
img_array10 = np.array(img10)
img_array11 = np.array(img11)

# 0-255 범위를 0-1 범위로 정규화
img_array1 = img_array1 / 255.0
img_array2 = img_array2 / 255.0
img_array3 = img_array3 / 255.0
img_array4 = img_array4 / 255.0
img_array5 = img_array5 / 255.0
img_array6 = img_array6 / 255.0
img_array7 = img_array7 / 255.0
img_array8 = img_array8 / 255.0
img_array9 = img_array9 / 255.0

def combine_vectors(*vectors):
    """"""
    return np.vstack(vectors)

# 이미지를 1차원 벡터로 펼치기 (28x28 이미지라면 784개의 값으로 변환)
img_flat1 = img_array1.flatten().astype(np.float32)
img_flat2 = img_array2.flatten().astype(np.float32)
img_flat3 = img_array3.flatten().astype(np.float32)
img_flat4 = img_array4.flatten().astype(np.float32)
img_flat5 = img_array5.flatten().astype(np.float32)
img_flat6 = img_array6.flatten().astype(np.float32)
img_flat7 = img_array7.flatten().astype(np.float32)
img_flat8 = img_array8.flatten().astype(np.float32)
img_flat9 = img_array9.flatten().astype(np.float32)

result_img2 = combine_vectors(img_flat1, img_flat2, img_flat3, img_flat4, img_flat5, img_flat6, img_flat7, img_flat8, img_flat9)
print(result_img2)

"""


from PIL import Image
import numpy as np

# 이미지 파일 이름 리스트 생성
image_files = [f'image{i}.jpg' for i in range(1, 29)]  # image1.jpg부터 image11.jpg까지

# 이미지 로드 및 처리
images = []
img_arrays = []
img_flats = []

for file_name in image_files:
    # 이미지 로드
    img = Image.open(f'./images/{file_name}')
    images.append(img)
    
    # 이미지 크기 조정 (현재 주석 처리되어 있음)
    # img = img.resize((64, 64))
    
    # 이미지를 numpy 배열로 변환 및 정규화
    img_array = np.array(img) / 255.0
    img_arrays.append(img_array)
    
    # 이미지를 1차원 벡터로 평탄화
    img_flat = img_array.flatten().astype(np.float32)
    img_flats.append(img_flat)

# 처리된 이미지들을 전역 변수로 할당 (원래 코드와 일치시키기 위해)
for i, (img, img_array, img_flat) in enumerate(zip(images, img_arrays, img_flats), 1):
    globals()[f'img{i}'] = img
    globals()[f'img_array{i}'] = img_array
    globals()[f'img_flat{i}'] = img_flat

# 첫 9개의 이미지만 결합 (원래 코드와 일치)
result_img = np.vstack(img_flats[:29])

print(result_img)









print("img_flat1")
#print(img_flat1)
#print(img_flat2)
#result_img = combine_vectors(img_flat1, img_flat2, img_flat3, img_flat4, img_flat5, img_flat6, img_flat7, img_flat8, img_flat9)
ydd=np.array([0, 1, 0,1, 1,0,1,1,1,1,0,0,1,0,1,0,0,1,0,0,0,0,1,1,0,0,1,1])
print(result_img)
print("ydd")
print(ydd)
X, y = load_digits(return_X_y=True)
print("X")
print(X)
print("y")
print(y)
# The sklearn Digits data-set, though it contains digit images, keeps these images in vectors
# so we need to reshape them to 2D first. The images are 8x8 px in size and monochrome
X = np.expand_dims(X.reshape((-1, 8, 8)), 1)

nplot = 4
fig, ax = plt.subplots(nplot, nplot, figsize=(6, 6))
for i in range(0, nplot):
    for j in range(0, nplot):
        ax[i, j].imshow(X[i * nplot + j, ::].squeeze())
plt.show()

x_train, x_test, y_train, y_test = train_test_split(
    result_img, ydd, test_size=0.1, shuffle=True, random_state=42
)#X, y
x_train2 = x_train.reshape(-1, 3, 128, 128)
x_test = x_test.reshape(-1, 3, 128, 128)




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

    def forward(self, indices, y, z):
        # Step 1: Scale inputs
        scaled_indices = indices * 0.1
        scaled_y = y * 0.2
        scaled_z = z * 0.3

        # Step 2: Combine inputs
        combined = scaled_indices + scaled_y + scaled_z

        # Step 3: Apply non-linear transformation
        transformed = torch.exp(torch.abs(torch.log(torch.pow(combined, 2) + 1e-5)))

        # Step 4: Apply modulo for additional complexity
        modulo_result = mod(transformed, 1.0)  # Modulo 연산으로 0~1 범위로 제한

        # Step 5: Apply final non-linear transformation
        final_hash = torch.tanh(modulo_result)  # 추가 비선형성
        return final_hash

# 난수 생성기 시드 설정
torch.manual_seed(42)

# 모델 인스턴스 생성
net = ImprovedCNN(2)

output = net(5.0 ,10.5, 5.0)

print(output)