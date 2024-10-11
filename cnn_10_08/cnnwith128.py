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

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 난수 생성기 시드 설정
torch.manual_seed(42)

# 모델 인스턴스 생성
net = ImprovedCNN(2)
def train_one_epoch(net, optimizer, train_loader):
    # Cross Entropy loss for classification when not using a softmax layer in the network
    loss = nn.CrossEntropyLoss()

    net.train()
    avg_loss = 0
    for data, target in train_loader:
        optimizer.zero_grad()
        output = net(data)
        loss_net = loss(output, target.long())
        loss_net.backward()
        optimizer.step()
        avg_loss += loss_net.item()

    return avg_loss / len(train_loader)


# Create the tiny CNN with 10 output classes
N_EPOCHS = 150

# Create a train data loader여기서 x_train이 원래 x_train2였는데
# x_train으로 바꿈
train_dataset = TensorDataset(torch.Tensor(x_train2), torch.Tensor(y_train))
train_dataloader = DataLoader(train_dataset, batch_size=64)

# Create a test data loader to supply batches for network evaluation (test)
test_dataset = TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test))
test_dataloader = DataLoader(test_dataset)

# Train the network with Adam, output the test set accuracy every epoch
net2 = TinyCNN(2)
losses_bits = []
optimizer = torch.optim.Adam(net.parameters())
for _ in tqdm(range(N_EPOCHS), desc="Training"):
    losses_bits.append(train_one_epoch(net, optimizer, train_dataloader))

fig = plt.figure(figsize=(8, 4))
plt.plot(losses_bits)
plt.ylabel("Cross Entropy Loss")
plt.xlabel("Epoch")
plt.title("Training set loss during training")
plt.grid(True)
plt.show()


def test_torch(net, test_loader):
    """Test the network: measure accuracy on the test set."""

    # Freeze normalization layers
    net.eval()

    all_y_pred = np.zeros((len(test_loader)), dtype=np.int64)
    all_targets = np.zeros((len(test_loader)), dtype=np.int64)

    # Iterate over the batches
    idx = 0
    for data, target in test_loader:
        # Accumulate the ground truth labels
        endidx = idx + target.shape[0]
        all_targets[idx:endidx] = target.numpy()

        # Run forward and get the predicted class id
        output = net(data).argmax(1).detach().numpy()
        all_y_pred[idx:endidx] = output

        idx += target.shape[0]

    # Print out the accuracy as a percentage
    n_correct = np.sum(all_targets == all_y_pred)
    print(
        f"Test accuracy for fp32 weights and activations: "
        f"{n_correct / len(test_loader) * 100:.2f}%"
    )




def test_with_concrete(quantized_module, test_loader, use_sim):
    """Test a neural network that is quantized and compiled with Concrete ML."""

    # Casting the inputs into int64 is recommended
    all_y_pred = np.zeros((len(test_loader)), dtype=np.int64)
    all_targets = np.zeros((len(test_loader)), dtype=np.int64)

    # Iterate over the test batches and accumulate predictions and ground truth labels in a vector
    idx = 0
    for data, target in tqdm(test_loader):
        data = data.numpy()
        target = target.numpy()

        fhe_mode = "simulate" if use_sim else "execute"

        # Quantize the inputs and cast to appropriate data type
        y_pred = quantized_module.forward(data, fhe=fhe_mode)
        print("y_pred")
        print(y_pred)
        print(idx)
        endidx = idx + target.shape[0]

        # Accumulate the ground truth labels
        all_targets[idx:endidx] = target

        # Get the predicted class id and accumulate the predictions
        y_pred = np.argmax(y_pred, axis=1)
        all_y_pred[idx:endidx] = y_pred

        # Update the index
        idx += target.shape[0]


        for i, (pred, true) in enumerate(zip(y_pred, target)):
            print(f"Image {i+1}: Predicted: {pred}, Actual: {true}")

    # Compute and report results
    n_correct = np.sum(all_targets == all_y_pred)

    return n_correct / len(test_loader)

n_bits = 7

use_gpu_if_available = False
devicew = "cpu"# "cuda" if use_gpu_if_available and check_gpu_available() else "cpu"
from concrete.fhe import Configuration

config = Configuration(
    enable_unsafe_features=True,
    use_insecure_key_cache=True,
    insecure_key_cache_location="~/.cml_keycache"
)

q_module = compile_torch_model(
    torch_model=net, 
    torch_inputset=x_train2, 
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
fhe_directory = '/home/giuk/fhe_client_server_files_128/'
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
start_time = time.time()

accs = test_with_concrete(
    q_module,
    test_dataloader,
    use_sim=True,
)
sim_time = time.time() - start_time

data_test = img_flat8
#y_pred_test = q_module.forward(data_test, fhe="execute")
print(f"Simulated FHE execution for {n_bits} bit network accuracy: {accs:.2f}%")
#print(y_pred_test)
test_torch(net, test_dataloader)




print("훈련 세트 크기:", x_train.shape[0])
print("테스트 세트 크기:", x_test.shape[0])

print("\n훈련 세트 첫 번째 이미지 형태:", x_train[0].shape)
print("테스트 세트 첫 번째 이미지 형태:", x_test[0].shape)


# 훈련 세트 이미지들의 특성 출력
print("\n훈련 세트 이미지들의 특성:")
for i, img in enumerate(x_train):
    print(f"Image {i+1} - 평균: {img.mean():.4f}, 최대값: {img.max():.4f}, 최소값: {img.min():.4f}")

# 테스트 세트 이미지들의 특성 출력
print("\n테스트 세트 이미지들의 특성:")
for i, img in enumerate(x_test):
    print(f"Image {i+1} - 평균: {img.mean():.4f}, 최대값: {img.max():.4f}, 최소값: {img.min():.4f}")

# 원본 이미지들의 특성 계산 및 출력
original_images = [img_array1, img_array2, img_array3, img_array4, img_array5, img_array6, img_array7, img_array8, img_array9]

print("\n원본 이미지들의 특성:")
for i, img in enumerate(original_images):
    print(f"Original Image {i+1} - 평균: {img.mean():.4f}, 최대값: {img.max():.4f}, 최소값: {img.min():.4f}")






"""
# Setup the client
client = FHEModelClient(path_dir=fhe_directory, key_dir="/home/giuk/keys_client/")
serialized_evaluation_keys = client.get_serialized_evaluation_keys()
"""
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


"""


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

"""
