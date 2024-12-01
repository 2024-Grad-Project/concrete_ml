import time

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
    """여러 개의 벡터를 2차원 배열로 결합합니다."""
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
print("img_flat1")
print(img_flat1)
print(img_flat2)
result_img = combine_vectors(img_flat1, img_flat2, img_flat3, img_flat4, img_flat5, img_flat6, img_flat7, img_flat8, img_flat9)
ydd=np.array([0, 0, 0, 0, 1,0,0,1,0])
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
    result_img, ydd, test_size=0.25, shuffle=True, random_state=42
)#X, y
x_train2 = x_train.reshape(-1, 3, 64, 64)
x_test = x_test.reshape(-1, 3, 64, 64)




class TinyCNN(nn.Module):
    """A very small CNN to classify the sklearn digits data-set."""

    def __init__(self, n_classes) -> None:
        """Construct the CNN with a configurable number of classes."""
        super().__init__()

        # This network has a total complexity of 1216 MAC
        self.conv1 = nn.Conv2d(3, 8, 3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(16, 32,2, stride=1, padding=0)
        #self.conv1 = nn.Conv2d(1, 8, 3, stride=1, padding=0)
        #self.conv2 = nn.Conv2d(8, 16, 3, stride=2, padding=0)
        #self.conv3 = nn.Conv2d(16, 32, 2, stride=1, padding=0)


        self.fc1 = nn.Linear(26912, n_classes)

    def forward(self, x):
        """Run inference on the tiny CNN, apply the decision layer on the reshaped conv output."""
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x = torch.relu(x)
        print(x.shape)
        x = x.flatten(1)


        x = self.fc1(x)
        return x
    

    torch.manual_seed(42)


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

# Create a train data loader
train_dataset = TensorDataset(torch.Tensor(x_train2), torch.Tensor(y_train))
train_dataloader = DataLoader(train_dataset, batch_size=64)

# Create a test data loader to supply batches for network evaluation (test)
test_dataset = TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test))
test_dataloader = DataLoader(test_dataset)

# Train the network with Adam, output the test set accuracy every epoch
net = TinyCNN(2)
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

n_bits = 6

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
    n_bits=6,  # n_bits 값을 줄여보세요.
    rounding_threshold_bits={"n_bits": 6, "method":"APPROXIMATE"},  # 기본 값을 사용해 볼 수 있습니다.
    p_error=0.05,  # 오류 허용 값을 비활성화
    global_p_error = None,
    verbose=False,
    inputs_encryption_status = None,
    reduce_sum_copy=False,
    device = "cpu"
)

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