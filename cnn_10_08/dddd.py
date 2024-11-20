import torch
import numpy as np
from concrete import fhe
from concrete.ml.torch.compile import compile_torch_model
from PIL import Image
from torchvision import transforms

# Define your CNN model
class ImprovedCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        # CNN layers and fully connected layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, n_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 16 * 16)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# Define the composition module
@fhe.module()
class ModelWithRandomMultiplier:
    def __init__(self, model):
        self.model = model
    
    @fhe.function({"x": "encrypted"})
    def compute_model_output(self, x):
        return self.model(x)
    
    @fhe.function({"output": "encrypted"})
    def apply_random_multiplier(self, output, random_constant):
        return output * random_constant

    # Define the composition
    composition = fhe.Wired([
        fhe.Wire(fhe.Output(compute_model_output, 0), fhe.Input(apply_random_multiplier, 0))
    ])

# Load image and preprocess
def image_to_tensor(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(image).unsqueeze(0)
    return tensor

# Instantiate and compile the model
image_path = './images/image5.jpg'
sample_input = image_to_tensor(image_path)

model = ImprovedCNN(n_classes=2)
composed_model = ModelWithRandomMultiplier(model)
config = fhe.Configuration(enable_unsafe_features=True, use_insecure_key_cache=True)
compiled_model = compile_torch_model(
    torch_model=composed_model, 
    torch_inputset=sample_input, 
    configuration=config
)

# Save and deploy model
#fhe_directory = '/home/giuk/zama_fhe_directory/test_4/'
#dev = FHEModelDev(path_dir=fhe_directory, model=compiled_model)
#dev.save()
