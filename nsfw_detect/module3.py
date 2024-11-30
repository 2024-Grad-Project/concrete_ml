import torch
from torch import nn
from torchvision import models
from torchvision import transforms
from PIL import Image
from torch.autograd import Variable
import matplotlib.pyplot as plt
import os
from torch.fx import symbolic_trace
import time
start = time.time() #The variable for measuring the compilation time.


####################################################################################################################################################################
####################################################################################################################################################################
# Image preprocessing transformations
test_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
testimage = Image.open('./img.jpg')
image_tensor3 = test_transforms(testimage).float()
image_tensor3 = image_tensor3.unsqueeze_(0)

iamage_input = Variable(image_tensor3)




####################################################################################################################################################################
####################################################################################################################################################################
#Fine-tuning ML models using operations supported by Zama Concrete ML
import torch.fx

# Define a transformation class to replace mean operation with matrix multiplication and scaling
class MeanToMatMulTransform(torch.fx.Transformer):
    def call_method(self, target: str, args, kwargs):
        # Replace the mean operation when found
        if target == 'mean':
            print(f"Replacing 'mean' with scaling at node {target}")
            
            # Flatten the input tensor and scale by dividing with H * W
            x = args[0]
            H, W = x.shape[2], x.shape[3]
            num_elements = H * W

            # Apply matrix multiplication and scaling instead of mean
            x = x.flatten(2)  # (N, C, H*W)
            x = x.sum(dim=-1)  # Sum the H * W dimension
            x = x / num_elements  # Approximate mean by dividing by H * W elements
            return x  # Return the transformed result
        return super().call_method(target, args, kwargs)
    


class CustomResNet50(nn.Module):
    def __init__(self):
        super(CustomResNet50, self).__init__()
        original_model = models.resnet50() 
        self.conv1 = original_model.conv1
        self.bn1 = original_model.bn1
        self.relu = original_model.relu
        self.maxpool = original_model.maxpool
        self.layer1 = original_model.layer1
        self.layer2 = original_model.layer2
        self.layer3 = original_model.layer3
        self.layer4 = original_model.layer4

        # Use sigmoid instead of softmax
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 10),
            nn.Sigmoid()  # softmax 대신 sigmoid 사용
        )

        # Pre-generate index_weights to match the number of classes (fixed at 10 here)
        self.register_buffer("index_weights", torch.arange(10).view(1, -1).float())

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        N, C, H, W = x.shape
        x = x.view(N, C, H * W)
        x = x.sum(dim=-1) / (H * W)
        
        # Obtain values passed through sigmoid
        activations = self.fc(x)
        
        # Approximate the index of the maximum activation value (for argmax)
        indices = self.get_approximate_argmax(activations, scale_factor=10)
        
        return indices  

    def get_approximate_argmax(self, activations, scale_factor=100):
        # Amplify the values by scale_factor and apply exponentiation to emphasize larger values
        exponents = torch.exp(activations * scale_factor)

        # Calculate weighted_sum using the entire index_weights
        weighted_sum = (exponents * self.index_weights).sum(dim=1)

        # Normalize by the sum of exponents to approximate the index
        normalization_factor = exponents.sum(dim=1)
        approximate_indices = (weighted_sum / normalization_factor).float()  # INT64 -> FLOAT 변환

        return approximate_indices.round()

# Create an instance of the model
model = CustomResNet50()

# Load pre-trained weights
state_dict = torch.load('ResNet50_nsfw_model.pth', map_location=torch.device('cpu'))
model.load_state_dict(state_dict, strict=False)
model.eval()

# Trace the model using Symbolic Trace
traced = symbolic_trace(model)

# Replace mean operation with matrix multiplication and scaling
transformed_model = MeanToMatMulTransform(traced).transform()

####################################################################################################################################################################
####################################################################################################################################################################
# Compile the model using Concrete-ML
from concrete.fhe import Configuration
config = Configuration(
    enable_unsafe_features=True,
    use_insecure_key_cache=True,
    insecure_key_cache_location="~/.cml_keycache"
)

from concrete.ml.torch.compile import compile_torch_model
print("Now, we start compile")

quantized_module = compile_torch_model(
    transformed_model,  # Use the transformed model
    iamage_input ,  # Input tensor
    import_qat=False,
    configuration = config,
    artifacts = None,
    show_mlir=False,
    n_bits = 7,  # Number of quantization bits
    rounding_threshold_bits= {"n_bits": 7, "method": "approximate"},
    p_error=0.05,  # Deactivate allowable error value
    global_p_error = None,
    verbose= False,
    inputs_encryption_status = None,
    reduce_sum_copy= False,
    device = "cpu"
)
print("here is after compile")

output_fhe = quantized_module.forward(iamage_input.numpy())


print(output_fhe)
fhe_end = time.time()
fhe_time = fhe_end - start
print(f"FHE execution completed in {fhe_time:.4f} seconds")


####################################################################################################################################################################
####################################################################################################################################################################
# build a module containing the Concrete ML model inference function
print("Now, we start compile with modules")
from concrete import fhe
@fhe.module()
class Counter:
    @fhe.function({"image": "encrypted"})
    def classify(image):
        #indices = transformed_model(image)
        indices = quantized_module.forward(image)
        
        return indices

    @fhe.function({"x": "encrypted", "y": "clear"})
    def mul(x, y):
        return (x * y) # y is constant



fhe_end = time.time()
fhe_time = fhe_end - start
print(f"FHE execution completed in {fhe_time:.4f} seconds")


CounterFhe = Counter.compile({"classify": [iamage_input.numpy()], "mul": [(4, 3)]})
x_enc = CounterFhe.classify.encrypt(iamage_input.numpy())
index = CounterFhe.classify.run(x_enc)
result = CounterFhe.mul.run(index, 10)

print("NSFW Index:", index)
print("Result after multiplying with constant:", result)