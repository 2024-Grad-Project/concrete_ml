import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np

class FaceDeepfakeDetector:
    def __init__(self, device='cpu', mtcnn_kwargs=None, inception_kwargs=None, checkpoint_path=None):
        self.device = device
        # Initialize MTCNN for face detection
        mtcnn_kwargs = mtcnn_kwargs or {}
        self.mtcnn = MTCNN(select_largest=False, post_process=False, device=device, **mtcnn_kwargs).eval()

        # Initialize InceptionResnetV1 for deepfake detection
        inception_kwargs = inception_kwargs or {}
        self.model = InceptionResnetV1(pretrained="vggface2", classify=True, num_classes=1, **inception_kwargs).to(device).eval()

        # Load model weights if a checkpoint is provided
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise ValueError("Checkpoint path is required to load pre-trained weights.")

    def detect(self, input_image):
        """
        Detect face and predict if it's deepfake or real.
        Args:
            input_image (PIL.Image): Input image.
        Returns:
            dict: Results with confidences and preprocessed face image.
        """
        # Step 1: Face detection
        face = self.mtcnn(input_image)
        if face is None:
            return {"error": "No face detected in the image"}

        # Preprocess the detected face
        face = face.unsqueeze(0)  # Add batch dimension
        face = F.interpolate(face, size=(256, 256), mode='bilinear', align_corners=False)
        face = face.to(self.device).to(torch.float32) / 255.0

        # Step 2: Deepfake detection
        with torch.no_grad():
            output = torch.sigmoid(self.model(face).squeeze(0))
            real_confidence = 1 - output.item()
            fake_confidence = output.item()

        # Construct the result dictionary
        result = {
            "real_confidence": real_confidence,
            "fake_confidence": fake_confidence,
            "prediction": "real" if real_confidence > fake_confidence else "fake",
        }
        return result




detector = FaceDeepfakeDetector(
    device='cpu',
    checkpoint_path='./resnetinceptionv1_epoch_32.pth'
)


from PIL import Image

input_image = Image.open('./images/fake_frame_1.png')
result = detector.detect(input_image)

print("Prediction:", result['prediction'])
print("Confidence (Real):", result['real_confidence'])
print("Confidence (Fake):", result['fake_confidence'])

import torch
import torch.fx
from torch.fx import symbolic_trace
from typing import Union, Tuple
import torch
import torch.fx
from torch.fx import symbolic_trace
from typing import Union, Tuple
import torch
import torch.fx
import torch.nn as nn
from torch.fx import symbolic_trace
from typing import Union, Tuple

import torch
import torch.fx
import torch.nn as nn
from torch.fx import symbolic_trace
from copy import deepcopy
from typing import Union, Tuple

import torch
import torch.fx
import torch.nn as nn
from torch.fx import symbolic_trace, GraphModule
from copy import deepcopy
from typing import Union, Tuple

class CustomAvgPool(nn.Module):
    """Custom average pooling implementation using sum and division"""
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        batch_size, channels, height, width = x.shape
        # Reshape and sum
        x = x.reshape(batch_size, channels, -1)
        x = torch.sum(x, dim=2)
        # Divide by number of elements
        x = x / (height * width)
        return x

class FHECompatibleTransformer(torch.fx.Transformer):
    def __init__(self, module: GraphModule):
        super().__init__(module)
        self.replacement_count = 0

    def call_method(self, target: str, args, kwargs):
        if target == 'mean':
            self.replacement_count += 1
            print(f"Transforming mean operation #{self.replacement_count}")
            
            x = args[0]
            dims = kwargs.get('dim', None) if kwargs else None
            
            if dims is None:
                # Global average pooling case
                if len(x.shape) == 4:  # (N, C, H, W)
                    N, C, H, W = x.shape
                    x = x.reshape(N, C, -1)
                    x = torch.sum(x, dim=-1)
                    x = x / (H * W)
                else:
                    x = torch.sum(x) / x.numel()
                return x
            else:
                if isinstance(dims, (list, tuple)):
                    num_elements = 1
                    for dim in dims:
                        num_elements *= x.shape[dim]
                else:
                    num_elements = x.shape[dims]
                
                x = torch.sum(x, dim=dims, keepdim=kwargs.get('keepdim', False))
                x = x / num_elements
                return x
                
        return super().call_method(target, args, kwargs)

    def call_function(self, target, args, kwargs):
        if target == torch.mean:
            self.replacement_count += 1
            print(f"Transforming torch.mean operation #{self.replacement_count}")
            
            x = args[0]
            dims = args[1] if len(args) > 1 else kwargs.get('dim', None)
            
            if dims is None:
                if len(x.shape) == 4:  # (N, C, H, W)
                    N, C, H, W = x.shape
                    x = x.reshape(N, C, -1)
                    x = torch.sum(x, dim=-1)
                    x = x / (H * W)
                else:
                    x = torch.sum(x) / x.numel()
                return x
            else:
                if isinstance(dims, (list, tuple)):
                    num_elements = 1
                    for dim in dims:
                        num_elements *= x.shape[dim]
                else:
                    num_elements = x.shape[dims]
                
                x = torch.sum(x, dim=dims, keepdim=kwargs.get('keepdim', False))
                x = x / num_elements
                return x
                
        return super().call_function(target, args, kwargs)

def replace_avg_pool_layers(model: nn.Module) -> nn.Module:
    """Replace all average pooling layers in the model with custom implementations"""
    for name, module in model.named_children():
        if isinstance(module, nn.AdaptiveAvgPool2d):
            setattr(model, name, CustomAvgPool())
        else:
            replace_avg_pool_layers(module)
    return model

def transform_model_for_fhe(model: nn.Module) -> nn.Module:
    """Transform a PyTorch model to be more FHE-compatible."""
    # First make a deep copy
    model_copy = deepcopy(model)
    
    # Put in eval mode
    model_copy.eval()
    
    # Replace average pooling layers first
    model_copy = replace_avg_pool_layers(model_copy)
    
    # Create symbolic trace
    try:
        traced_model = symbolic_trace(model_copy)
        
        # Apply transformations
        transformer = FHECompatibleTransformer(traced_model)
        transformed_model = transformer.transform()
        
        print(f"Transformed {transformer.replacement_count} operations for FHE compatibility")
        
        return transformed_model
    except Exception as e:
        print(f"Error during model transformation: {str(e)}")
        return model_copy  # Return the partially transformed model if tracing fails






# Apply transformations
transformed_model = transform_model_for_fhe(detector.model)

from concrete.ml.torch.compile import compile_torch_model
from concrete.fhe import Configuration
config = Configuration(
    enable_unsafe_features=True,
    use_insecure_key_cache=True,
    insecure_key_cache_location="~/.cml_keycache"
)




from torchvision import transforms
# Preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Convert the PIL image to a tensor
image_tensor = preprocess(input_image).unsqueeze(0)  # Add batch dimension
print("now, we start compile")

quantized_module = compile_torch_model(
    torch_model = transformed_model,  # Use the converted model
    torch_inputset = image_tensor,  # Use the converted model
    import_qat=False,
    configuration = config,
    artifacts = None,
    show_mlir=False,
    n_bits = 7,    # Number of quantization bits
    rounding_threshold_bits= {"n_bits": 7, "method": "approximate"},
    p_error=0.05,  # Disable error tolerance
    global_p_error = None,
    verbose= False,
    inputs_encryption_status = None,
    reduce_sum_copy= False,
    device = "cpu"
)




