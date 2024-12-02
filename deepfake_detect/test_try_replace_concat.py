import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
from torch.fx import symbolic_trace


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

'''
from PIL import Image
input_image = Image.open('./images/fake_frame_1.png')
result = detector.detect(input_image)
'''


# Define custom transformation class
class MeanToMatMulTransform(torch.fx.Transformer):
    def call_method(self, target: str, args, kwargs):
        if target == 'mean':
            print(f"Replacing 'mean' with matrix multiplication and scaling at node {target}")
            
            # Flatten the input tensor and scale it by H * W
            x = args[0]
            H, W = x.shape[2], x.shape[3]
            num_elements = H * W

            # Replace mean operation with matrix multiplication and scaling
            x = x.flatten(2)  # (N, C, H*W)
            x = x.sum(dim=-1)  # Sum over H * W dimension
            x = x / num_elements  # Approximate mean by dividing by H * W
            return x  # Return the transformed result
        return super().call_method(target, args, kwargs)


# Define the manual_rescale function
def manual_rescale(tensors):
    """
    Convert all input tensors to have the same scale and zero_point.
    Args:
        tensors (list of torch.Tensor): List of input tensors.
    Returns:
        list of torch.Tensor: List of tensors with the same scale and zero_point.
    """
    # Get reference scale and zero_point from the first tensor
    target_scale = tensors[0].q_scale()  # Quantization scale
    target_zero_point = tensors[0].q_zero_point()  # Quantization zero_point

    # Convert all tensors to the same scale and zero_point
    rescaled_tensors = [
        tensor.dequantize().mul(target_scale / tensor.q_scale()).add(
            target_zero_point - tensor.q_zero_point()
        ).to(dtype=torch.float32)
        for tensor in tensors
    ]

    # Requantize tensors back to quantized format
    quantized_tensors = [
        torch.quantize_per_tensor(rescaled_tensor, scale=target_scale, zero_point=target_zero_point, dtype=torch.qint8)
        for rescaled_tensor in rescaled_tensors
    ]
    return quantized_tensors

# Modify the manual_rescale function
def manual_rescale(tensors):
    """
    Convert all input tensors to have the same scale and zero_point.
    """
    # Define reference scale and zero_point
    target_scale = 1.0
    target_zero_point = 0

    # Convert all tensors to the same scale and zero_point
    rescaled_tensors = []
    for tensor in tensors:
        # Dequantize if the tensor is quantized
        if hasattr(tensor, "dequantize"):
            tensor = tensor.dequantize()
        rescaled_tensor = tensor * target_scale + target_zero_point
        rescaled_tensors.append(rescaled_tensor)
    return rescaled_tensors


# Modify the ReplaceConcat class
class ReplaceConcat(torch.fx.Transformer):
    def call_function(self, target: callable, args, kwargs):
        if target == torch.cat:
            #print(f"Replacing 'torch.cat' at Node: {args}")
            tensors = args[0]  # List of input tensors for 'cat' operation
            rescaled_tensors = manual_rescale(tensors)  # Rescale tensors
            return torch.cat(rescaled_tensors, dim=args[1])  # Recombine tensors
        return super().call_function(target, args, kwargs)

    def call_module(self, target: str, args, kwargs):
        if "cat" in target:
            #print(f"Replacing 'torch.cat' module at Node: {args}")
            tensors = args[0]
            rescaled_tensors = manual_rescale(tensors)
            return torch.cat(rescaled_tensors, dim=args[1])
        return super().call_module(target, args, kwargs)


# Execute model transformation
traced_model = symbolic_trace(detector.model)


# Remove GlobalAveragePool operation & apply avgpool_1a node transformation
# transformed_model_1 = ReplaceGlobalAveragePool(traced_model).transform()

# Replace mean operations
transformed_model_2 = MeanToMatMulTransform(traced_model).transform()

# Trace the model with Symbolic Trace
traced_model = symbolic_trace(detector.model)


transformed_model_3 = ReplaceConcat(transformed_model_2).transform()


from PIL import Image

input_image = Image.open('./images/fake_frame_1.png')
result = detector.detect(input_image) 
# result2 = transformed_model_final(input_image)
print("Prediction:", result['prediction'])
print("Confidence (Real):", result['real_confidence'])
print("Confidence (Fake):", result['fake_confidence'])


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
    torch_model = transformed_model_3,  # Use the transformed model
    torch_inputset = image_tensor,  # Input tensor
    import_qat=True,
    configuration = config,
    artifacts = None,
    show_mlir=False,
    n_bits = 7,  # Number of quantization bits
    rounding_threshold_bits= {"n_bits": 7, "method": "approximate"},
    p_error=0.05,  # Disable error tolerance
    global_p_error = None,
    verbose= False,
    inputs_encryption_status = None,
    reduce_sum_copy= False,
    device = "cpu"
)
