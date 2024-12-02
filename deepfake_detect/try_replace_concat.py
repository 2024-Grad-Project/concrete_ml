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


# Custom 변환 클래스 정의
class MeanToMatMulTransform(torch.fx.Transformer):
    def call_method(self, target: str, args, kwargs):
        if target == 'mean':
            print(f"Replacing 'mean' with matrix multiplication and scaling at node {target}")
            
            # 입력 텐서를 펼친 후, H * W로 나눠 스케일링
            x = args[0]
            H, W = x.shape[2], x.shape[3]
            num_elements = H * W

            # 평균을 내는 대신 행렬 곱과 스케일링 적용
            x = x.flatten(2)  # (N, C, H*W)
            x = x.sum(dim=-1)  # H * W 차원을 sum으로 합침
            x = x / num_elements  # H * W 요소로 나눠 평균 근사
            return x  # 변환된 결과 반환
        return super().call_method(target, args, kwargs)





# manual_rescale 함수 정의
def manual_rescale(tensors):
    """
    모든 입력 텐서를 동일한 scale과 zero_point로 변환합니다.
    Args:
        tensors (list of torch.Tensor): 입력 텐서 리스트.
    Returns:
        list of torch.Tensor: 동일한 scale과 zero_point로 변환된 텐서 리스트.
    """
    # 기준 scale과 zero_point를 첫 번째 텐서에서 가져옴
    target_scale = tensors[0].q_scale()  # Quantization scale
    target_zero_point = tensors[0].q_zero_point()  # Quantization zero_point

    # 모든 텐서를 동일한 scale과 zero_point로 변환
    rescaled_tensors = [
        tensor.dequantize().mul(target_scale / tensor.q_scale()).add(
            target_zero_point - tensor.q_zero_point()
        ).to(dtype=torch.float32)
        for tensor in tensors
    ]

    # 다시 양자화된 텐서로 변환
    quantized_tensors = [
        torch.quantize_per_tensor(rescaled_tensor, scale=target_scale, zero_point=target_zero_point, dtype=torch.qint8)
        for rescaled_tensor in rescaled_tensors
    ]
    return quantized_tensors

# manual_rescale 함수 수정
def manual_rescale(tensors):
    """
    모든 입력 텐서를 동일한 scale과 zero_point로 변환합니다.
    """
    # 기준 스케일과 제로 포인트 정의
    target_scale = 1.0
    target_zero_point = 0

    # 모든 텐서를 동일한 스케일과 제로 포인트로 변환
    rescaled_tensors = []
    for tensor in tensors:
        # 양자화된 텐서라면 디양자화 후 스케일 맞추기
        if hasattr(tensor, "dequantize"):
            tensor = tensor.dequantize()
        rescaled_tensor = tensor * target_scale + target_zero_point
        rescaled_tensors.append(rescaled_tensor)
    return rescaled_tensors


# ReplaceConcat 클래스 수정
class ReplaceConcat(torch.fx.Transformer):
    def call_function(self, target: callable, args, kwargs):
        if target == torch.cat:
            #print(f"Replacing 'torch.cat' at Node: {args}")
            tensors = args[0]  # 'cat' 연산의 입력 텐서 리스트
            rescaled_tensors = manual_rescale(tensors)  # 스케일 맞추기
            return torch.cat(rescaled_tensors, dim=args[1])  # 재결합
        return super().call_function(target, args, kwargs)

    def call_module(self, target: str, args, kwargs):
        if "cat" in target:
            #print(f"Replacing 'torch.cat' module at Node: {args}")
            tensors = args[0]
            rescaled_tensors = manual_rescale(tensors)
            return torch.cat(rescaled_tensors, dim=args[1])
        return super().call_module(target, args, kwargs)







# 모델 변환 실행
traced_model = symbolic_trace(detector.model)


# GlobalAveragePool 연산 제거 & avgpool_1a 노드 변환 적용
#transformed_model_1 = ReplaceGlobalAveragePool(traced_model).transform()

# Mean 연산 대체
transformed_model_2 = MeanToMatMulTransform(traced_model).transform()

# 모델을 Symbolic Trace로 추적
traced_model = symbolic_trace(detector.model)


transformed_model_3 = ReplaceConcat(transformed_model_2).transform()





from PIL import Image

input_image = Image.open('./images/fake_frame_1.png')
result = detector.detect(input_image) 
#result2 = transformed_model_final(input_image)
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
# 전처리 파이프라인
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# PIL 이미지를 텐서로 변환
image_tensor = preprocess(input_image).unsqueeze(0)  # Add batch dimension
print("now, we start compile")
quantized_module = compile_torch_model(
    torch_model = transformed_model_3,  # 변환된 모델 사용
    torch_inputset = image_tensor,  # 입력 텐서
    import_qat=True,
    configuration = config,
    artifacts = None,
    show_mlir=False,
    n_bits = 7,  # 양자화 비트 수
    rounding_threshold_bits= {"n_bits": 7, "method": "approximate"},
    p_error=0.05,  # 오류 허용 값을 비활성화
    global_p_error = None,
    verbose= False,
    inputs_encryption_status = None,
    reduce_sum_copy= False,
    device = "cpu"
)




