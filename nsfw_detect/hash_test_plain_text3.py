import torch

def approximate_hash_function(indices, y, z):
    def mod(a, b):
        # Modulo 구현
        div_result = torch.div(a, b)
        floor_result = torch.floor(div_result)
        product = torch.mul(b, floor_result)
        return torch.sub(a, product)

    # Step 1: Scale inputs
    scaled_indices = indices * 0.1
    scaled_y = y * 1e-11
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

# 테스트 입력
indices = torch.tensor(5.0)  # PyTorch 텐서
y = torch.tensor(20241125182911.0)
z = torch.tensor(1919.0)

# 함수 실행
result = approximate_hash_function(indices, y, z)
print(f"Result: {result}")
