import math

def mod(a, b):
    # Modulo  연산  구현
    return a - math.floor(a / b) * b
  
def custom_forward(indices, y, z):
    # Step 1: Scale inputs
    scaled_indices = indices * 0.1
    scaled_y = y * 0.2
    scaled_z = z * 0.3

    # Step 2: Combine inputs
    combined = scaled_indices + scaled_y + scaled_z

    # Step 3: Apply non-linear transformation
    transformed = math.exp(abs(math.log((combined ** 2) + 1e-5)))

    # Step 4: Apply modulo for additional complexity
    modulo_result = mod(transformed, 1.0)  # Modulo 연산으로 0~1 범위로 제한


    # Step 5: Apply final non-linear transformation
    final_hash = math.tanh(modulo_result)  # 추가 비선형성
    return final_hash

# 예제 입력
indices = 5.0  # 예: 1.0
y = 10.5        # 예: 2.0
z = 1919.0       # 예: 3.0

# 함수 실행
result = custom_forward(indices, y, z)
print(f"Result: {result}")