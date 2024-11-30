from concrete import fhe

@fhe.module()
class Counter:
    @fhe.function({"x": "encrypted"})
    def add(x):
        return (x + 1) % 20  # 덧셈

    @fhe.function({"x": "encrypted", "y": "clear"})
    def mul(x, y):
        return (x * y) % 20  # 곱셈
    

# 입력셋 설정 ( 각 함수별 입력셋 구분 )
inputset = list(range(20))
inputset2 = [(20, 3), (5, 3), (2, 2), (3, 1), (1, 4), (11, 1), (15, 1), (17, 2)]
CounterFhe = Counter.compile({"inc": inputset, "dec": inputset2})

# `inc` 함수 실행 테스트
x = 5
x_enc = CounterFhe.inc.encrypt(x)
x_inc_enc = CounterFhe.inc.run(x_enc)
x_inc = CounterFhe.inc.decrypt(x_inc_enc)
assert x_inc == 6
print(x_inc)
# `dec` 함수 실행 테스트
x_inc_dec_enc = CounterFhe.dec.run(x_inc_enc, 3)
x_inc_dec = CounterFhe.dec.decrypt(x_inc_dec_enc)
#assert x_inc_dec == 12  # 예상 결과 수정 (x_inc * 2 % 20 = 12)
print(x_inc_dec)
# `inc` 함수 반복 실행 테스트
for _ in range(10):
    x_enc = CounterFhe.inc.run(x_enc)
x_dec = CounterFhe.inc.decrypt(x_enc)
#assert x_dec == 15
print(x_dec)