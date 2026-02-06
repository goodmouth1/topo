import theano
from theano import tensor as T
import numpy as np

# 1. 설정 확인
print(f"Testing Theano on device: {theano.config.device}")

# 2. 간단한 행렬 곱셈 함수 만들기
x = T.matrix('x')
y = T.matrix('y')
z = T.dot(x, y)
f = theano.function([x, y], z)

# 3. 실제로 어디서 계산하는지 뜯어보기
print("\n[Computation Graph Check]")
has_gpu = False
if any([x.op.__class__.__name__.lower().startswith('gpu') for x in f.maker.fgraph.toposort()]):
    has_gpu = True
    print("✅ SUCCESS: Found 'Gpu' operations! (GPU is working)")
else:
    print("❌ FAIL: Only CPU operations found. (Fallback to CPU)")

# 4. 연산 실행
a = np.random.randn(1000, 1000).astype('float32')
b = np.random.randn(1000, 1000).astype('float32')
f(a, b)
print("Calculation finished.")