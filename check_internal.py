# ------------------------------------------------------------------
# [호환성 패치] Theano/Lasagne
# ------------------------------------------------------------------
import theano.tensor.signal.pool
import theano.tensor.signal
import sys
theano.tensor.signal.downsample = theano.tensor.signal.pool
# ------------------------------------------------------------------

import gcnmodel
import inspect
import numpy as np

print("🔍 GraphConv 모델 내부 구조를 심층 해부합니다...")

# 1. 모델 객체 생성
try:
    # 더미 값으로 생성
    clf = gcnmodel.GraphConv(
        input_size=100, 
        output_size=10, 
        hid_size_list=[300], 
        regul_coef=0.001, 
        drop_out=0.5
    )
    print("✅ 모델 객체(껍데기) 생성 성공!")
except Exception as e:
    print(f"⚠️ 생성 실패 ({e}). 기본 생성자로 시도...")
    clf = gcnmodel.GraphConv()

# ------------------------------------------------------------------
# [핵심] build_model()을 호출해서 엔진을 강제로 만듭니다!
# ------------------------------------------------------------------
print("🔨 build_model() 호출 중...")
try:
    # build_model이 인자가 필요한지 확인
    sig = inspect.signature(clf.build_model)
    print(f"ℹ️ build_model 인자: {list(sig.parameters.keys())}")
    
    # 인자 없이 호출 시도 (보통 self 변수를 씀)
    clf.build_model()
    print("✅ build_model() 실행 성공! 엔진이 생성되었을 겁니다.")
except Exception as e:
    print(f"⚠️ build_model() 실행 중 에러: {e}")
    # 인자가 필요하다면 더미 데이터라도 넣어서 실행해야 함 (여기선 일단 패스)

# ------------------------------------------------------------------

# 2. 객체가 가진 모든 속성 다시 스캔
attributes = dir(clf)
public_attributes = [attr for attr in attributes if not attr.startswith('__')]

print("\n" + "="*40)
print("📂 [최종 확인] 시동 건 후 변수 목록:")
print("="*40)
print(public_attributes)
print("="*40 + "\n")

# 3. Lasagne Layer 찾기
print("🕵️‍♂️ 범인(Network Layer) 최종 추적...")
found = False
for name in public_attributes:
    try:
        attr = getattr(clf, name)
        type_str = str(type(attr))
        
        # 이름이 수상하거나, 타입이 레이어인 경우
        if 'Layer' in type_str or name in ['network', 'net', 'l_out', 'l_in', 'model', 'prob']:
            print(f"👉 [검거 완료] self.{name}  (Type: {type_str})")
            found = True
    except:
        pass

if not found:
    print("❌ 여전히 엔진을 못 찾았습니다. 위 변수 목록을 보여주세요.")