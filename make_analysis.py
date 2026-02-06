import pickle
import gzip  # 압축 해제를 위해 필요
import numpy as np
import pandas as pd
import scipy.sparse as sp
import sys
import os

# 1. [필수] Theano/Lasagne 버전 호환성 패치
try:
    import theano.tensor.signal.pool
    import theano.tensor.signal
    if not hasattr(theano.tensor.signal, 'downsample'):
        print("🔧 Patching Theano (pool -> downsample)...")
        theano.tensor.signal.downsample = theano.tensor.signal.pool
except ImportError:
    pass

import lasagne
from gcnmodel import GraphConv

print("\n📊 [오답 노트] 분석을 시작합니다...")

# 2. 데이터 로드 (수정됨: gzip.open 사용)
dump_path = './data/dump.pkl'
if not os.path.exists(dump_path):
    print("❌ Error: './data/dump.pkl' 파일이 없습니다. gcnmain.py를 한 번이라도 돌렸어야 합니다.")
    sys.exit(1)

print(f"📦 Loading data from {dump_path} (Unzipping)...")

# [수정 포인트] 일반 open() -> gzip.open()으로 변경
try:
    with gzip.open(dump_path, 'rb') as f:
        data = pickle.load(f)
except OSError:
    # 혹시 압축이 안 된 파일일 경우를 대비한 보험
    print("⚠️ Gzip 실패, 일반 파일로 다시 시도합니다...")
    with open(dump_path, 'rb') as f:
        data = pickle.load(f)

# 필요한 데이터만 쏙쏙 뽑기
A = data[0].astype('float32')
X_train, X_dev, X_test = data[1], data[3], data[5]
Y_test = data[6]
U_test = data[9] # 유저 ID

# 전체 데이터 스택 (모델 입력용)
print("📚 Stacking features for prediction...")
X = sp.vstack([X_train, X_dev, X_test]).astype('float32')

# Test Index 계산
test_start_idx = X_train.shape[0] + X_dev.shape[0]
test_indices = np.arange(test_start_idx, X.shape[0]).astype('int32')

# 3. 챔피언 모델 구조 복원 (Hidden 600, Dropout 0.5)
print("🏗 Building Champion Model (Hidden=600, Dropout=0.5)...")
input_size = X.shape[1]
output_size = np.max(Y_test) + 1

clf = GraphConv(
    input_size=input_size, 
    output_size=output_size, 
    hid_size_list=[600], 
    regul_coef=1e-6, 
    drop_out=0.5, 
    highway=False
)

# 모델 빌드
clf.build_model(A, use_text=True, use_labels=False)

# 4. 챔피언 가중치(Weights) 주입
weight_file = 'best_model.pkl'
print(f"📥 Injecting weights from {weight_file}...")
with gzip.open(weight_file, 'rb') as f:
    saved_weights = pickle.load(f)
    lasagne.layers.set_all_param_values(clf.l_out, saved_weights)

# 5. 예측 수행
print("🔮 Predicting...")
preds, probs = clf.predict(X, A, test_indices)

# 6. 결과 정리 및 CSV 저장
print("📝 Writing results to CSV...")
results = []

for i in range(len(preds)):
    user_id = U_test[i]
    true_lbl = Y_test[i]
    pred_lbl = preds[i]
    conf = probs[i][pred_lbl]
    
    is_correct = (true_lbl == pred_lbl)
    
    results.append({
        'User_ID': user_id,
        'True_Label': true_lbl,
        'Pred_Label': pred_lbl,
        'Is_Correct': 'Correct' if is_correct else 'Wrong',
        'Confidence': f"{conf:.4f}"
    })

# DataFrame으로 변환 후 저장
df = pd.DataFrame(results)
output_csv = 'final_error_analysis.csv'
df.to_csv(output_csv, index=False, encoding='utf-8-sig')

print("\n" + "="*40)
print(f"✅ 분석 완료! '{output_csv}' 파일 생성됨.")
print(f"📊 총 테스트 샘플: {len(df)}")
print(f"🏆 정답 개수: {len(df[df['Is_Correct']=='Correct'])} ({len(df[df['Is_Correct']=='Correct'])/len(df)*100:.2f}%)")
print(f"❌ 오답 개수: {len(df[df['Is_Correct']=='Wrong'])}")
print("="*40 + "\n")