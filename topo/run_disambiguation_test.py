import sys
import os

# 1. 상위 폴더 모듈(gcnmodel, data 등)을 불러오기 위해 경로 추가
sys.path.append("..")

# 2. [호환성 패치] Theano downsample -> pool 매핑
# Lasagne가 구버전 Theano API(downsample)를 찾아서 나는 에러 해결
import theano.tensor.signal.pool
import theano.tensor.signal
theano.tensor.signal.downsample = theano.tensor.signal.pool

import pickle
import numpy as np
import random
from haversine import haversine
import scipy.sparse as sp
import lasagne
from gcnmodel import GraphConv

# 1. 설정
DATA_DIR = "../data"
DUMP_FILE = os.path.join(DATA_DIR, "dump.pkl")
GT_FILE = "test_with_ground_truth.pkl"
MODEL_FILE = "../model.pkl" # best_model.pkl이 없다면 model.pkl 사용 (확인 필요)

print("🚀 [최종 실험] GCN 기반 지명 중의성 해소 평가 시작...")

# 2. 데이터 로드 (Ground Truth)
print(f"📖 정답지 로드 중: {GT_FILE}")
with open(GT_FILE, 'rb') as f:
    ground_truth_data = pickle.load(f)

# 3. 데이터 로드 (GCN 입력용)
print(f"📦 모델 입력 데이터 로드 중: {DUMP_FILE}")
try:
    with open(DUMP_FILE, 'rb') as f:
        data = pickle.load(f)
except:
    import gzip
    with gzip.open(DUMP_FILE, 'rb') as f:
        data = pickle.load(f)

# 데이터 분해 및 스택
A = data[0].astype('float32')
X_train, X_dev, X_test = data[1], data[3], data[5]
Y_test = data[6]
U_test = data[9] # 유저 ID 리스트

# U_test 리스트를 {User_ID: Index} 딕셔너리로 변환 (빠른 검색용)
user_id_to_idx = {uid: i for i, uid in enumerate(U_test)}

# 전체 데이터 스택
X = sp.vstack([X_train, X_dev, X_test]).astype('float32')
test_start_idx = X_train.shape[0] + X_dev.shape[0]
test_indices = np.arange(test_start_idx, X.shape[0]).astype('int32')

# 4. 챔피언 모델 로드 및 예측
print("🧠 GCN 모델 로드 및 예측 수행 중...")
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
clf.build_model(A, use_text=True, use_labels=False)

# 가중치 주입
with open(MODEL_FILE, 'rb') as f: # pickle.load는 gzip 여부 자동 처리 안될 수 있으니 주의
    try:
        saved_weights = pickle.load(f)
    except:
        import gzip
        with gzip.open(MODEL_FILE, 'rb') as f:
            saved_weights = pickle.load(f)
            
lasagne.layers.set_all_param_values(clf.l_out, saved_weights)

# 예측 (좌표 정보 필요)
# 주의: 모델은 Class ID를 뱉으므로, 이를 좌표로 변환해야 함
from data import DataLoader
dl = DataLoader(data_home=DATA_DIR, bucket_size=300, encoding='utf-8')
dl.load_data() # 데이터 로드
dl.assignClasses() # [수정] 클러스터링 수행 및 cluster_median 생성

classLatMedian = {c: dl.cluster_median[c][0] for c in dl.cluster_median}
classLonMedian = {c: dl.cluster_median[c][1] for c in dl.cluster_median}

preds, probs = clf.predict(X, A, test_indices)

# 5. 채점 (Evaluation)
print("\n⚖️ 채점 진행 중...")

gcn_correct_count = 0
random_correct_count = 0
total_cases = 0

for item in ground_truth_data:
    user_id = item['user_id']
    
    # 이 유저가 Test 셋 어디에 있는지 찾기
    if user_id not in user_id_to_idx:
        continue # 혹시 모를 예외 처리
        
    # Test 셋 내에서의 상대 인덱스
    relative_idx = user_id_to_idx[user_id]
    
    # 모델의 예측 (Class ID -> Lat/Lon)
    pred_class = preds[relative_idx]
    gcn_pred_loc = (classLatMedian[pred_class], classLonMedian[pred_class])
    
    # 각 지명(Toponym)에 대해 평가
    for toponym_info in item['toponyms']:
        word = toponym_info['word']
        candidates = toponym_info['candidates']
        true_loc = (toponym_info['true_lat'], toponym_info['true_lon'])
        
        # [GCN 전략] 예측된 유저 위치와 가장 가까운 후보 선택
        gcn_best_cand = None
        gcn_min_dist = float('inf')
        
        for cand in candidates:
            cand_loc = (cand['lat'], cand['lon'])
            dist = haversine(gcn_pred_loc, cand_loc)
            if dist < gcn_min_dist:
                gcn_min_dist = dist
                gcn_best_cand = cand
                
        # [Random 전략] 그냥 아무거나 찍기 (비교용)
        random_cand = random.choice(candidates)
        
        # 정답 여부 확인 (좌표가 일치하면 정답)
        # float 비교이므로 아주 작은 오차 허용 (0.0001)
        if abs(gcn_best_cand['lat'] - true_loc[0]) < 0.0001 and abs(gcn_best_cand['lon'] - true_loc[1]) < 0.0001:
            gcn_correct_count += 1
            
        if abs(random_cand['lat'] - true_loc[0]) < 0.0001 and abs(random_cand['lon'] - true_loc[1]) < 0.0001:
            random_correct_count += 1
            
        total_cases += 1

# 6. 결과 출력
gcn_acc = (gcn_correct_count / total_cases) * 100
random_acc = (random_correct_count / total_cases) * 100

print("\n" + "="*50)
print(f"📊 [최종 성적표] 총 평가 건수: {total_cases} 건")
print("-" * 50)
print(f"🎲 Random Baseline 정확도 : {random_acc:.2f}%")
print(f"🏆 GCN Disambiguation 정확도: {gcn_acc:.2f}%")
print("-" * 50)

improvement = gcn_acc - random_acc
print(f"💡 결론: GCN을 사용하면 랜덤 추측보다 {improvement:.2f}%p 더 정확하게 지명을 찾아냅니다!")
print("="*50 + "\n")