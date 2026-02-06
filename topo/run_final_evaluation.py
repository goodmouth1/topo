import pickle
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
from haversine import haversine
import scipy.sparse as sp

# Monkey patch for Theano/Lasagne compatibility
try:
    import theano.tensor.signal.pool
    theano.tensor.signal.downsample = theano.tensor.signal.pool
except ImportError:
    pass

import lasagne
import theano
import theano.tensor as T
from gcnmodel import GraphConv

# ==========================================
# ⚙️ 설정
# ==========================================
DATA_DIR = "/home/wang/master_thesis/data"
DUMP_FILE = os.path.join(DATA_DIR, "dump.pkl")
MODEL_FILE = "../best_model.pkl"
LABELED_FILE = "../labeling/manual_labeled_data.pkl"

print("🚀 [Phase 5] 최종 GCN 성능 평가 시작...")

# 1. 수제 라벨 데이터 로드
if not os.path.exists(LABELED_FILE):
    print(f"❌ 오류: {LABELED_FILE} 파일이 없습니다.")
    sys.exit()

with open(LABELED_FILE, 'rb') as f:
    labeled_data = pickle.load(f)

print(f"📖 수작업 검증 데이터 로드 완료: {len(labeled_data)}명의 유저")

# 2. GCN 입력 데이터 로드
print(f"📦 모델 입력 데이터(dump.pkl) 로드 중...")
try:
    with open(DUMP_FILE, 'rb') as f:
        data = pickle.load(f)
except:
    import gzip
    with gzip.open(DUMP_FILE, 'rb') as f:
        data = pickle.load(f)

# 데이터 분해
# gcnmain.py 저장 순서: (A, X_train, Y_train, X_dev, Y_dev, X_test, Y_test, ...)
A = data[0].astype('float32')
X_train, X_dev, X_test = data[1], data[3], data[5]
Y_test = data[6]
U_test = data[9] # 유저 ID 리스트 (Test Set)

# 전체 행렬 스택 (인덱싱을 위해)
X = sp.vstack([X_train, X_dev, X_test]).astype('float32')
test_start_idx = X_train.shape[0] + X_dev.shape[0]

# 유저 ID -> Test Set 내의 인덱스 매핑
# U_test는 test_start_idx부터 시작하는 순서대로 들어있음
user_id_to_test_idx = {uid: i for i, uid in enumerate(U_test)}

# 3. 모델 로드 및 예측
print("🧠 GCN 모델 복원 및 예측 수행 중...")
input_size = X.shape[1]
output_size = np.max(Y_test) + 1

# 모델 그래프 정의 (학습 때와 동일해야 함)
clf = GraphConv(
    input_size=input_size, 
    output_size=output_size, 
    hid_size_list=[600], 
    regul_coef=1e-6, 
    drop_out=0.5, 
    highway=False
)
clf.build_model(A, use_text=True, use_labels=False)

# 가중치 불러오기
with open(MODEL_FILE, 'rb') as f:
    try:
        saved_weights = pickle.load(f)
    except:
        import gzip
        with gzip.open(MODEL_FILE, 'rb') as f:
            saved_weights = pickle.load(f)
            
lasagne.layers.set_all_param_values(clf.l_out, saved_weights)

# 좌표 매핑 정보 로드 (Class ID -> Lat/Lon)
from data import DataLoader
dl = DataLoader(data_home=DATA_DIR, bucket_size=300, encoding='utf-8')
dl.load_data()
dl.assignClasses()
classLatMedian = {c: dl.cluster_median[c][0] for c in dl.cluster_median}
classLonMedian = {c: dl.cluster_median[c][1] for c in dl.cluster_median}

# 전체 예측 수행 (Batch로 하거나 한 번에)
# Test Set 인덱스만 추출
test_indices = np.arange(test_start_idx, X.shape[0]).astype('int32')
preds, probs = clf.predict(X, A, test_indices)

# 4. 채점 (Evaluation)
print("\n⚖️ 채점 진행 중... (호날두 님의 정답지와 비교)")

total_toponyms = 0
correct_gcn = 0
correct_random = 0
correct_heuristic = 0 # (참고용) 유저 실제 위치와 가장 가까운 후보

# 킬러 문항 분석용 (User Correction Cases)
hard_cases_total = 0
hard_cases_gcn_correct = 0

for item in labeled_data:
    user_id = item['user_id']
    user_true_loc = item['user_true_loc']
    
    if user_id not in user_id_to_test_idx:
        continue # Test 셋에 없는 유저 (혹시 모를 예외)
        
    # GCN 예측 위치 가져오기
    relative_idx = user_id_to_test_idx[user_id]
    pred_class = preds[relative_idx]
    gcn_pred_loc = (classLatMedian[pred_class], classLonMedian[pred_class])
    
    for topo in item['toponyms']:
        if 'human_label_idx' not in topo:
            continue # 라벨링 안 된 건 패스
            
        candidates = topo['candidates']
        human_idx = topo['human_label_idx']
        
        # 1. GCN의 선택: 예측된 위치와 가장 가까운 후보
        best_cand_idx = -1
        min_dist = float('inf')
        for i, cand in enumerate(candidates):
            dist = haversine(gcn_pred_loc, (cand['lat'], cand['lon']))
            if dist < min_dist:
                min_dist = dist
                best_cand_idx = i
        
        # 2. Heuristic(거리기반)의 선택: 실제 위치와 가장 가까운 후보 (0번일 확률 높음)
        # (이미 candidates가 거리순 정렬되어 있다고 가정하면 0번이지만, 안전하게 다시 계산)
        heuristic_idx = -1
        min_h_dist = float('inf')
        for i, cand in enumerate(candidates):
            dist = haversine(user_true_loc, (cand['lat'], cand['lon']))
            if dist < min_h_dist:
                min_h_dist = dist
                heuristic_idx = i

        # 3. 채점
        total_toponyms += 1
        
        # GCN 정답 여부
        if best_cand_idx == human_idx:
            correct_gcn += 1
            
        # Random 정답 여부
        if random.randint(0, len(candidates)-1) == human_idx:
            correct_random += 1
            
        # Heuristic 정답 여부 (Baseline)
        if heuristic_idx == human_idx:
            correct_heuristic += 1
        else:
            # 여기가 바로 "Hard Case" (거리 기반 가정이 틀린 경우)
            hard_cases_total += 1
            if best_cand_idx == human_idx:
                hard_cases_gcn_correct += 1

# 5. 결과 리포트
acc_gcn = correct_gcn / total_toponyms * 100
acc_random = correct_random / total_toponyms * 100
acc_heuristic = correct_heuristic / total_toponyms * 100

print("\n" + "="*60)
print(f"📊 [최종 성적표] 총 평가 문제: {total_toponyms}개")
print("="*60)
print(f"🎲 Random Baseline 정확도    : {acc_random:.2f}%")
print(f"📏 Distance Heuristic 정확도 : {acc_heuristic:.2f}% (Baseline)")
print(f"🏆 GCN Model 정확도          : {acc_gcn:.2f}% (Ours)")
print("-" * 60)

if hard_cases_total > 0:
    hard_acc = hard_cases_gcn_correct / hard_cases_total * 100
    print(f"🔥 [킬러 문항 분석] 총 {hard_cases_total}개의 '여행/출장' 의심 케이스 중")
    print(f"   👉 GCN이 맞춰낸 개수: {hard_cases_gcn_correct}개 ({hard_acc:.2f}%)")
    if hard_cases_gcn_correct > 0:
        print("   💡 대박! 단순 거리 계산으로는 절대 못 맞추는 문제를 GCN이 풀어냈습니다!")
    else:
        print("   😅 아쉽게도 킬러 문항은 GCN에게도 어려웠습니다.")
else:
    print("   (킬러 문항이 발견되지 않았습니다. 모든 정답이 집 근처였습니다.)")
    
print("="*60 + "\n")