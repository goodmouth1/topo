import pickle
import numpy as np
import os
import sys
# 상위 디렉토리 모듈 임포트
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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
# ⚙️ 설정 (호날두 님 환경에 맞춤)
# ==========================================
DATA_DIR = "/home/wang/master_thesis/data"
DUMP_FILE = os.path.join(DATA_DIR, "dump.pkl")
MODEL_FILE = "../best_model.pkl"
LABELED_FILE = "../labeling/manual_labeled_data.pkl"

# 색상 코드
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
BOLD = '\033[1m'
END = '\033[0m'

print("🚀 상세 결과 분석 시작...")

# 1. 수제 라벨 데이터 로드
if not os.path.exists(LABELED_FILE):
    print(f"❌ 오류: {LABELED_FILE} 파일이 없습니다.")
    sys.exit()

with open(LABELED_FILE, 'rb') as f:
    labeled_data = pickle.load(f)

# 2. GCN 입력 데이터 로드
print(f"📦 모델 입력 데이터(dump.pkl) 로드 중...")
try:
    with open(DUMP_FILE, 'rb') as f:
        data = pickle.load(f)
except:
    import gzip
    with gzip.open(DUMP_FILE, 'rb') as f:
        data = pickle.load(f)

A = data[0].astype('float32')
X_train, X_dev, X_test = data[1], data[3], data[5]
Y_test = data[6]
U_test = data[9] 
X = sp.vstack([X_train, X_dev, X_test]).astype('float32')
test_start_idx = X_train.shape[0] + X_dev.shape[0]
user_id_to_test_idx = {uid: i for i, uid in enumerate(U_test)}

# 3. 모델 복원
print("🧠 GCN 모델 복원 및 예측 수행 중...")
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

with open(MODEL_FILE, 'rb') as f:
    try: 
        weights = pickle.load(f)
    except: 
        import gzip
        with gzip.open(MODEL_FILE, 'rb') as f: 
            weights = pickle.load(f)
lasagne.layers.set_all_param_values(clf.l_out, weights)

# 좌표 매핑 정보 로드
from data import DataLoader
dl = DataLoader(data_home=DATA_DIR, bucket_size=300, encoding='utf-8')
dl.load_data()
dl.assignClasses() # 유저 코드에서 추가된 부분 반영
classLatMedian = {c: dl.cluster_median[c][0] for c in dl.cluster_median}
classLonMedian = {c: dl.cluster_median[c][1] for c in dl.cluster_median}

test_indices = np.arange(test_start_idx, X.shape[0]).astype('int32')
preds, probs = clf.predict(X, A, test_indices)

# 4. 상세 분석 루프
print("\n🔍 케이스별 상세 분석 중...\n")

for item in labeled_data:
    user_id = item['user_id']
    user_true_loc = item['user_true_loc']
    
    if user_id not in user_id_to_test_idx: continue
    
    relative_idx = user_id_to_test_idx[user_id]
    pred_class = preds[relative_idx]
    gcn_pred_loc = (classLatMedian[pred_class], classLonMedian[pred_class])
    
    for topo in item['toponyms']:
        if 'human_label_idx' not in topo: continue
        
        candidates = topo['candidates']
        human_idx = topo['human_label_idx']
        word = topo['word']
        # orig_text가 없을 경우를 대비해 안전하게 가져오기
        tweet = topo.get('orig_text', "").replace('\n', ' ')
        
        # GCN 선택
        gcn_idx = -1
        min_dist = float('inf')
        for i, cand in enumerate(candidates):
            dist = haversine(gcn_pred_loc, (cand['lat'], cand['lon']))
            if dist < min_dist:
                min_dist = dist
                gcn_idx = i
        
        # Heuristic 선택
        heu_idx = -1
        min_h_dist = float('inf')
        for i, cand in enumerate(candidates):
            dist = haversine(user_true_loc, (cand['lat'], cand['lon']))
            if dist < min_h_dist:
                min_h_dist = dist
                heu_idx = i
        
        # 판정
        gcn_correct = (gcn_idx == human_idx)
        heu_correct = (heu_idx == human_idx)
        
        # 출력 포맷 결정
        header = ""
        if gcn_correct and not heu_correct:
            header = f"\n{MAGENTA}{BOLD}🏆 [GCN SUPER PLAY] Heuristic은 틀리고 GCN이 맞춤! (논문감){END}"
        elif not gcn_correct and heu_correct:
            header = f"\n{YELLOW}�� [GCN MISS] Heuristic은 맞는데 GCN이 틀림 (집 근처인데 실수){END}"
        elif gcn_correct and heu_correct:
            # 둘 다 맞춘 건 너무 많으니 패스 (필요하면 주석 해제)
            # header = f"\n{GREEN}✅ [BOTH CORRECT] 둘 다 정답{END}"
            continue 
        else:
            header = f"\n{RED}❌ [BOTH WRONG] 둘 다 틀림 (난제){END}"
            
        print("="*80)
        print(header)
        print(f"👤 User: {user_id} | �� True Home: {user_true_loc}")
        print(f"🧠 GCN Predicted Home: {gcn_pred_loc} (Distance from True: {haversine(user_true_loc, gcn_pred_loc):.1f}km)")
        print(f"📝 Tweet: {tweet[:100]}...")
        print(f"🎯 Target: {BOLD}{word}{END}")
        
        print("-" * 20)
        for i, cand in enumerate(candidates):
            marker = ""
            if i == human_idx: marker += f" {GREEN}👈[정답]{END}"
            if i == gcn_idx: marker += f" {BLUE}🧠[GCN선택]{END}"
            if i == heu_idx: marker += f" {YELLOW}📏[Heuristic선택]{END}"
            
            dist_gcn = haversine(gcn_pred_loc, (cand['lat'], cand['lon']))
            print(f"   [{i}] {cand['name']} ({cand['state']}) - GCN과의 거리: {dist_gcn:.1f}km {marker}")
        print("="*80)
