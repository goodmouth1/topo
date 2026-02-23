import pickle
import numpy as np
import os
import sys
import scipy.sparse as sp
import pandas as pd
import gzip

# ==========================================
# 🔧 Theano/cuDNN Path Setup (Automatic)
# ==========================================
env_root = os.path.dirname(os.path.dirname(sys.executable))
inc_path = os.path.join(env_root, 'include')
lib_path = os.path.join(env_root, 'lib')
dnn_flags = f"dnn.include_path={inc_path},dnn.library_path={lib_path}"
# Update os.environ['THEANO_FLAGS']
cur_flags = os.environ.get('THEANO_FLAGS', '')
if cur_flags:
    os.environ['THEANO_FLAGS'] = cur_flags + "," + dnn_flags
else:
    os.environ['THEANO_FLAGS'] = dnn_flags
print(f"🔧 Configured Theano flags: {os.environ['THEANO_FLAGS']}")

# ==========================================
# 🚨 [중요] Theano/Lasagne 호환성 패치
# ==========================================
try:
    import theano.tensor.signal.pool
    theano.tensor.signal.downsample = theano.tensor.signal.pool
    print("✅ Theano 호환성 패치 적용 완료")
except:
    pass

import lasagne
import theano
import theano.tensor as T

# ==========================================
# ⚙️ 설정 (본인 환경에 맞게 확인)
# ==========================================
DATA_DIR = "data"                  
DUMP_FILE = os.path.join(DATA_DIR, "dump.pkl")
MODEL_FILE = "best_model.pkl"      
LABELED_FILE = "labeling/manual_labeled_data.pkl" 
HIDDEN_SIZE = [600]                   # 안되면 [100]으로 수정

from gcnmodel import GraphConv

def haversine_np(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a)) * 6371
    return c

# 1. 데이터 로드
print("📦 데이터를 불러오는 중...")
with gzip.open(DUMP_FILE, 'rb') as f:
    data = pickle.load(f)

(A, X_train, Y_train, X_dev, Y_dev, X_test, Y_test, U_train, U_dev, U_test, classLatMedian, classLonMedian, userLocation) = data

# 유저 매핑
user_to_idx = {u: i for i, u in enumerate(U_train + U_dev + U_test)}
X = sp.vstack([X_train, X_dev, X_test]).astype('float32')
A = A.astype('float32')

# 2. 모델 로드 및 복원
print("🧠 GCN 모델 복원 중...")
clf = GraphConv(input_size=X.shape[1], output_size=len(classLatMedian), 
                hid_size_list=HIDDEN_SIZE, regul_coef=1e-6, drop_out=0.5)
clf.build_model(A, use_text=True, use_labels=False)

try:
    with gzip.open(MODEL_FILE, 'rb') as f:
        saved_weights = pickle.load(f)
except OSError: # gzip이 아닐 경우 대비
    with open(MODEL_FILE, 'rb') as f:
        saved_weights = pickle.load(f)

lasagne.layers.set_all_param_values(clf.l_out, saved_weights)

# 3. 정답 데이터 로드
with open(LABELED_FILE, 'rb') as f:
    labeled_data = pickle.load(f)

# 4. 실제 예측 및 채점
print("⏳ GCN이 문제를 푸는 중입니다 (데이터 전수 조사)...")
results = []
target_ids = []
target_indices = []

for item in labeled_data:
    if item['user_id'] in user_to_idx:
        target_ids.append(item['user_id'])
        target_indices.append(user_to_idx[item['user_id']])

# 모델 예측 (Batch Prediction)
if len(target_indices) > 0:
    preds, probs = clf.predict(X, A, np.array(target_indices, dtype='int32'))
else:
    print("❌ 경고: 평가할 대상 유저를 찾을 수 없습니다.")
    sys.exit()

# 상세 분석 루프
print("⚖️ 상세 채점표 작성 중 (GCN vs 거리 vs 인구수)...")
for i, item in enumerate(labeled_data):
    if item['user_id'] not in target_ids: continue
    
    # 해당 유저의 예측 결과 인덱스 찾기
    try:
        pred_idx = target_ids.index(item['user_id'])
        pred_class = preds[pred_idx]
    except ValueError:
        continue

    # GCN 예측 좌표
    gcn_loc = (classLatMedian[str(pred_class)], classLonMedian[str(pred_class)])
    
    topo = item['toponyms'][0]
    candidates = topo['candidates']
    human_idx = topo['human_label_idx']
    true_loc = (candidates[human_idx]['lat'], candidates[human_idx]['lon'])
    
    # ---------------------------------------------------------
    # 1️⃣ GCN 선택 (예측 좌표와 가장 가까운 후보)
    # ---------------------------------------------------------
    gcn_final_idx = np.argmin([haversine_np(*gcn_loc, c['lat'], c['lon']) for c in candidates])
    
    # ---------------------------------------------------------
    # 2️⃣ 거리 기반(Distance Heuristic) 선택 (유저 집과 가장 가까운 후보)
    # ---------------------------------------------------------
    heur_final_idx = np.argmin([haversine_np(*item['user_true_loc'], c['lat'], c['lon']) for c in candidates])
    
    # ---------------------------------------------------------
    # 3️⃣ 인구수 기반(Population Heuristic) 선택 (인구수가 가장 많은 후보)
    # ---------------------------------------------------------
    # candidates에 'population' 키가 없거나 비어있을 경우 0으로 처리
    pops = [int(c.get('population', 0) or 0) for c in candidates]
    pop_final_idx = np.argmax(pops)

    # 여행 여부 (300km 기준)
    dist_from_home = haversine_np(*item['user_true_loc'], *true_loc)
    is_travel = dist_from_home > 300
    
    results.append({
        'user_id': item['user_id'],
        'toponym_text': topo['word'], # 원문 텍스트
        'is_travel': is_travel,
        'dist_from_home': dist_from_home,
        'human_answer': candidates[human_idx]['name'] + f" ({candidates[human_idx]['state']})",
        
        # GCN 결과
        'gcn_answer': candidates[gcn_final_idx]['name'] + f" ({candidates[gcn_final_idx]['state']})",
        'gcn_correct': (gcn_final_idx == human_idx),
        
        # 거리 기반 결과
        'dist_answer': candidates[heur_final_idx]['name'] + f" ({candidates[heur_final_idx]['state']})",
        'dist_correct': (heur_final_idx == human_idx),
        
        # 인구수 기반 결과
        'pop_answer': candidates[pop_final_idx]['name'] + f" ({candidates[pop_final_idx]['state']})",
        'pop_correct': (pop_final_idx == human_idx)
    })

# 5. CSV 저장 및 요약 보고
df_res = pd.DataFrame(results)
df_res.to_csv("gcn_detailed_evaluation.csv", index=False, encoding='utf-8-sig')

total = len(df_res)
travel_df = df_res[df_res['is_travel'] == True]
local_df = df_res[df_res['is_travel'] == False]

# 정확도 계산 함수
def get_acc(df, col):
    return df[col].mean() * 100 if len(df) > 0 else 0

print("\n" + "="*80)
print(f"📊 [최종 성적표 분석 - 3파전]")
print("="*80)
print(f"{'구분':<15} | {'GCN (Ours)':<12} | {'Distance (거리)':<15} | {'Population (인구)':<15}")
print("-" * 80)
print(f"{'전체 (Total)':<15} | {get_acc(df_res, 'gcn_correct'):.1f}%{'':<6} | {get_acc(df_res, 'dist_correct'):.1f}%{'':<9} | {get_acc(df_res, 'pop_correct'):.1f}%")
print(f"{'로컬 (Local)':<15} | {get_acc(local_df, 'gcn_correct'):.1f}%{'':<6} | {get_acc(local_df, 'dist_correct'):.1f}%{'':<9} | {get_acc(local_df, 'pop_correct'):.1f}%")
print(f"{'여행 (Travel)':<15} | {get_acc(travel_df, 'gcn_correct'):.1f}%{'':<6} | {get_acc(travel_df, 'dist_correct'):.1f}%{'':<9} | {get_acc(travel_df, 'pop_correct'):.1f}%")
print("="*80)
print(f"💡 해석 팁:")
print(f"1. 로컬에서는 '거리 기반'이 압도적일 것입니다 (집 근처니까).")
print(f"2. 인구수 기반은 'New York' 같은 대도시는 잘 맞추지만, 시골이나 로컬 지명은 다 틀릴 것입니다.")
print(f"3. 우리의 목표는 '여행(Travel)'에서 GCN이 거리/인구수 기반보다 나은지 확인하는 것입니다.")
print("-" * 80)
print(f"📂 상세 리포트가 'gcn_detailed_evaluation.csv'로 저장되었습니다.")