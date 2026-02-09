import pickle
import numpy as np
import os
import sys
import scipy.sparse as sp
import pandas as pd
import gzip

# ==========================================
# � Theano/cuDNN Path Setup (Automatic)
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
# �🚨 [중요] Theano/Lasagne 호환성 패치
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
import matplotlib.pyplot as plt

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

with gzip.open(MODEL_FILE, 'rb') as f:
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

# 모델 예측
preds, probs = clf.predict(X, A, np.array(target_indices, dtype='int32'))

# 상세 분석 루프
print("⚖️ 상세 채점표 작성 중...")
for i, item in enumerate(labeled_data):
    if item['user_id'] not in target_ids: continue
    
    # GCN 결과
    pred_class = preds[i]
    gcn_loc = (classLatMedian[str(pred_class)], classLonMedian[str(pred_class)])
    
    topo = item['toponyms'][0]
    candidates = topo['candidates']
    human_idx = topo['human_label_idx']
    true_loc = (candidates[human_idx]['lat'], candidates[human_idx]['lon'])
    
    # GCN이 선택한 최종 후보 찾기 (가장 가까운 후보)
    gcn_final_idx = np.argmin([haversine_np(*gcn_loc, c['lat'], c['lon']) for c in candidates])
    
    # 거리 기반(Heuristic)이 선택한 후보 찾기
    heur_final_idx = np.argmin([haversine_np(*item['user_true_loc'], c['lat'], c['lon']) for c in candidates])
    
    # 여행 여부 (300km 기준)
    dist_from_home = haversine_np(*item['user_true_loc'], *true_loc)
    is_travel = dist_from_home > 300
    
    results.append({
        'user_id': item['user_id'],
        'toponym': topo['word'],
        'is_travel': is_travel,
        'dist_from_home': dist_from_home,
        'human_answer': candidates[human_idx]['name'] + f" ({candidates[human_idx]['state']})",
        'gcn_answer': candidates[gcn_final_idx]['name'] + f" ({candidates[gcn_final_idx]['state']})",
        'heuristic_answer': candidates[heur_final_idx]['name'] + f" ({candidates[heur_final_idx]['state']})",
        'gcn_correct': (gcn_final_idx == human_idx),
        'heuristic_correct': (heur_final_idx == human_idx)
    })

# 5. CSV 저장 및 요약 보고
df_res = pd.DataFrame(results)
df_res.to_csv("gcn_detailed_evaluation.csv", index=False, encoding='utf-8-sig')

total = len(df_res)
travel_df = df_res[df_res['is_travel'] == True]
local_df = df_res[df_res['is_travel'] == False]

print("\n" + "="*60)
print(f"📊 [최종 성적표 분석]")
print("-" * 60)
print(f"✅ 전체 정확도 - GCN: {df_res['gcn_correct'].mean()*100:.1f}% | Heuristic: {df_res['heuristic_correct'].mean()*100:.1f}%")
print(f"✈️ 여행자 케이스({len(travel_df)}건) - GCN: {travel_df['gcn_correct'].mean()*100:.1f}% | Heuristic: {travel_df['heuristic_correct'].mean()*100:.1f}%")
print(f"🏠 로컬 케이스({len(local_df)}건) - GCN: {local_df['gcn_correct'].mean()*100:.1f}% | Heuristic: {local_df['heuristic_correct'].mean()*100:.1f}%")
print("="*60)
print(f"📂 상세 리포트가 'gcn_detailed_evaluation.csv'로 저장되었습니다.")