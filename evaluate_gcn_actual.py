import pickle
import numpy as np
import os
import sys
import scipy.sparse as sp
import gzip

# ==========================================
# 🚨 [중요] 호환성 패치 (import lasagne보다 먼저 실행되어야 함)
# ==========================================
try:
    import theano.tensor.signal.pool
    # 'pool' 모듈을 'downsample'이라는 이름으로 복사해서 Lasagne를 속임
    theano.tensor.signal.downsample = theano.tensor.signal.pool
    print("✅ Theano 'downsample' 호환성 패치 적용 완료")
except ImportError:
    print("⚠️ Theano 패치 실패 (이미 구버전일 수 있음)")
except Exception as e:
    print(f"⚠️ 패치 중 에러 발생: {e}")

# 이제 안심하고 라이브러리 로드
import lasagne
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

# ==========================================
# ⚙️ 설정 (경로 확인 필수!)
# ==========================================
DATA_DIR = "data"                  # dump.pkl 있는 폴더
DUMP_FILE = os.path.join(DATA_DIR, "dump.pkl")
MODEL_FILE = "best_model.pkl"      # 학습된 GCN 모델 파일
LABELED_FILE = "labeling/manual_labeled_data.pkl" # 호날두 님이 만든 정답 데이터
HIDDEN_SIZE = [600]                   # 모델 학습 때 쓴 히든 레이어 크기 (보통 [600] or [100])

from gcnmodel import GraphConv

# 거리 계산 함수
def haversine_np(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a)) * 6371
    return c

print(f"🚀 [최종 평가] GCN 모델 실제 채점 시작...")

# 1. 데이터 로드 (dump.pkl)
if not os.path.exists(DUMP_FILE):
    print(f"❌ 데이터 파일이 없습니다: {DUMP_FILE}")
    sys.exit(1)

print(f"📦 데이터 로딩 중: {DUMP_FILE}...")
with gzip.open(DUMP_FILE, 'rb') as f:
    data = pickle.load(f)

# 데이터 분해 (gcnmain.py 저장 구조)
(A, X_train, Y_train, X_dev, Y_dev, X_test, Y_test, U_train, U_dev, U_test, classLatMedian, classLonMedian, userLocation) = data

# 2. 유저 매핑 (User ID -> Graph Index)
print("🗺️ 유저 인덱스 매핑 중...")
user_to_idx = {}
offset = 0
for u_list in [U_train, U_dev, U_test]:
    for u in u_list:
        user_to_idx[u] = offset
        offset += 1

# 전체 행렬 스택
X = sp.vstack([X_train, X_dev, X_test]).astype('float32')
A = A.astype('float32')

# 3. 모델 로드
print(f"🧠 GCN 모델({MODEL_FILE}) 복원 중...")
input_size = X.shape[1]
output_size = len(classLatMedian) # 클래스 개수

clf = GraphConv(input_size=input_size, output_size=output_size, 
                hid_size_list=HIDDEN_SIZE, regul_coef=1e-6, drop_out=0.5)
clf.build_model(A, use_text=True, use_labels=False)

# 가중치 덮어쓰기
if not os.path.exists(MODEL_FILE):
    print(f"❌ 모델 파일이 없습니다: {MODEL_FILE}")
    sys.exit(1)

with gzip.open(MODEL_FILE, 'rb') as f:
    saved_weights = pickle.load(f)
lasagne.layers.set_all_param_values(clf.l_out, saved_weights)

# 4. 수제 라벨 데이터 로드 및 예측 대상 선정
print(f"📝 정답지({LABELED_FILE}) 로드 중...")
with open(LABELED_FILE, 'rb') as f:
    labeled_data = pickle.load(f)

target_indices = []
valid_items = []

for item in labeled_data:
    if item['user_id'] in user_to_idx:
        target_indices.append(user_to_idx[item['user_id']])
        valid_items.append(item)

target_indices = np.array(target_indices, dtype='int32')
print(f"👉 평가 대상: 총 {len(labeled_data)}명 중 {len(valid_items)}명 (그래프에 포함된 유저)")

if len(target_indices) == 0:
    print("❌ 평가할 대상이 없습니다. (ID 매칭 실패)")
    sys.exit()

# 5. GCN 예측 수행 (시험 치는 중...)
print("⏳ GCN이 문제를 풀고 있습니다...")
preds, probs = clf.predict(X, A, target_indices)

# 6. 채점 (Scoring)
print("⚖️ 채점 중...")

correct_gcn = 0
correct_heuristic = 0
total_travel_cases = 0
correct_gcn_travel = 0
correct_heuristic_travel = 0 # 0이어야 정상

total = 0

for i, item in enumerate(valid_items):
    # GCN의 답 (Class ID -> 좌표 변환)
    pred_class = preds[i]
    gcn_loc = (classLatMedian[str(pred_class)], classLonMedian[str(pred_class)])
    
    # 정답지 정보
    if not item['toponyms']: continue
    topo = item['toponyms'][0]
    if 'human_label_idx' not in topo: continue
    
    candidates = topo['candidates']
    human_idx = topo['human_label_idx']
    true_cand = candidates[human_idx]
    true_loc = (true_cand['lat'], true_cand['lon'])
    
    # 여행 여부 판단 (집에서 300km 이상)
    is_travel = haversine_np(*item['user_true_loc'], *true_loc) > 300
    if is_travel: total_travel_cases += 1
    
    # (A) GCN 채점: 예측 좌표와 가장 가까운 후보 선택
    best_cand_idx = -1
    min_dist = float('inf')
    for c_idx, cand in enumerate(candidates):
        d = haversine_np(*gcn_loc, cand['lat'], cand['lon'])
        if d < min_dist:
            min_dist = d
            best_cand_idx = c_idx
            
    if best_cand_idx == human_idx:
        correct_gcn += 1
        if is_travel: correct_gcn_travel += 1
        
    # (B) Heuristic 채점: 집에서 가장 가까운 후보 선택
    heur_idx = -1
    min_h_dist = float('inf')
    for c_idx, cand in enumerate(candidates):
        # 이미 dist_from_user가 계산되어 있을 수 있음
        d = haversine_np(*item['user_true_loc'], cand['lat'], cand['lon'])
        if d < min_h_dist:
            min_h_dist = d
            heur_idx = c_idx
            
    if heur_idx == human_idx:
        correct_heuristic += 1
        if is_travel: correct_heuristic_travel += 1 # 우연히 맞을 수도 있음
        
    total += 1

# 7. 결과 출력
acc_gcn = correct_gcn / total * 100
acc_heur = correct_heuristic / total * 100

print("\n" + "="*60)
print(f"📊 [최종 성적표] 총 {total}문제 채점 완료")
print("="*60)
print(f"📏 기존 방식(Heuristic): {acc_heur:.2f}%")
print(f"🏆 GCN 모델(Ours)      : {acc_gcn:.2f}%")
print("-" * 60)
print(f"✈️ 여행/출장(Non-local) 난제 해결력 ({total_travel_cases}건)")
print(f"   - 기존 방식: {correct_heuristic_travel}개 정답")
print(f"   - GCN 모델 : {correct_gcn_travel}개 정답 ({(correct_gcn_travel/total_travel_cases)*100:.1f}%)")
print("="*60)

# 8. 그래프 저장
methods = ['Distance Heuristic', 'GCN (Ours)']
scores = [acc_heur, acc_gcn]
colors = ['#3498db', '#2ecc71']

plt.figure(figsize=(8, 6))
bars = plt.bar(methods, scores, color=colors, width=0.5, edgecolor='black')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, height + 1, f'{height:.1f}%', 
             ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.ylabel('Accuracy (%)')
plt.title('Final Performance Comparison')
plt.ylim(0, 100)
plt.savefig('final_result_graph_actual.png')
print("✅ 그래프 저장 완료: final_result_graph_actual.png")