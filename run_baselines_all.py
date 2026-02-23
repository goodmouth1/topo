import pickle
import numpy as np
import os
import sys
import scipy.sparse as sp
import pandas as pd
import gzip
import time

# ==========================================
# 🔧 Theano/Lasagne 설정 및 패치
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
# ⚙️ 설정
# ==========================================
DATA_DIR = "data"
DUMP_FILE = os.path.join(DATA_DIR, "dump.pkl")
INPUT_CSV = "gcn_detailed_evaluation.csv" 
LABELED_FILE = "topo/manual_labeled_data.pkl"
HIDDEN_SIZE = [600] 
LEARNING_RATE = 0.001
EPOCHS = 50 

def haversine_np(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a)) * 6371
    return c

# ==========================================
# 🧠 1. MLP 모델 클래스 (Text-Only)
# ==========================================
class MLP:
    def __init__(self, input_size, output_size, hid_size_list, drop_out=0.5):
        l_in = lasagne.layers.InputLayer(shape=(None, input_size))
        
        layer = l_in
        if drop_out > 0:
            layer = lasagne.layers.DropoutLayer(layer, p=drop_out)
        
        for hid in hid_size_list:
            layer = lasagne.layers.DenseLayer(layer, num_units=hid, nonlinearity=lasagne.nonlinearities.rectify)
            if drop_out > 0:
                layer = lasagne.layers.DropoutLayer(layer, p=drop_out)
        
        self.l_out = lasagne.layers.DenseLayer(layer, num_units=output_size, nonlinearity=lasagne.nonlinearities.softmax)
        
        X_sym = T.matrix()
        y_sym = T.ivector()
        
        pred_train = lasagne.layers.get_output(self.l_out, X_sym, deterministic=False)
        loss = T.mean(lasagne.objectives.categorical_crossentropy(pred_train, y_sym))
        
        params = lasagne.layers.get_all_params(self.l_out, trainable=True)
        updates = lasagne.updates.adam(loss, params, learning_rate=LEARNING_RATE)
        
        self.train_fn = theano.function([X_sym, y_sym], loss, updates=updates)
        
        pred_test = lasagne.layers.get_output(self.l_out, X_sym, deterministic=True)
        self.predict_fn = theano.function([X_sym], pred_test)

    def train(self, X, y):
        return self.train_fn(X, y)

    def predict(self, X):
        return self.predict_fn(X)

# ==========================================
# 🚀 메인 로직
# ==========================================

print("📦 데이터 로딩 중...")
# gzip 처리가 안 되어 있을 수도 있으니 예외처리
try:
    with gzip.open(DUMP_FILE, 'rb') as f:
        data = pickle.load(f)
except OSError:
    with open(DUMP_FILE, 'rb') as f:
        data = pickle.load(f)

(A, X_train, Y_train, X_dev, Y_dev, X_test, Y_test, U_train, U_dev, U_test, classLatMedian, classLonMedian, userLocation) = data

# 전체 X
X_all = sp.vstack([X_train, X_dev, X_test]).astype('float32')

# 🚨 [수정됨] Y_train 차원 확인 및 처리
# Y_train이 1D(Vector)인지 2D(One-hot)인지 확인
if len(Y_train.shape) == 1:
    print("ℹ️ Label format: 1D Index Array (No argmax needed)")
    y_train_indices = Y_train.astype('int32')
else:
    print("ℹ️ Label format: 2D One-hot Matrix (Applying argmax)")
    y_train_indices = np.argmax(Y_train, axis=1).astype('int32')

num_train = X_train.shape[0]

# 유저 매핑
user_to_idx = {u: i for i, u in enumerate(U_train + U_dev + U_test)}

# 기존 결과 로드
if not os.path.exists(INPUT_CSV):
    print(f"❌ '{INPUT_CSV}' 파일이 없습니다. 이전 단계(인구수 비교)를 먼저 실행하세요.")
    sys.exit()
df_res = pd.read_csv(INPUT_CSV)

# 정답 데이터 로드
with open(LABELED_FILE, 'rb') as f:
    labeled_data = pickle.load(f)

# ==========================================
# 🤖 2. MLP 학습 및 예측 (Text-Only)
# ==========================================
print("\n" + "="*50)
print(f"🔥 [1/2] MLP (Text-Only) 모델 학습 시작 (Epochs: {EPOCHS})")
print("="*50)

input_size = X_all.shape[1]
output_size = len(classLatMedian)
mlp_model = MLP(input_size, output_size, HIDDEN_SIZE)

# 메모리 절약을 위해 X_train을 Dense로 변환 (MLP 입력용)
# 데이터가 너무 크면 배치 처리가 필요하지만, 일단 변환 시도
try:
    X_train_dense = X_train.toarray()
except:
    print("⚠️ 메모리 부족으로 Sparse Matrix 그대로 사용 시도 (느릴 수 있음)")
    X_train_dense = X_train

for epoch in range(EPOCHS):
    start_time = time.time()
    loss = mlp_model.train(X_train_dense, y_train_indices)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss:.4f} | Time: {time.time()-start_time:.2f}s")

print("✅ MLP 학습 완료! 예측 수행 중...")

target_indices = []
for idx, row in df_res.iterrows():
    u_id = row['user_id']
    if u_id in user_to_idx:
        target_indices.append(user_to_idx[u_id])
    else:
        target_indices.append(0) # 예외 처리 (없는 유저)

# 타겟 유저에 대해서만 예측
X_target_dense = X_all[target_indices].toarray()
mlp_preds_proba = mlp_model.predict(X_target_dense)
mlp_pred_classes = np.argmax(mlp_preds_proba, axis=1)

# ==========================================
# 🕸️ 3. LPA 예측 (Graph-Only)
# ==========================================
print("\n" + "="*50)
print(f"🕸️ [2/2] LPA (Label Propagation) 계산 시작")
print("="*50)

# A_train_part: 전체 유저(N) x 트레이닝 유저(Train_N)
A_all_train = A[:, :num_train]

# 라벨 전파: A * Y_train (One-hot 변환 필요)
# Y_train이 1D라면 One-hot으로 변환해서 곱해야 함
if len(Y_train.shape) == 1:
    num_classes = output_size
    Y_train_onehot = np.zeros((num_train, num_classes), dtype='float32')
    Y_train_onehot[np.arange(num_train), y_train_indices] = 1.0
else:
    Y_train_onehot = Y_train

lpa_scores = A_all_train.dot(Y_train_onehot)
lpa_pred_classes = np.argmax(lpa_scores, axis=1)

# ==========================================
# 📝 4. 채점 및 병합
# ==========================================
print("⚖️ 채점 및 결과 병합 중...")

mlp_correct_list = []
lpa_correct_list = []

# 빠른 검색을 위한 매핑
label_map = {(item['user_id'], item['toponyms'][0]['word']): item for item in labeled_data if item['toponyms']}

for i, row in df_res.iterrows():
    uid = row['user_id']
    if uid not in user_to_idx:
        mlp_correct_list.append(False)
        lpa_correct_list.append(False)
        continue
    
    # 1. MLP 채점
    mlp_cls = mlp_pred_classes[i]
    mlp_loc = (classLatMedian[str(mlp_cls)], classLonMedian[str(mlp_cls)])
    
    # 2. LPA 채점
    u_idx = user_to_idx[uid]
    lpa_cls = lpa_pred_classes[u_idx]
    lpa_loc = (classLatMedian[str(lpa_cls)], classLonMedian[str(lpa_cls)])
    
    # 정답 가져오기
    key = (uid, row['toponym_text'])
    if key not in label_map:
        # 텍스트 매칭 실패 시 fallback (user_id만으로 검색 - 첫번째꺼)
        found = False
        for item in labeled_data:
            if item['user_id'] == uid:
                item_cand = item
                found = True
                break
        if not found:
            mlp_correct_list.append(False)
            lpa_correct_list.append(False)
            continue
    else:
        item_cand = label_map[key]

    candidates = item_cand['toponyms'][0]['candidates']
    human_idx = item_cand['toponyms'][0]['human_label_idx']
    
    # MLP 매칭
    mlp_cand_idx = np.argmin([haversine_np(*mlp_loc, c['lat'], c['lon']) for c in candidates])
    mlp_correct_list.append(mlp_cand_idx == human_idx)
    
    # LPA 매칭
    lpa_cand_idx = np.argmin([haversine_np(*lpa_loc, c['lat'], c['lon']) for c in candidates])
    lpa_correct_list.append(lpa_cand_idx == human_idx)

df_res['mlp_correct'] = mlp_correct_list
df_res['lpa_correct'] = lpa_correct_list

OUTPUT_FILE = "gcn_detailed_evaluation_ALL.csv"
df_res.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

# ==========================================
# 📊 최종 요약
# ==========================================
def get_acc(df, col):
    return df[col].mean() * 100 if len(df) > 0 else 0

travel_df = df_res[df_res['is_travel'] == True]

print("\n" + "="*80)
print(f"🏆 [최종 5파전 성적표] (총 {len(df_res)}건)")
print("="*80)
print(f"{'모델':<20} | {'전체 정확도':<12} | {'여행(Travel) 정확도':<15}")
print("-" * 80)
print(f"{'1. GCN (Ours)':<20} | {get_acc(df_res, 'gcn_correct'):.1f}%{'':<7} | {get_acc(travel_df, 'gcn_correct'):.1f}%  <-- ★ Main")
print(f"{'2. Distance':<20} | {get_acc(df_res, 'dist_correct'):.1f}%{'':<7} | {get_acc(travel_df, 'dist_correct'):.1f}%")
print(f"{'3. Population':<20} | {get_acc(df_res, 'pop_correct'):.1f}%{'':<7} | {get_acc(travel_df, 'pop_correct'):.1f}%")
print(f"{'4. MLP (Text-only)':<20} | {get_acc(df_res, 'mlp_correct'):.1f}%{'':<7} | {get_acc(travel_df, 'mlp_correct'):.1f}%")
print(f"{'5. LPA (Graph-only)':<20} | {get_acc(df_res, 'lpa_correct'):.1f}%{'':<7} | {get_acc(travel_df, 'lpa_correct'):.1f}%")
print("="*80)
print(f"📂 결과 저장 완료: {OUTPUT_FILE}")