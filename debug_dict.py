import sys, pickle, gzip, numpy as np

with open('gcn_1.0_percent_pred_930.pkl', 'rb') as f:
    preds = pickle.load(f)

y_p = np.array(preds[0].todense()) if hasattr(preds[0], 'todense') else np.array(preds[0])
y_pred = np.argmax(y_p, axis=1) if len(np.shape(y_p)) > 1 else y_p.flatten()

y_t = np.array(preds[1].todense()) if hasattr(preds[1], 'todense') else np.array(preds[1])
y_true = np.argmax(y_t, axis=1) if len(np.shape(y_t)) > 1 else y_t.flatten()

with gzip.open('./data/world/world.pkl.gz', 'rb') as f:
    data = pickle.load(f, encoding='latin1')

dicts = sorted([x for x in data if isinstance(x, dict)], key=len)
classLatMedian, classLonMedian, userLocation = dicts[0], dicts[1], dicts[2]

lists = [x for x in data if isinstance(x, list)]
U_test = lists[-1]
if len(U_test) != len(y_pred):
    match = [l for l in lists if len(l) == len(y_pred)]
    if match: U_test = match[-1]

def get_loc(u):
    if u in userLocation: return userLocation[u]
    if isinstance(u, str):
        u_b = u.encode('latin1', errors='ignore')
        if u_b in userLocation: return userLocation[u_b]
    if isinstance(u, bytes):
        u_s = u.decode('latin1', errors='ignore')
        if u_s in userLocation: return userLocation[u_s]
    return ["?", "?"]

# 1. 정답을 완벽하게 맞춘 학생들 찾기
correct_indices = np.where(y_pred == y_true)[0]
print("\n" + "🔍"*20)
print(f"✅ AI가 완벽하게 맞춘 문제 수: {len(correct_indices)}개 (이게 18.92% 입니다!)")

# 2. 첫 번째로 맞춘 문제 정밀 분석
i = correct_indices[0]
u = U_test[i]
loc = get_loc(u)
t = y_true[i]

print(f"\n[AI가 맞춘 {t}번 구역 좌표 분석]")
print(f"👤 유저 실제 좌표 : {loc[0]}, {loc[1]}")

lat_val = classLatMedian.get(t, classLatMedian.get(str(t), '못 찾음!'))
lon_val = classLonMedian.get(t, classLonMedian.get(str(t), '못 찾음!'))
print(f"📍 사전에서 찾은 {t}번 구역 위도 : {lat_val}")
print(f"📍 사전에서 찾은 {t}번 구역 경도 : {lon_val}")

print(f"\n[사전의 암호키 생김새 확인]")
print(f"🔑 위도 사전 키 형태: {type(list(classLatMedian.keys())[0])}")
print(f"🔑 위도 사전 키 예시 5개: {list(classLatMedian.keys())[:5]}")
print("🔍"*20 + "\n")
