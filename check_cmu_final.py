import sys, pickle, gzip, numpy as np

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

pkl_file = sys.argv[1] if len(sys.argv) > 1 else 'gcn_1.0_percent_pred_930.pkl'

with open(pkl_file, 'rb') as f:
    preds = pickle.load(f)

y_p = np.array(preds[0].todense()) if hasattr(preds[0], 'todense') else np.array(preds[0])
y_pred = np.argmax(y_p, axis=1) if len(np.shape(y_p)) > 1 else y_p.flatten()

# AI가 파일 안에 같이 제출한 100% 확실한 진짜 정답지 사용!
y_t = np.array(preds[1].todense()) if hasattr(preds[1], 'todense') else np.array(preds[1])
Y_test = np.argmax(y_t, axis=1) if len(np.shape(y_t)) > 1 else y_t.flatten()

acc_class = np.mean(y_pred == Y_test) * 100

# CMU 원본 데이터에서 맵핵 스캔! (여기가 ./data/cmu/cmu.pkl.gz 로 바뀜)
with gzip.open('./data/cmu/cmu.pkl.gz', 'rb') as f:
    data = pickle.load(f, encoding='latin1')

dicts = sorted([x for x in data if isinstance(x, dict)], key=len)
userLocation = dicts[-1]

def get_loc(u):
    if u in userLocation: return userLocation[u]
    if isinstance(u, str):
        u_b = u.encode('latin1', errors='ignore')
        if u_b in userLocation: return userLocation[u_b]
    if isinstance(u, bytes):
        u_s = u.decode('latin1', errors='ignore')
        if u_s in userLocation: return userLocation[u_s]
    return None

arrays = [x for x in data if isinstance(x, np.ndarray)]
lists = [x for x in data if isinstance(x, list)]
u_cands = [l for l in lists if len(l) == len(y_pred)]
U_test = u_cands[-1] if u_cands else lists[-1]

bucket_lat, bucket_lon = {}, {}
for l in lists:
    for a in arrays:
        if len(l) > 0 and len(l) == len(a):
            a_cls = np.argmax(a, axis=1) if len(np.shape(a)) > 1 else a.flatten()
            for u, y in zip(l, a_cls):
                loc = get_loc(u)
                if loc is not None:
                    try:
                        bucket_lat.setdefault(y, []).append(float(loc[0]))
                        bucket_lon.setdefault(y, []).append(float(loc[1]))
                    except: pass

safe_lat, safe_lon = {}, {}
for y in bucket_lat:
    if len(bucket_lat[y]) > 0:
        safe_lat[y] = np.median(bucket_lat[y])
        safe_lon[y] = np.median(bucket_lon[y])

lat_true, lon_true, lat_pred, lon_pred = [], [], [], []
error_count = 0

for u, p in zip(U_test, y_pred):
    loc = get_loc(u)
    if loc is None: continue
    try:
        t_lat, t_lon = float(loc[0]), float(loc[1])
        if p not in safe_lat:
            error_count += 1
            continue
        
        p_lat = safe_lat[p]
        p_lon = safe_lon[p]
        
        lat_true.append(t_lat); lon_true.append(t_lon)
        lat_pred.append(p_lat); lon_pred.append(p_lon)
    except: continue

lat_true, lon_true = np.array(lat_true), np.array(lon_true)
lat_pred, lon_pred = np.array(lat_pred), np.array(lon_pred)

dist = haversine(lat_true, lon_true, lat_pred, lon_pred)
acc_161 = np.mean(dist <= 161.0) * 100

print("\n" + "🔥"*25)
print(f"🇺🇸 CMU 모델 진짜 최종 성적표 (930 Buckets) 🇺🇸")
print(f"🎯 구역 적중률 (Class Acc)   : {acc_class:.2f} %")
print(f"🎯 거리 정확도 (Acc @ 161km) : {acc_161:.2f} %")
print(f"📏 Median Error (중간 오차)  : {np.median(dist):.2f} km")
print(f"📐 Mean Error (평균 오차)    : {np.mean(dist):.2f} km")
if error_count > 0:
    print(f"⚠️ 좌표 미상 구역(Skip)      : {error_count}건")
print("🔥"*25 + "\n")
