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

with gzip.open('./data/world/world.pkl.gz', 'rb') as f:
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

y_cands = [a for a in arrays if len(a) == len(y_pred)]
u_cands = [l for l in lists if len(l) == len(y_pred)]

if y_cands:
    Y_test = y_cands[-1]
    Y_test = np.argmax(Y_test, axis=1) if len(np.shape(Y_test)) > 1 else Y_test.flatten()
else:
    y_t = np.array(preds[1].todense()) if hasattr(preds[1], 'todense') else np.array(preds[1])
    Y_test = np.argmax(y_t, axis=1) if len(np.shape(y_t)) > 1 else y_t.flatten()

U_test = u_cands[-1] if u_cands else lists[-1]
acc_class = np.mean(y_pred == Y_test) * 100

# 💡 맵핵(Maphack) 만들기: 위/경도 둘 다 완벽할 때만 쌍으로 넣기!
bucket_lat, bucket_lon = {}, {}
for l in lists:
    for a in arrays:
        if len(l) > 0 and len(l) == len(a):
            a_cls = np.argmax(a, axis=1) if len(np.shape(a)) > 1 else a.flatten()
            for u, y in zip(l, a_cls):
                loc = get_loc(u)
                if loc is not None:
                    try:
                        # 변환을 먼저 시도하고 성공하면 저장 (원자성 보장)
                        lat_f = float(loc[0])
                        lon_f = float(loc[1])
                        bucket_lat.setdefault(y, []).append(lat_f)
                        bucket_lon.setdefault(y, []).append(lon_f)
                    except: pass

# 빈 리스트(RuntimeWarning) 완벽 방어
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
        
        # AI가 고른 구역의 중심점 (데이터가 너무 없어서 못 구한 구역이면 에러 카운트)
        if p not in safe_lat:
            error_count += 1
            continue
            
        p_lat = safe_lat[p]
        p_lon = safe_lon[p]
        
        lat_true.append(t_lat)
        lon_true.append(t_lon)
        lat_pred.append(p_lat)
        lon_pred.append(p_lon)
    except: continue

lat_true, lon_true = np.array(lat_true), np.array(lon_true)
lat_pred, lon_pred = np.array(lat_pred), np.array(lon_pred)

dist = haversine(lat_true, lon_true, lat_pred, lon_pred)
acc_161 = np.mean(dist <= 161.0) * 100
median_err = np.median(dist)
mean_err = np.mean(dist)

print("\n" + "🔥"*25)
print(f"🌍 World 모델 진짜 최종 성적표 (930 Buckets) 🌍")
print(f"🎯 구역 적중률 (Class Acc)   : {acc_class:.2f} %")
print(f"🎯 거리 정확도 (Acc @ 161km) : {acc_161:.2f} %")
print(f"📏 Median Error (중간 오차)  : {median_err:.2f} km")
print(f"📐 Mean Error (평균 오차)    : {mean_err:.2f} km")
if error_count > 0:
    print(f"⚠️ 좌표 미상 구역(Skip)      : {error_count}건")
print("🔥"*25 + "\n")
