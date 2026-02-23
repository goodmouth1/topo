import sys
import pickle
import numpy as np
import gzip

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

pkl_file = sys.argv[1]

# 1. AI가 푼 시험지 로드
with open(pkl_file, 'rb') as f:
    preds = pickle.load(f)
    
y_pred = preds[0] if isinstance(preds, tuple) else preds
if len(np.shape(y_pred)) > 1:
    y_pred = np.argmax(y_pred, axis=1)

# 2. World 정답지 로드
with gzip.open('./data/world/world.pkl.gz', 'rb') as f:
    data = pickle.load(f, encoding='latin1')

lists = [x for x in data if isinstance(x, list)]
dicts = [x for x in data if isinstance(x, dict)]

U_test = [l for l in lists if len(l) == len(y_pred)][0]
classLatMedian = dicts[0]
classLonMedian = dicts[1]
userLocation = dicts[2]

# 3. 좌표 변환 (결측치 '.' 예외 처리 완벽 추가!)
lat_true, lon_true = [], []
lat_pred, lon_pred = [], []
error_count = 0

# 정답지와 시험지를 동시에 펼쳐놓고 한 줄씩 채점
for u, p in zip(U_test, y_pred):
    loc = userLocation[u]
    try:
        # 불량 데이터('.')가 섞여 있으면 여기서 에러가 나면서 except로 빠짐
        t_lat = float(loc[0])
        t_lon = float(loc[1])
        
        # 예측한 구역(Bucket) 좌표
        p_lat = float(classLatMedian.get(p, classLatMedian.get(str(p), 0.0)))
        p_lon = float(classLonMedian.get(p, classLonMedian.get(str(p), 0.0)))
        
        lat_true.append(t_lat)
        lon_true.append(t_lon)
        lat_pred.append(p_lat)
        lon_pred.append(p_lon)
    except ValueError:
        # 불량 데이터는 채점에서 아예 제외 (무효 처리)
        error_count += 1
        continue

lat_true = np.array(lat_true, dtype=float)
lon_true = np.array(lon_true, dtype=float)
lat_pred = np.array(lat_pred, dtype=float)
lon_pred = np.array(lon_pred, dtype=float)

# 4. 하버사인 거리 계산 및 최종 채점
dist = haversine(lat_true, lon_true, lat_pred, lon_pred)
acc = np.mean(dist <= 161.0) * 100
median_err = np.median(dist)
mean_err = np.mean(dist)

print("\n" + "🔥"*25)
print(f"🌍 World 모델 최종 성적표 🌍")
print(f"🎯 Accuracy @ 161km : {acc:.2f} %")
print(f"📏 Median Error     : {median_err:.2f} km")
print(f"📐 Mean Error       : {mean_err:.2f} km")
if error_count > 0:
    print(f"⚠️ 불량 데이터 제외 : {error_count}건 (채점 무효 처리)")
print("🔥"*25 + "\n")