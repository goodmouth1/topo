import sys, pickle, numpy as np
import pandas as pd

# Haversine 거리 계산 함수
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

print("🌍 GeoNames 인구수 사전 로딩 중...")
cols = ['geonameid', 'name', 'asciiname', 'alternatenames', 'latitude', 'longitude', 
        'feature class', 'feature code', 'country code', 'cc2', 'admin1 code', 
        'admin2 code', 'admin3 code', 'admin4 code', 'population', 'elevation', 'dem', 'timezone', 'modification date']
try:
    geo_df = pd.read_csv('cities1000.txt', sep='\t', names=cols, low_memory=False)
    us_cities = geo_df[geo_df['country code'] == 'US']
    pop_dict = {}
    for _, row in us_cities.iterrows():
        key = (str(row['name']).strip(), str(row['admin1 code']).strip())
        pop = int(row['population'])
        if key not in pop_dict or pop_dict[key] < pop:
            pop_dict[key] = pop
except:
    print("⚠️ cities1000.txt 오류! (기본값으로 진행)")
    pop_dict = {}

# 1. 수동 데이터 로드
with open('/home/wang/master_thesis/labeling/manual_labeled_data.pkl', 'rb') as f:
    manual_data = pickle.load(f)

# 2. GCN 파일 로드 (터미널에서 입력받음)
gcn_file = sys.argv[1] if len(sys.argv) > 1 else 'gcn_1.0_percent_pred_930.pkl'
with open(gcn_file, 'rb') as f:
    preds = pickle.load(f)

arr1 = np.array(preds[1])
arr2 = np.array(preds[2])

correct = 0
total = 0
matched_users = 0

print(f"\n🚀 [본게임] GCN 좌표 + Gravity Model 채점 시작! (모델: {gcn_file})")

for item in manual_data:
    true_loc = item.get('user_true_loc')
    if not true_loc: continue
    
    # 3. 지문 매칭: 수동 데이터의 진짜 GPS로 GCN 예측 결과 찾기!
    diff1 = np.linalg.norm(arr1 - true_loc, axis=1)
    diff2 = np.linalg.norm(arr2 - true_loc, axis=1)
    
    min1, min2 = np.min(diff1), np.min(diff2)
    
    # 유격 허용 오차(0.01) 내에서 똑같은 좌표 찾기
    if min1 < min2 and min1 < 0.01:
        idx = np.argmin(diff1)
        pred_loc = arr2[idx] # arr1이 진짜면 arr2가 GCN 예측값!
    elif min2 < min1 and min2 < 0.01:
        idx = np.argmin(diff2)
        pred_loc = arr1[idx]
    else:
        continue # 이 유저는 현재 GCN 파일(예: CMU)에 없는 유저임
        
    matched_users += 1
    pred_lat, pred_lon = pred_loc[0], pred_loc[1]
    text = item.get('orig_text', '')
    
    for top in item.get('toponyms', []):
        candidates = top.get('candidates', [])
        human_idx = top.get('human_label_idx', None)
        if human_idx is None: continue
        
        best_score = -1
        best_idx = -1
        
        for idx_cand, cand in enumerate(candidates):
            name, state = cand.get('name'), cand.get('state')
            cand_lat, cand_lon = cand.get('lat'), cand.get('lon')
            
            # 🔥 핵심: 진짜 GPS가 아니라, 'GCN이 예측한 위치'와의 거리를 잰다! 🔥
            dist = haversine(pred_lat, pred_lon, cand_lat, cand_lon)
            if dist < 1: dist = 1
            
            pop = pop_dict.get((name, state), 1000)
            bonus = 5.0 if state in text else 1.0
            
            # 대망의 Gravity Model
            score = (pop / (dist ** 1.5)) * bonus
            
            if score > best_score:
                best_score = score
                best_idx = idx_cand
                
        if best_idx == human_idx:
            correct += 1
        total += 1

if total > 0:
    acc = (correct / total) * 100
    print("="*55)
    print(f"🔗 매칭된 유저 수: {matched_users}명 (수동 데이터 ∩ GCN 데이터)")
    print(f"🏆 최종 논문용 GCN 적중률: {acc:.2f}% ({correct}/{total})")
    print("="*55)
else:
    print("⚠️ 매칭된 유저가 없습니다! 다른 모델(256.pkl 또는 930.pkl)을 넣어보세요.")
