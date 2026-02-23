import sys, pickle, numpy as np
import pandas as pd

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

print("🌍 GeoNames 로딩 및 역지오코딩(Reverse Geocoding) 엔진 구축 중...")
cols = ['geonameid', 'name', 'asciiname', 'alternatenames', 'latitude', 'longitude', 
        'feature class', 'feature code', 'country code', 'cc2', 'admin1 code', 
        'admin2 code', 'admin3 code', 'admin4 code', 'population', 'elevation', 'dem', 'timezone', 'modification date']

geo_df = pd.read_csv('cities1000.txt', sep='\t', names=cols, low_memory=False)
us_cities = geo_df[geo_df['country code'] == 'US'].dropna(subset=['latitude', 'longitude', 'admin1 code'])

# 1. 인구수 사전 구축
pop_dict = {}
for _, row in us_cities.iterrows():
    key = (str(row['name']).strip(), str(row['admin1 code']).strip())
    pop = int(row['population'])
    if key not in pop_dict or pop_dict[key] < pop:
        pop_dict[key] = pop

# 2. GCN 좌표 -> State 변환을 위한 미국 도시 좌표 배열 구축
us_coords = us_cities[['latitude', 'longitude']].values
us_states = us_cities['admin1 code'].values

def get_state_from_gcn(lat, lon):
    # GCN 예측 좌표와 가장 가까운 미국 도시를 찾아서 그 도시의 State 반환!
    diff = np.linalg.norm(us_coords - np.array([lat, lon]), axis=1)
    nearest_idx = np.argmin(diff)
    return str(us_states[nearest_idx]).strip()

with open('/home/wang/master_thesis/labeling/manual_labeled_data.pkl', 'rb') as f:
    manual_data = pickle.load(f)

gcn_file = sys.argv[1] if len(sys.argv) > 1 else 'gcn_1.0_percent_pred_256.pkl'
with open(gcn_file, 'rb') as f:
    preds = pickle.load(f)

arr1, arr2 = np.array(preds[1]), np.array(preds[2])

correct, total, matched_users = 0, 0, 0

print(f"\n🚀 [State 보너스 켜짐] GCN Gravity Model 채점 시작! (모델: {gcn_file})")

for item in manual_data:
    true_loc = item.get('user_true_loc')
    if not true_loc: continue
    
    diff1 = np.linalg.norm(arr1 - true_loc, axis=1)
    diff2 = np.linalg.norm(arr2 - true_loc, axis=1)
    min1, min2 = np.min(diff1), np.min(diff2)
    
    if min1 < min2 and min1 < 0.01:
        pred_loc = arr2[np.argmin(diff1)] 
    elif min2 < min1 and min2 < 0.01:
        pred_loc = arr1[np.argmin(diff2)]
    else:
        continue 
        
    matched_users += 1
    pred_lat, pred_lon = pred_loc[0], pred_loc[1]
    
    # 🌟 드디어 추가된 호날두 님의 핵심 아이디어! GCN이 예측한 State 추출 🌟
    gcn_predicted_state = get_state_from_gcn(pred_lat, pred_lon)
    
    text = item.get('orig_text', '')
    
    for top in item.get('toponyms', []):
        candidates = top.get('candidates', [])
        human_idx = top.get('human_label_idx', None)
        if human_idx is None: continue
        
        best_score, best_idx = -1, -1
        
        for idx_cand, cand in enumerate(candidates):
            name, cand_state = cand.get('name'), cand.get('state')
            cand_lat, cand_lon = cand.get('lat'), cand.get('lon')
            
            dist = haversine(pred_lat, pred_lon, cand_lat, cand_lon)
            if dist < 1: dist = 1
            
            pop = pop_dict.get((name, cand_state), 1000)
            
            # 🔥 완벽하게 구현된 3단계 하이브리드 보너스 🔥
            text_match = cand_state in text
            gcn_match = (cand_state == gcn_predicted_state)
            
            if text_match and gcn_match:
                bonus = 10.0  # 텍스트와 AI가 동시에 가리킴 (확실함)
            elif text_match or gcn_match:
                bonus = 5.0   # 둘 중 하나라도 증거가 있음
            else:
                bonus = 1.0   # 증거 없음
            
            score = (pop / (dist ** 1.5)) * bonus
            
            if score > best_score:
                best_score = score
                best_idx = idx_cand
                
        if best_idx == human_idx:
            correct += 1
        total += 1

if total > 0:
    acc = (correct / total) * 100
    print("="*60)
    print(f"🔗 매칭된 유저 수: {matched_users}명")
    print(f"🏆 [최종] State 보너스 탑재 GCN 적중률: {acc:.2f}% ({correct}/{total})")
    print("="*60)
