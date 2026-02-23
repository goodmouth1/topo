import sys, pickle, numpy as np, re, csv
import pandas as pd

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

STATE_MAP = {
    'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California',
    'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia',
    'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa',
    'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
    'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi', 'MO': 'Missouri',
    'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey',
    'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio',
    'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
    'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont',
    'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming'
}

print("🌍 GeoNames 및 역지오코딩 엔진, 데이터 로딩 중...")
cols = ['geonameid', 'name', 'asciiname', 'alternatenames', 'latitude', 'longitude', 
        'feature class', 'feature code', 'country code', 'cc2', 'admin1 code', 
        'admin2 code', 'admin3 code', 'admin4 code', 'population', 'elevation', 'dem', 'timezone', 'modification date']
geo_df = pd.read_csv('cities1000.txt', sep='\t', names=cols, low_memory=False)
us_cities = geo_df[geo_df['country code'] == 'US'].dropna(subset=['latitude', 'longitude', 'admin1 code'])

pop_dict = {}
for _, row in us_cities.iterrows():
    key = (str(row['name']).strip(), str(row['admin1 code']).strip())
    pop = int(row['population'])
    if key not in pop_dict or pop_dict[key] < pop:
        pop_dict[key] = pop

us_coords = us_cities[['latitude', 'longitude']].values
us_states = us_cities['admin1 code'].values

def get_state_from_gcn(lat, lon):
    diff = np.linalg.norm(us_coords - np.array([lat, lon]), axis=1)
    return str(us_states[np.argmin(diff)]).strip()

with open('/home/wang/master_thesis/labeling/manual_labeled_data.pkl', 'rb') as f:
    manual_data = pickle.load(f)

gcn_file = sys.argv[1] if len(sys.argv) > 1 else 'gcn_1.0_percent_pred_256.pkl'
with open(gcn_file, 'rb') as f:
    preds = pickle.load(f)

arr1, arr2 = np.array(preds[1]), np.array(preds[2])

# CSV에 담을 데이터를 모으는 리스트
csv_rows = []

c_pop, c_dist, c_grav, c_final, total = 0, 0, 0, 0, 0

print(f"\n🚀 [Ablation Study] 4개 모델 동시 채점 및 CSV 추출 시작! (모델: {gcn_file})")

for item in manual_data:
    true_loc = item.get('user_true_loc')
    if not true_loc: continue
    
    diff1 = np.linalg.norm(arr1 - true_loc, axis=1)
    diff2 = np.linalg.norm(arr2 - true_loc, axis=1)
    min1, min2 = np.min(diff1), np.min(diff2)
    
    if min1 < min2 and min1 < 0.01: pred_loc = arr2[np.argmin(diff1)] 
    elif min2 < min1 and min2 < 0.01: pred_loc = arr1[np.argmin(diff2)]
    else: continue 
        
    pred_lat, pred_lon = pred_loc[0], pred_loc[1]
    gcn_pred_state = get_state_from_gcn(pred_lat, pred_lon)
    
    text = item.get('orig_text', '')
    text_lower = text.lower()
    user_id = item.get('user_id', 'Unknown')
    
    for top in item.get('toponyms', []):
        word = top.get('word', 'Unknown')
        candidates = top.get('candidates', [])
        human_idx = top.get('human_label_idx', None)
        if human_idx is None: continue
        
        s_pop, s_dist, s_grav, s_final = -1, -1, -1, -1
        i_pop, i_dist, i_grav, i_final = -1, -1, -1, -1
        
        for idx_cand, cand in enumerate(candidates):
            name, cand_state = cand.get('name'), cand.get('state')
            cand_lat, cand_lon = cand.get('lat'), cand.get('lon')
            
            dist = haversine(pred_lat, pred_lon, cand_lat, cand_lon)
            if dist < 1: dist = 1
            pop = pop_dict.get((name, cand_state), 1000)
            
            state_full = STATE_MAP.get(cand_state, "").lower()
            match_abbr = bool(re.search(r'\b' + cand_state + r'\b', text))
            match_full = bool(re.search(r'\b' + state_full + r'\b', text_lower)) if state_full else False
            text_match = match_abbr or match_full
            gcn_match = (cand_state == gcn_pred_state)
            
            bonus = 10.0 if (text_match and gcn_match) else (5.0 if text_match or gcn_match else 1.0)
            
            # 1. 인구수 100%
            if pop > s_pop: s_pop = pop; i_pop = idx_cand
            # 2. GCN 거리 100%
            if (1/dist) > s_dist: s_dist = (1/dist); i_dist = idx_cand
            # 3. 기본 중력 (인구수/거리)
            if (pop / (dist**1.5)) > s_grav: s_grav = (pop / (dist**1.5)); i_grav = idx_cand
            # 4. 최종 모델 제안 (중력 * 보너스)
            if ((pop / (dist**1.5)) * bonus) > s_final: s_final = ((pop / (dist**1.5)) * bonus); i_final = idx_cand
                
        # 정답 확인
        gt_cand = candidates[human_idx]
        gt_name = f"{gt_cand.get('name')}, {gt_cand.get('state')}"
        
        def get_cand_name(idx):
            if idx == -1: return "None"
            c = candidates[idx]
            return f"{c.get('name')}, {c.get('state')}"
            
        is_pop_correct = (i_pop == human_idx)
        is_dist_correct = (i_dist == human_idx)
        is_grav_correct = (i_grav == human_idx)
        is_final_correct = (i_final == human_idx)
        
        if is_pop_correct: c_pop += 1
        if is_dist_correct: c_dist += 1
        if is_grav_correct: c_grav += 1
        if is_final_correct: c_final += 1
        total += 1
        
        # CSV 행 추가
        csv_rows.append({
            'User_ID': user_id,
            'Toponym': word,
            'Ground_Truth (정답)': gt_name,
            '1. Pop_Only (선택)': get_cand_name(i_pop),
            '1. Pop_Only (결과)': 'O' if is_pop_correct else 'X',
            '2. Dist_Only (선택)': get_cand_name(i_dist),
            '2. Dist_Only (결과)': 'O' if is_dist_correct else 'X',
            '3. Basic_Gravity (선택)': get_cand_name(i_grav),
            '3. Basic_Gravity (결과)': 'O' if is_grav_correct else 'X',
            '4. Proposed_Model (선택)': get_cand_name(i_final),
            '4. Proposed_Model (결과)': 'O' if is_final_correct else 'X',
            'Text_Snippet': text.replace('\n', ' ')[:100] + '...' # 텍스트가 너무 길면 잘라서 표기
        })

# CSV 파일 저장
df = pd.DataFrame(csv_rows)
csv_filename = 'ablation_results.csv'
df.to_csv(csv_filename, index=False, encoding='utf-8-sig') # 엑셀에서 한글 안 깨지게 utf-8-sig 적용

print("\n📊 [논문 표 추출 완료] ��")
print("="*55)
print(f"1️⃣ Population Only (인구수 100%) : {(c_pop/total)*100:.2f}% ({c_pop}/{total})")
print(f"2️⃣ Distance Only (GCN 거리 100%) : {(c_dist/total)*100:.2f}% ({c_dist}/{total})")
print(f"3️⃣ Basic Gravity (GCN+인구수)    : {(c_grav/total)*100:.2f}% ({c_grav}/{total})")
print(f"4️⃣ Proposed Model (+State Bonus) : {(c_final/total)*100:.2f}% ({c_final}/{total})")
print("="*55)
print(f"✅ 상세 오답 분석 파일이 생성되었습니다: {csv_filename}")
