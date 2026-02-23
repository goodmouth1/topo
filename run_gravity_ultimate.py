import sys, pickle, numpy as np, re
import pandas as pd

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

# 🌟 미국 50개 주 약자 <-> 풀네임 매핑 사전 탑재 🌟
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

print("🌍 GeoNames 및 역지오코딩 엔진, State Dictionary 로딩 중...")
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
correct, total, matched_users = 0, 0, 0

print(f"\n🚀 [궁극의 마스터 코드] 풀네임 Regex + 곱셈 가중치! (모델: {gcn_file})")

for item in manual_data:
    true_loc = item.get('user_true_loc')
    if not true_loc: continue
    
    diff1 = np.linalg.norm(arr1 - true_loc, axis=1)
    diff2 = np.linalg.norm(arr2 - true_loc, axis=1)
    min1, min2 = np.min(diff1), np.min(diff2)
    
    if min1 < min2 and min1 < 0.01: pred_loc = arr2[np.argmin(diff1)] 
    elif min2 < min1 and min2 < 0.01: pred_loc = arr1[np.argmin(diff2)]
    else: continue 
        
    matched_users += 1
    pred_lat, pred_lon = pred_loc[0], pred_loc[1]
    gcn_predicted_state = get_state_from_gcn(pred_lat, pred_lon)
    
    text = item.get('orig_text', '')
    text_lower = text.lower()
    
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
            
            # 🔥 1. 정확한 State 매칭 (약자 대문자 OR 풀네임 대소문자 무시) 🔥
            state_full = STATE_MAP.get(cand_state, "").lower()
            
            # 약자는 대문자 독립 단어로만! (in, me 방지)
            match_abbr = bool(re.search(r'\b' + cand_state + r'\b', text))
            # 풀네임은 대소문자 상관없이! (Texas, texas, CALIFORNIA 모두 인정)
            match_full = bool(re.search(r'\b' + state_full + r'\b', text_lower)) if state_full else False
            
            text_match = match_abbr or match_full
            gcn_match = (cand_state == gcn_predicted_state)
            
            # 🔥 2. 중력 모델의 본질: 곱셈(Multiplicative) 보너스로 회귀 🔥
            if text_match and gcn_match:
                bonus = 10.0
            elif text_match or gcn_match:
                bonus = 5.0
            else:
                bonus = 1.0
                
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
    print(f"🏆 [Ultimate] 풀네임 매칭 + 가중치 GCN 적중률: {acc:.2f}% ({correct}/{total})")
    print("="*60)
