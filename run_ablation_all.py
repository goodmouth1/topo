"""
=============================================================
GCN 기여도 분리 실험 포함 전체 Ablation Study
=============================================================
기존 조건: Pop Only, Dist Only, Basic Gravity, Proposed, Oracle
추가 조건: 
  A) Text State + Pop (GCN 완전 제거)
  B) State Centroid + Gravity (GCN 대신 State 중심 좌표 사용)
=============================================================
사용법: python run_ablation_all.py [gcn_pkl_file]
"""

import sys, pickle, numpy as np, re
import pandas as pd
from collections import defaultdict

# ===== Haversine 거리 함수 =====
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

# ===== 미국 50개 주 약자 <-> 풀네임 매핑 =====
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

# ===== 데이터 로딩 =====
print("=" * 60)
print("  GCN 기여도 분리 포함 전체 Ablation Study")
print("=" * 60)

print("\n[1/4] GeoNames cities1000 로딩 중...")
cols = ['geonameid', 'name', 'asciiname', 'alternatenames', 'latitude', 'longitude', 
        'feature class', 'feature code', 'country code', 'cc2', 'admin1 code', 
        'admin2 code', 'admin3 code', 'admin4 code', 'population', 'elevation', 'dem', 'timezone', 'modification date']
geo_df = pd.read_csv('cities1000.txt', sep='\t', names=cols, low_memory=False)
us_cities = geo_df[geo_df['country code'] == 'US'].dropna(subset=['latitude', 'longitude', 'admin1 code'])

# 인구수 사전
pop_dict = {}
for _, row in us_cities.iterrows():
    key = (str(row['name']).strip(), str(row['admin1 code']).strip())
    pop = int(row['population'])
    if key not in pop_dict or pop_dict[key] < pop:
        pop_dict[key] = pop

# 역지오코딩용 좌표/주 배열
us_coords = us_cities[['latitude', 'longitude']].values
us_states_arr = us_cities['admin1 code'].values

def get_state_from_coords(lat, lon):
    """좌표 -> 가장 가까운 미국 도시의 주 반환"""
    diff = np.linalg.norm(us_coords - np.array([lat, lon]), axis=1)
    return str(us_states_arr[np.argmin(diff)]).strip()

# ===== State Centroid 계산 (조건 B용) =====
print("[2/4] State Centroid 계산 중...")
state_centroids = {}
for state_abbr in STATE_MAP.keys():
    state_cities = us_cities[us_cities['admin1 code'].str.strip() == state_abbr]
    if len(state_cities) > 0:
        # 인구 가중 중심 좌표
        pops = state_cities['population'].values.astype(float)
        total_pop = pops.sum()
        if total_pop > 0:
            w_lat = np.average(state_cities['latitude'].values, weights=pops)
            w_lon = np.average(state_cities['longitude'].values, weights=pops)
        else:
            w_lat = state_cities['latitude'].mean()
            w_lon = state_cities['longitude'].mean()
        state_centroids[state_abbr] = (w_lat, w_lon)
print(f"  -> {len(state_centroids)}개 주의 centroid 계산 완료")

# US 전체 중심 (fallback용)
US_CENTER = (39.8283, -98.5795)

# ===== 데이터 로딩 =====
print("[3/4] Manual labeled data 로딩 중...")
with open('/home/wang/master_thesis/labeling/manual_labeled_data.pkl', 'rb') as f:
    manual_data = pickle.load(f)

print("[4/4] GCN predictions 로딩 중...")
gcn_file = sys.argv[1] if len(sys.argv) > 1 else 'gcn_1.0_percent_pred_256.pkl'
with open(gcn_file, 'rb') as f:
    preds = pickle.load(f)
arr1, arr2 = np.array(preds[1]), np.array(preds[2])

print(f"  -> GCN 모델: {gcn_file}")
print(f"  -> 라벨링 데이터: {len(manual_data)}개 항목")

# ===== State 감지 함수 =====
def detect_states_from_text(text):
    """텍스트에서 미국 주 약자/풀네임 감지 -> set of state abbreviations"""
    detected = set()
    text_lower = text.lower()
    for abbr, fullname in STATE_MAP.items():
        # 약자: 대문자 독립 단어만 (IN, ME 같은 일반 단어 방지)
        if re.search(r'\b' + abbr + r'\b', text):
            detected.add(abbr)
        # 풀네임: 대소문자 무시
        if re.search(r'\b' + fullname.lower() + r'\b', text_lower):
            detected.add(abbr)
    return detected

def check_text_match(cand_state, text, text_lower):
    """후보의 주가 텍스트에 언급되었는지 확인"""
    state_full = STATE_MAP.get(cand_state, "").lower()
    match_abbr = bool(re.search(r'\b' + cand_state + r'\b', text))
    match_full = bool(re.search(r'\b' + state_full + r'\b', text_lower)) if state_full else False
    return match_abbr or match_full

# ===== 7가지 Ablation 조건 =====
CONDITIONS = [
    'pop_only',           # 1. 인구수만
    'dist_only',          # 2. GCN 거리만
    'text_state_pop',     # 3. [NEW] 텍스트 State + 인구수 (GCN 완전 제거)
    'state_centroid_grav', # 4. [NEW] State Centroid + Gravity (GCN 대신 State 중심)
    'basic_gravity',      # 5. GCN Gravity (State Bonus 없음)
    'proposed',           # 6. GCN Gravity + State Bonus (제안 모델)
    'oracle',             # 7. 실제 GPS + Gravity + State Bonus
]

results = {c: {'correct': 0, 'total': 0} for c in CONDITIONS}
# 케이스별 상세 기록 (오답 분석용)
detail_records = []

# ===== 메인 루프 =====
print("\n" + "=" * 60)
print("  실험 진행 중...")
print("=" * 60)

matched_users = 0

for item in manual_data:
    true_loc = item.get('user_true_loc')
    if not true_loc:
        continue
    
    # --- GCN 예측 매칭 ---
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
    gcn_lat, gcn_lon = pred_loc[0], pred_loc[1]
    gcn_state = get_state_from_coords(gcn_lat, gcn_lon)
    
    # 실제 GPS 위치
    true_lat, true_lon = true_loc[0], true_loc[1]
    true_state = get_state_from_coords(true_lat, true_lon)
    
    # 텍스트 정보
    text = item.get('orig_text', '')
    text_lower = text.lower()
    text_states = detect_states_from_text(text)
    
    # State Centroid 결정 (조건 B용)
    if text_states:
        # 텍스트에서 감지된 주 중 첫 번째의 centroid 사용
        centroid_state = list(text_states)[0]
        centroid_lat, centroid_lon = state_centroids.get(centroid_state, US_CENTER)
    else:
        # 텍스트에서 주를 감지 못하면 US 중심 사용
        centroid_lat, centroid_lon = US_CENTER
    centroid_detected_state = get_state_from_coords(centroid_lat, centroid_lon)
    
    # --- 각 토포님에 대해 모든 조건 평가 ---
    for top in item.get('toponyms', []):
        candidates = top.get('candidates', [])
        human_idx = top.get('human_label_idx', None)
        if human_idx is None:
            continue
        
        toponym_name = top.get('toponym', 'unknown')
        
        # 각 조건별 best 후보 추적
        best = {c: {'score': -1, 'idx': -1} for c in CONDITIONS}
        
        for idx_cand, cand in enumerate(candidates):
            name = cand.get('name')
            cand_state = cand.get('state')
            cand_lat, cand_lon = cand.get('lat'), cand.get('lon')
            pop = pop_dict.get((name, cand_state), 1000)
            
            # 텍스트 State 매칭
            text_match = check_text_match(cand_state, text, text_lower)
            
            # === 1. Pop Only ===
            score_pop = pop
            if score_pop > best['pop_only']['score']:
                best['pop_only'] = {'score': score_pop, 'idx': idx_cand}
            
            # === 2. Dist Only (GCN) ===
            dist_gcn = haversine(gcn_lat, gcn_lon, cand_lat, cand_lon)
            if dist_gcn < 1: dist_gcn = 1
            score_dist = 1.0 / (dist_gcn ** 1.5)
            if score_dist > best['dist_only']['score']:
                best['dist_only'] = {'score': score_dist, 'idx': idx_cand}
            
            # === 3. [NEW] Text State + Pop (GCN 완전 제거) ===
            # 텍스트에서 State가 감지되었고, 후보가 그 State에 있으면 인구수 * 100배 부스트
            # 감지 안 되었으면 순수 인구수만
            if text_states:
                if cand_state in text_states:
                    score_tsp = pop * 100  # State 매칭 후보 우선
                else:
                    score_tsp = pop * 0.01  # State 안 맞으면 거의 제외
            else:
                score_tsp = pop  # State 힌트 없으면 순수 인구수
            if score_tsp > best['text_state_pop']['score']:
                best['text_state_pop'] = {'score': score_tsp, 'idx': idx_cand}
            
            # === 4. [NEW] State Centroid + Gravity (GCN 대체) ===
            dist_centroid = haversine(centroid_lat, centroid_lon, cand_lat, cand_lon)
            if dist_centroid < 1: dist_centroid = 1
            
            gcn_match_centroid = (cand_state == centroid_detected_state)
            if text_match and gcn_match_centroid:
                bonus_centroid = 10.0
            elif text_match or gcn_match_centroid:
                bonus_centroid = 5.0
            else:
                bonus_centroid = 1.0
            score_scg = (pop / (dist_centroid ** 1.5)) * bonus_centroid
            if score_scg > best['state_centroid_grav']['score']:
                best['state_centroid_grav'] = {'score': score_scg, 'idx': idx_cand}
            
            # === 5. Basic Gravity (GCN, State Bonus 없음) ===
            score_bg = pop / (dist_gcn ** 1.5)
            if score_bg > best['basic_gravity']['score']:
                best['basic_gravity'] = {'score': score_bg, 'idx': idx_cand}
            
            # === 6. Proposed (GCN + Gravity + State Bonus) ===
            gcn_match = (cand_state == gcn_state)
            if text_match and gcn_match:
                bonus_proposed = 10.0
            elif text_match or gcn_match:
                bonus_proposed = 5.0
            else:
                bonus_proposed = 1.0
            score_prop = (pop / (dist_gcn ** 1.5)) * bonus_proposed
            if score_prop > best['proposed']['score']:
                best['proposed'] = {'score': score_prop, 'idx': idx_cand}
            
            # === 7. Oracle (GPS + Gravity + State Bonus) ===
            dist_true = haversine(true_lat, true_lon, cand_lat, cand_lon)
            if dist_true < 1: dist_true = 1
            true_match = (cand_state == true_state)
            if text_match and true_match:
                bonus_oracle = 10.0
            elif text_match or true_match:
                bonus_oracle = 5.0
            else:
                bonus_oracle = 1.0
            score_oracle = (pop / (dist_true ** 1.5)) * bonus_oracle
            if score_oracle > best['oracle']['score']:
                best['oracle'] = {'score': score_oracle, 'idx': idx_cand}
        
        # --- 결과 집계 ---
        record = {
            'toponym': toponym_name,
            'human_idx': human_idx,
            'text_states_detected': ','.join(text_states) if text_states else 'NONE',
        }
        
        for c in CONDITIONS:
            is_correct = (best[c]['idx'] == human_idx)
            results[c]['correct'] += int(is_correct)
            results[c]['total'] += 1
            record[f'{c}_pred'] = best[c]['idx']
            record[f'{c}_correct'] = is_correct
        
        detail_records.append(record)

# ===== 결과 출력 =====
print("\n" + "=" * 60)
print("  Ablation Study 결과")
print("=" * 60)
print(f"  매칭된 유저 수: {matched_users}명")
print(f"  총 토포님 수: {results['proposed']['total']}개")
print("=" * 60)

CONDITION_NAMES = {
    'pop_only':            '① Pop Only (Baseline)',
    'dist_only':           '② Dist Only (GCN)',
    'text_state_pop':      '③ Text State + Pop (No GCN)  ★NEW',
    'state_centroid_grav': '④ State Centroid + Gravity (No GCN)  ★NEW',
    'basic_gravity':       '⑤ Basic Gravity (GCN)',
    'proposed':            '⑥ Proposed (GCN+Gravity+State)',
    'oracle':              '⑦ Oracle (GPS+Gravity+State)',
}

print(f"\n{'모델':<50} {'정확도':>8}  {'정답/전체':>10}")
print("-" * 72)
for c in CONDITIONS:
    r = results[c]
    if r['total'] > 0:
        acc = r['correct'] / r['total'] * 100
        print(f"  {CONDITION_NAMES[c]:<48} {acc:>6.2f}%  ({r['correct']}/{r['total']})")

# ===== GCN 기여도 분석 =====
print("\n" + "=" * 60)
print("  GCN 기여도 분석")
print("=" * 60)

total = results['proposed']['total']
if total > 0:
    proposed_acc = results['proposed']['correct'] / total * 100
    tsp_acc = results['text_state_pop']['correct'] / total * 100
    scg_acc = results['state_centroid_grav']['correct'] / total * 100
    oracle_acc = results['oracle']['correct'] / total * 100
    pop_acc = results['pop_only']['correct'] / total * 100
    
    print(f"\n  Proposed({proposed_acc:.1f}%) vs Text State+Pop({tsp_acc:.1f}%)")
    print(f"  -> GCN + Gravity 기여: +{proposed_acc - tsp_acc:.1f}%p")
    
    print(f"\n  Proposed({proposed_acc:.1f}%) vs State Centroid+Gravity({scg_acc:.1f}%)")
    print(f"  -> GCN의 정밀 위치 예측 기여: +{proposed_acc - scg_acc:.1f}%p")
    
    print(f"\n  Proposed({proposed_acc:.1f}%) vs Oracle({oracle_acc:.1f}%)")
    print(f"  -> GPS 대비 GCN 성능 차이: {proposed_acc - oracle_acc:+.1f}%p")
    
    print(f"\n  State Centroid+Gravity({scg_acc:.1f}%) vs Pop Only({pop_acc:.1f}%)")
    print(f"  -> State 힌트만으로도 기여: +{scg_acc - pop_acc:.1f}%p")

# ===== 케이스별 교차 분석 =====
print("\n" + "=" * 60)
print("  교차 분석: GCN이 결정적으로 기여한 케이스")
print("=" * 60)

gcn_decisive = 0  # Text State+Pop은 틀렸지만 Proposed는 맞춘 케이스
gcn_redundant = 0  # 둘 다 맞춘 케이스 (GCN 없어도 됐음)
gcn_fail = 0  # 둘 다 틀린 케이스
gcn_hurt = 0  # Text State+Pop은 맞았지만 Proposed가 틀린 케이스

for rec in detail_records:
    tsp_ok = rec['text_state_pop_correct']
    prop_ok = rec['proposed_correct']
    if prop_ok and not tsp_ok:
        gcn_decisive += 1
    elif prop_ok and tsp_ok:
        gcn_redundant += 1
    elif not prop_ok and not tsp_ok:
        gcn_fail += 1
    elif not prop_ok and tsp_ok:
        gcn_hurt += 1

print(f"  GCN 결정적 기여 (TSP❌ → Proposed✅): {gcn_decisive}건")
print(f"  GCN 없어도 정답 (TSP✅ → Proposed✅): {gcn_redundant}건")
print(f"  둘 다 실패 (TSP❌ → Proposed❌):       {gcn_fail}건")
print(f"  GCN이 오히려 해침 (TSP✅ → Proposed❌): {gcn_hurt}건")

# ===== CSV 저장 =====
df_detail = pd.DataFrame(detail_records)
csv_file = 'ablation_gcn_contribution.csv'
df_detail.to_csv(csv_file, index=False)
print(f"\n  상세 결과 저장: {csv_file}")

print("\n" + "=" * 60)
print("  실험 완료!")
print("=" * 60)
