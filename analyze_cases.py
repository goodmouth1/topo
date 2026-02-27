"""
GCN 기여 케이스 상세 분석
- GCN이 결정적으로 기여한 16건
- GCN이 오히려 해친 7건
- 둘 다 실패한 11건
"""
import sys, pickle, numpy as np, re
import pandas as pd

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

STATE_MAP = {
    'AL':'Alabama','AK':'Alaska','AZ':'Arizona','AR':'Arkansas','CA':'California',
    'CO':'Colorado','CT':'Connecticut','DE':'Delaware','FL':'Florida','GA':'Georgia',
    'HI':'Hawaii','ID':'Idaho','IL':'Illinois','IN':'Indiana','IA':'Iowa',
    'KS':'Kansas','KY':'Kentucky','LA':'Louisiana','ME':'Maine','MD':'Maryland',
    'MA':'Massachusetts','MI':'Michigan','MN':'Minnesota','MS':'Mississippi','MO':'Missouri',
    'MT':'Montana','NE':'Nebraska','NV':'Nevada','NH':'New Hampshire','NJ':'New Jersey',
    'NM':'New Mexico','NY':'New York','NC':'North Carolina','ND':'North Dakota','OH':'Ohio',
    'OK':'Oklahoma','OR':'Oregon','PA':'Pennsylvania','RI':'Rhode Island','SC':'South Carolina',
    'SD':'South Dakota','TN':'Tennessee','TX':'Texas','UT':'Utah','VT':'Vermont',
    'VA':'Virginia','WA':'Washington','WV':'West Virginia','WI':'Wisconsin','WY':'Wyoming'
}

# 데이터 로딩
cols = ['geonameid','name','asciiname','alternatenames','latitude','longitude',
        'feature class','feature code','country code','cc2','admin1 code',
        'admin2 code','admin3 code','admin4 code','population','elevation','dem','timezone','modification date']
geo_df = pd.read_csv('cities1000.txt', sep='\t', names=cols, low_memory=False)
us_cities = geo_df[geo_df['country code']=='US'].dropna(subset=['latitude','longitude','admin1 code'])

pop_dict = {}
for _, row in us_cities.iterrows():
    key = (str(row['name']).strip(), str(row['admin1 code']).strip())
    pop = int(row['population'])
    if key not in pop_dict or pop_dict[key] < pop:
        pop_dict[key] = pop

us_coords = us_cities[['latitude','longitude']].values
us_states_arr = us_cities['admin1 code'].values

def get_state(lat, lon):
    diff = np.linalg.norm(us_coords - np.array([lat, lon]), axis=1)
    return str(us_states_arr[np.argmin(diff)]).strip()

with open('/home/wang/master_thesis/labeling/manual_labeled_data.pkl','rb') as f:
    manual_data = pickle.load(f)

gcn_file = sys.argv[1] if len(sys.argv) > 1 else 'gcn_1.0_percent_pred_256.pkl'
with open(gcn_file,'rb') as f:
    preds = pickle.load(f)
arr1, arr2 = np.array(preds[1]), np.array(preds[2])

# 메인 분석
case_id = 0
decisive_cases = []  # GCN 결정적
hurt_cases = []      # GCN 해침
both_fail_cases = [] # 둘 다 실패

for item in manual_data:
    true_loc = item.get('user_true_loc')
    if not true_loc: continue
    
    diff1 = np.linalg.norm(arr1 - true_loc, axis=1)
    diff2 = np.linalg.norm(arr2 - true_loc, axis=1)
    min1, min2 = np.min(diff1), np.min(diff2)
    
    if min1 < min2 and min1 < 0.01: pred_loc = arr2[np.argmin(diff1)]
    elif min2 < min1 and min2 < 0.01: pred_loc = arr1[np.argmin(diff2)]
    else: continue
    
    gcn_lat, gcn_lon = pred_loc[0], pred_loc[1]
    gcn_state = get_state(gcn_lat, gcn_lon)
    true_lat, true_lon = true_loc[0], true_loc[1]
    true_state = get_state(true_lat, true_lon)
    
    text = item.get('orig_text', '')
    
    for top in item.get('toponyms', []):
        candidates = top.get('candidates', [])
        human_idx = top.get('human_label_idx', None)
        if human_idx is None: continue
        
        # Pop Only (TSP와 동일하게 text_states=NONE일 때)
        pop_scores = []
        grav_scores = []
        for idx_c, cand in enumerate(candidates):
            name = cand.get('name')
            cand_state = cand.get('state')
            cand_lat, cand_lon = cand.get('lat'), cand.get('lon')
            pop = pop_dict.get((name, cand_state), 1000)
            dist = haversine(gcn_lat, gcn_lon, cand_lat, cand_lon)
            if dist < 1: dist = 1
            pop_scores.append(pop)
            grav_scores.append(pop / (dist ** 1.5))
        
        tsp_pred = int(np.argmax(pop_scores))
        proposed_pred = int(np.argmax(grav_scores))
        
        tsp_correct = (tsp_pred == human_idx)
        proposed_correct = (proposed_pred == human_idx)
        
        # 정답 후보 정보
        correct_cand = candidates[human_idx]
        correct_name = correct_cand.get('name', '?')
        correct_state = correct_cand.get('state', '?')
        correct_pop = pop_dict.get((correct_name, correct_state), 1000)
        correct_dist = haversine(gcn_lat, gcn_lon, correct_cand.get('lat',0), correct_cand.get('lon',0))
        
        case_info = {
            'case_id': case_id,
            'toponym_surface': top.get('toponym', top.get('surface_form', '?')),
            'num_candidates': len(candidates),
            'correct_answer': f"{correct_name}, {correct_state}",
            'correct_pop': correct_pop,
            'correct_dist_from_gcn': f"{correct_dist:.0f}km",
            'gcn_predicted_state': gcn_state,
            'true_state': true_state,
            'gcn_state_correct': gcn_state == true_state,
            'candidates_summary': [],
            'text_snippet': text[:150].replace('\n',' '),
        }
        
        for idx_c, cand in enumerate(candidates):
            cname = cand.get('name','?')
            cstate = cand.get('state','?')
            cpop = pop_dict.get((cname, cstate), 1000)
            cdist = haversine(gcn_lat, gcn_lon, cand.get('lat',0), cand.get('lon',0))
            marker = ""
            if idx_c == human_idx: marker += " ★정답"
            if idx_c == tsp_pred and not tsp_correct: marker += " ←TSP선택(오답)"
            if idx_c == proposed_pred and not proposed_correct: marker += " ←Proposed선택(오답)"
            if idx_c == proposed_pred and proposed_correct: marker += " ←Proposed선택(정답)"
            if idx_c == tsp_pred and tsp_correct and not proposed_correct: marker += " ←TSP선택(정답)"
            case_info['candidates_summary'].append(
                f"  [{idx_c}] {cname}, {cstate} | pop={cpop:,} | dist={cdist:.0f}km | grav={cpop/(max(cdist,1)**1.5):.2f}{marker}"
            )
        
        if proposed_correct and not tsp_correct:
            decisive_cases.append(case_info)
        elif tsp_correct and not proposed_correct:
            hurt_cases.append(case_info)
        elif not tsp_correct and not proposed_correct:
            both_fail_cases.append(case_info)
        
        case_id += 1

def print_cases(title, cases):
    print(f"\n{'='*80}")
    print(f"  {title} ({len(cases)}건)")
    print('='*80)
    for c in cases:
        print(f"\n--- Case #{c['case_id']}: \"{c['toponym_surface']}\" ({c['num_candidates']}개 후보) ---")
        print(f"  정답: {c['correct_answer']} (pop={c['correct_pop']:,}, GCN거리={c['correct_dist_from_gcn']})")
        print(f"  GCN 예측 주: {c['gcn_predicted_state']} | 실제 주: {c['true_state']} | 주 예측 정확: {c['gcn_state_correct']}")
        print(f"  텍스트: \"{c['text_snippet']}...\"")
        print(f"  후보 목록:")
        for cs in c['candidates_summary']:
            print(f"    {cs}")

print_cases("GCN 결정적 기여 (Pop Only는 틀리고, GCN Gravity는 맞춤)", decisive_cases)
print_cases("GCN이 오히려 해침 (Pop Only는 맞고, GCN Gravity가 틀림)", hurt_cases)
print_cases("둘 다 실패", both_fail_cases)

# 패턴 요약
print(f"\n{'='*80}")
print("  패턴 요약")
print('='*80)

print("\n[GCN 결정적 기여 패턴]")
for c in decisive_cases:
    print(f"  {c['toponym_surface']}: 정답={c['correct_answer']}, "
          f"GCN주={c['gcn_predicted_state']}, 실제주={c['true_state']}, "
          f"GCN주정확={c['gcn_state_correct']}")

print("\n[GCN 해침 패턴]")
for c in hurt_cases:
    print(f"  {c['toponym_surface']}: 정답={c['correct_answer']}, "
          f"GCN주={c['gcn_predicted_state']}, 실제주={c['true_state']}, "
          f"GCN주정확={c['gcn_state_correct']}")
