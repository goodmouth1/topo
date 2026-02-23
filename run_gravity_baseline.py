import pickle
import numpy as np
import os
import pandas as pd
import sys

# ==========================================
# ⚙️ 설정
# ==========================================
POSSIBLE_PATHS = [
    "goodmouth1/topo/topo-d2aec3f8b9de23a580ef0114868bce8d8ae91c3c/labeling/manual_labeled_data.pkl",
    "labeling/manual_labeled_data.pkl",
    "manual_labeled_data.pkl",
    "../labeling/manual_labeled_data.pkl"
]
INPUT_CSV = "gcn_detailed_evaluation_ALL.csv"

def haversine_np(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a)) * 6371
    return c

# 1. 데이터 로드
print("📦 데이터 로딩 중...")
labeled_file_path = None
for path in POSSIBLE_PATHS:
    if os.path.exists(path):
        labeled_file_path = path
        break

if labeled_file_path is None:
    print(f"❌ Error: 'manual_labeled_data.pkl' 파일을 찾을 수 없습니다.")
    sys.exit()

with open(labeled_file_path, 'rb') as f:
    labeled_data = pickle.load(f)

if not os.path.exists(INPUT_CSV):
    print(f"❌ Error: '{INPUT_CSV}' 파일이 없습니다.")
    sys.exit()

df = pd.read_csv(INPUT_CSV)
print(f"📄 분석 대상: {len(df)}건")

# 2. Gravity Model 계산 (Fixed)
print("\n🪐 [Fixed Baseline] Gravity Model (Log-Scale) 계산 중...")
print("   - 기존 문제: 인구수(100만)가 거리(10km)를 압도함")
print("   - 수정 공식: Score = log10(Population + 1000) / (Distance + 10)")
print("   - 효과: 인구수의 영향력을 줄이고, 거리의 영향력을 정상화함")

gravity_correct_list = []
gravity_answers = []

label_map = {}
for item in labeled_data:
    if item.get('toponyms'):
        key = (item['user_id'], item['toponyms'][0]['word'])
        label_map[key] = item

# 디버깅용 샘플 출력
print("\n🔎 [Debug] 첫 3개 데이터의 점수 계산 예시:")

for i, row in df.iterrows():
    uid = row['user_id']
    toponym = row['toponym_text']
    key = (uid, toponym)
    
    if key not in label_map:
        # Fallback
        found = False
        for item in labeled_data:
            if item['user_id'] == uid:
                item_cand = item
                found = True
                break
        if not found:
            gravity_correct_list.append(False)
            gravity_answers.append("N/A")
            continue
    else:
        item_cand = label_map[key]

    try:
        candidates = item_cand['toponyms'][0]['candidates']
        human_idx = item_cand['toponyms'][0]['human_label_idx']
        user_home = item_cand['user_true_loc']
    except:
        gravity_correct_list.append(False)
        gravity_answers.append("Error")
        continue
    
    # ---------------------------------------------------------
    # 🛠️ 수리된 Gravity 공식 (Log Scale)
    # ---------------------------------------------------------
    scores = []
    debug_info = []
    
    for cand in candidates:
        dist = haversine_np(*user_home, cand['lat'], cand['lon'])
        
        # 인구수 전처리 (문자열 등 예외 처리)
        try:
            raw_pop = cand.get('population', 0)
            if isinstance(raw_pop, str):
                raw_pop = float(raw_pop.replace(',', ''))
            pop = float(raw_pop or 0)
        except:
            pop = 0.0
            
        # 🔥 [FIX] Log 적용
        # 인구가 0이면 0점, 아니면 log10(pop)
        # +1000은 소도시(Pop=0 or small)가 너무 낮은 점수를 받지 않도록 하는 Base
        if pop > 0:
            log_pop = np.log10(pop + 1000) 
        else:
            log_pop = np.log10(1000) # 기본 점수 부여 (3.0)
            
        # 거리 + 10km (너무 가까운 거리에서 분모가 0에 가까워지는 것 방지)
        score = log_pop / (dist + 10.0)
        
        scores.append(score)
        
        if i < 3: # 디버깅용 저장
            debug_info.append(f"{cand['name']}: Pop {int(pop)}->Log {log_pop:.2f} / Dist {dist:.1f} = {score:.4f}")
    
    if i < 3:
        print(f"   User {uid} ('{toponym}'):")
        for info in debug_info[:3]: # 후보 3개만 출력
            print(f"     - {info}")
        print(f"     -> Best Candidate Index: {np.argmax(scores)}")

    if len(scores) > 0:
        best_idx = np.argmax(scores)
        is_correct = (best_idx == human_idx)
        ans_str = candidates[best_idx]['name'] + f" ({candidates[best_idx]['state']})"
    else:
        is_correct = False
        ans_str = "No Candidates"

    gravity_correct_list.append(is_correct)
    gravity_answers.append(ans_str)

df['gravity_correct'] = gravity_correct_list
df['gravity_answer'] = gravity_answers

OUTPUT_FILE = "gcn_detailed_evaluation_GRAVITY_FIXED.csv"
df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

# ==========================================
# 📊 최종 성적표 (Fixed)
# ==========================================
def get_acc(df, col):
    return df[col].mean() * 100 if len(df) > 0 else 0

travel_df = df[df['is_travel'] == True]
long_travel_df = df[df['dist_from_home'] >= 500]

print("\n" + "="*85)
print(f"🏆 [수리된 최종 성적표: GCN vs Gravity(Fixed) vs Distance]")
print("="*85)
print(f"{'모델 (Model)':<20} | {'전체 (Total)':<12} | {'여행 (Travel >300km)':<22} | {'장거리 (Long >500km)':<22}")
print("-" * 85)

gcn_long = get_acc(long_travel_df, 'gcn_correct')
grav_long = get_acc(long_travel_df, 'gravity_correct')
dist_long = get_acc(long_travel_df, 'dist_correct')

print(f"{'1. GCN (Ours)':<20} | {get_acc(df, 'gcn_correct'):.1f}%{'':<7} | {get_acc(travel_df, 'gcn_correct'):.1f}%{'':<17} | {gcn_long:.1f}%")
print(f"{'2. Gravity (Fixed)':<20} | {get_acc(df, 'gravity_correct'):.1f}%{'':<7} | {get_acc(travel_df, 'gravity_correct'):.1f}%{'':<17} | {grav_long:.1f}%")
print(f"{'3. Distance':<20} | {get_acc(df, 'dist_correct'):.1f}%{'':<7} | {get_acc(travel_df, 'dist_correct'):.1f}%{'':<17} | {dist_long:.1f}%")
print("="*85)

if gcn_long > grav_long:
    print(f"✅ GCN 승리! 정상적인 중력 모델을 상대로도 {gcn_long - grav_long:.1f}%p 차이로 이겼습니다.")
else:
    print(f"⚠️ Gravity가 매우 강력해졌습니다. (Distance와 비슷해졌을 것입니다)")
    print(f"   -> 하지만 여전히 '여행' 상황에서 GCN이 선방했는지 확인하세요.")