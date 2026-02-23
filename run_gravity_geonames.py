import pickle
import pandas as pd

print("🌍 GeoNames 데이터 로딩 중...")
# GeoNames 데이터는 tab으로 구분되어 있습니다. 필요한 열(이름, 주, 인구수, 국가)만 가져옵니다.
cols = ['geonameid', 'name', 'asciiname', 'alternatenames', 'latitude', 'longitude', 
        'feature class', 'feature code', 'country code', 'cc2', 'admin1 code', 
        'admin2 code', 'admin3 code', 'admin4 code', 'population', 'elevation', 'dem', 'timezone', 'modification date']

geo_df = pd.read_csv('cities1000.txt', sep='\t', names=cols, low_memory=False)

# 미국(US) 데이터만 필터링하고, 주(State) 코드는 'admin1 code'에 들어있습니다.
us_cities = geo_df[geo_df['country code'] == 'US']

# (City, State) 형태의 인구수 사전(Dictionary)을 자동으로 생성합니다!
pop_dict = {}
for _, row in us_cities.iterrows():
    city_name = str(row['name']).strip()
    state_code = str(row['admin1 code']).strip() # 미국은 여기가 State 약자(TX, CA 등)입니다.
    population = int(row['population'])
    
    # 동명이인이 있을 수 있으므로(예: 텍사스 내의 여러 동네), 가장 인구수가 많은 곳을 저장
    key = (city_name, state_code)
    if key not in pop_dict or pop_dict[key] < population:
        pop_dict[key] = population

print(f"✅ 미국 내 {len(pop_dict)}개 도시의 인구수 사전 구축 완료!\n")

# --- 여기서부터는 아까와 동일한 Gravity Model ---
with open('/home/wang/master_thesis/labeling/manual_labeled_data.pkl', 'rb') as f:
    data = pickle.load(f)

correct = 0
total = 0

print("🔍 GeoNames 기반 Gravity Model 채점 시작...")

for item in data:
    text = item.get('orig_text', '')
    for top in item.get('toponyms', []):
        candidates = top.get('candidates', [])
        human_idx = top.get('human_label_idx', None)
        
        if human_idx is None: continue
        
        best_score = -1
        best_idx = -1
        
        for idx, cand in enumerate(candidates):
            name = cand.get('name')
            state = cand.get('state')
            
            # (1) Distance (거리)
            dist = cand.get('dist_from_user', 10000)
            if dist < 1: dist = 1 
            
            # (2) Population (인구수): GeoNames 사전에서 가져옵니다! (못 찾으면 소도시 1000명 기준)
            pop = pop_dict.get((name, state), 1000)
            
            # (3) SameState Bonus: 우리가 설계한 가장 강력한 킥!
            bonus = 5.0 if state in text else 1.0
            
            # 🌟 Gravity Model 공식 🌟
            score = (pop / (dist ** 1.5)) * bonus
            
            if score > best_score:
                best_score = score
                best_idx = idx
                
        if best_idx == human_idx:
            correct += 1
        total += 1

acc = (correct / total) * 100
print("="*50)
print(f"🏆 GeoNames + Gravity Model 적중률: {acc:.2f}% ({correct}/{total})")
print("="*50)
