#!/usr/bin/env python
"""
Step 4: 지명 모호성 해소용 평가 데이터 구축

Yelp 비즈니스의 city 필드를 toponym으로,
실제 lat/lon을 ground truth로 사용하여 평가 데이터를 자동 생성.

핵심 아이디어:
- business.city = "Portland" -> 모호한 지명
- state 정보를 제거하면 Portland, OR인지 Portland, ME인지 모름
- GeoNames에서 "Portland" 검색 -> 후보 리스트
- Gravity Model로 어떤 Portland인지 예측
- business.lat/lon과 비교하여 정답 확인
"""

import json, pickle, gzip, os
import pandas as pd
from collections import defaultdict

DATA_DIR = '.'  # Yelp JSON 파일들이 있는 디렉토리
GEOTEXT_DIR = 'yelp_geotext_format'

# cities1000.txt 경로 (부모 디렉토리에 있음)
CITIES_FILE = os.path.join(os.path.dirname(__file__), '..', 'cities1000.txt')
if not os.path.exists(CITIES_FILE):
    CITIES_FILE = 'cities1000.txt'  # 현재 디렉토리에 있을 수도 있음

# --- 1. GeoNames 로드 ---
print("[Step 4-1] GeoNames 로딩 중...")
cols = ['geonameid', 'name', 'asciiname', 'alternatenames',
        'latitude', 'longitude', 'feature class', 'feature code',
        'country code', 'cc2', 'admin1 code', 'admin2 code',
        'admin3 code', 'admin4 code', 'population', 'elevation',
        'dem', 'timezone', 'modification date']

geo_df = pd.read_csv(CITIES_FILE, sep='\t', names=cols, low_memory=False)
us_cities = geo_df[geo_df['country code'] == 'US'].copy()

# city name -> 후보 리스트
geonames_candidates = defaultdict(list)
for _, row in us_cities.iterrows():
    name = str(row['name']).strip()
    geonames_candidates[name].append({
        'geonameid': int(row['geonameid']),
        'name': name,
        'state': str(row['admin1 code']).strip(),
        'lat': float(row['latitude']),
        'lon': float(row['longitude']),
        'population': int(row['population']) if pd.notna(row['population']) else 0
    })

print(f"  GeoNames US 도시 수: {len(geonames_candidates)}")

# --- 2. Yelp 비즈니스 로드 & 모호 도시 필터 ---
print("[Step 4-2] business.json 로딩...")
city_states = defaultdict(set)
businesses = []

with open(f'{DATA_DIR}/yelp_academic_dataset_business.json', 'r') as f:
    for line in f:
        b = json.loads(line)
        city = b.get('city', '').strip()
        state = b.get('state', '').strip()
        if city and state:
            city_states[city].add(state)
            businesses.append(b)

# 2개 이상 state에 존재하는 도시만 = 모호한 도시
ambiguous_cities = {city for city, states in city_states.items()
                    if len(states) >= 2}

# GeoNames에도 2+ 후보가 있어야 함
truly_ambiguous = {city for city in ambiguous_cities
                   if len(geonames_candidates.get(city, [])) >= 2}

print(f"  Yelp 내 모호 도시: {len(ambiguous_cities)}")
print(f"  GeoNames에도 다중 후보 있는 도시: {len(truly_ambiguous)}")

# --- 3. 유저별 리뷰 매핑 ---
print("[Step 4-3] review.json 로딩...")
# business_id -> [(user_id, text), ...]
biz_reviews = defaultdict(list)
with open(f'{DATA_DIR}/yelp_academic_dataset_review.json', 'r') as f:
    for line in f:
        r = json.loads(line)
        biz_reviews[r['business_id']].append({
            'user_id': r['user_id'],
            'text': r.get('text', '')
        })

# test set 유저 로드 (GCN 예측이 있는 유저만 평가)
print("[Step 4-4] test 유저 로딩...")
test_users = set()
test_file = os.path.join(GEOTEXT_DIR, 'user_info.test.gz')
with gzip.open(test_file, 'rt') as f:
    for line in f:
        parts = line.strip().split('\t')
        if parts:
            test_users.add(parts[0])

print(f"  test 유저 수: {len(test_users)}")

# --- 4. 평가 데이터 생성 ---
print("[Step 4-5] 평가 데이터 구축 중...")
eval_data = []
skipped_no_state_match = 0

for biz in businesses:
    city = biz.get('city', '').strip()
    state = biz.get('state', '').strip()

    # 모호한 도시만
    if city not in truly_ambiguous:
        continue

    # GeoNames 후보 가져오기
    candidates = geonames_candidates.get(city, [])
    if len(candidates) < 2:
        continue

    # 정답 인덱스 찾기: state가 일치하는 후보
    true_idx = None
    for idx, cand in enumerate(candidates):
        if cand['state'] == state:
            true_idx = idx
            break

    # 정답 후보가 없으면 건너뛰기
    if true_idx is None:
        skipped_no_state_match += 1
        continue

    # 이 비즈니스를 리뷰한 test 유저 찾기
    bid = biz['business_id']
    for rev in biz_reviews.get(bid, []):
        uid = rev['user_id']
        if uid not in test_users:
            continue
        eval_data.append({
            'user_id': uid,
            'business_id': bid,
            'city_name': city,            # toponym (모호한 지명)
            'true_state': state,          # 정답 state (평가용)
            'true_lat': biz['latitude'],
            'true_lon': biz['longitude'],
            'review_text': rev['text'],
            'candidates': candidates,     # GeoNames 후보 리스트
            'true_candidate_idx': true_idx,
        })

print(f"\n  평가 데이터: {len(eval_data)}개 (toponym instances)")
print(f"  고유 도시: {len(set(e['city_name'] for e in eval_data))}")
print(f"  고유 유저: {len(set(e['user_id'] for e in eval_data))}")
print(f"  state 미매칭 스킵: {skipped_no_state_match}")

# 저장
output_file = 'yelp_eval_data.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(eval_data, f)
print(f"\n저장: {output_file}")
print("Step 4 완료!")
