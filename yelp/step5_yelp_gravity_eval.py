#!/usr/bin/env python
"""
Step 5: Gravity Model 평가

GCN이 예측한 유저 좌표(anchor)와 Gravity Model로
Yelp 도시명을 해소한다.

Score = Population / Distance^beta

사용법:
  python step5_yelp_gravity_eval.py [gcn_pred_file]
  예: python step5_yelp_gravity_eval.py gcn_1.0_percent_pred_256.pkl
"""

import sys, pickle, os
import numpy as np
import gzip

GEOTEXT_DIR = 'yelp_geotext_format'


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


# --- 1. GCN 예측 좌표 로드 ---
gcn_file = sys.argv[1] if len(sys.argv) > 1 else 'gcn_1.0_percent_pred_256.pkl'
print(f"[Step 5-1] GCN 예측 좌표 로딩: {gcn_file}")

with open(gcn_file, 'rb') as f:
    distances, latlon_true, latlon_pred = pickle.load(f)

# test 유저 ID 목록 로드 (순서 매칭)
test_users = []
test_file = os.path.join(GEOTEXT_DIR, 'user_info.test.gz')
with gzip.open(test_file, 'rt') as f:
    for line in f:
        parts = line.strip().split('\t')
        if parts:
            test_users.append(parts[0])

# user_id -> GCN 예측 좌표
user_gcn_pred = {}
for i, uid in enumerate(test_users):
    if i < len(latlon_pred):
        user_gcn_pred[uid] = latlon_pred[i]  # [lat, lon]

print(f"  GCN 예측 좌표: {len(user_gcn_pred)} users")

# --- 2. 평가 데이터 로드 ---
print("[Step 5-2] 평가 데이터 로딩...")
with open('yelp_eval_data.pkl', 'rb') as f:
    eval_data = pickle.load(f)

print(f"  평가 데이터: {len(eval_data)} instances")

# GCN 예측이 있는 데이터만 필터
eval_with_gcn = [item for item in eval_data if item['user_id'] in user_gcn_pred]
print(f"  GCN 예측 있는 인스턴스: {len(eval_with_gcn)}")

# --- 3. 다양한 beta로 Gravity Model 평가 ---
print("\n" + "=" * 60)
print("Gravity Model 평가 결과 (GCN anchor)")
print("=" * 60)

for beta in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    correct = 0
    total = 0

    for item in eval_with_gcn:
        uid = item['user_id']
        pred_lat, pred_lon = user_gcn_pred[uid]
        candidates = item['candidates']
        true_idx = item['true_candidate_idx']

        # Gravity Model: 각 후보에 점수 매기기
        best_score = -1
        best_idx = -1

        for idx, cand in enumerate(candidates):
            dist = haversine(pred_lat, pred_lon, cand['lat'], cand['lon'])
            if dist < 1:
                dist = 1  # 0 방지

            pop = max(cand['population'], 1000)  # 최소 인구
            score = pop / (dist ** beta)

            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx == true_idx:
            correct += 1
        total += 1

    if total > 0:
        acc = correct / total * 100
        print(f"  beta={beta:.1f}: {acc:.2f}% ({correct}/{total})")

# --- 4. Baseline: Population Only (GCN 없이) ---
print("\n" + "=" * 60)
print("Baseline 평가 (Population Only, GCN 없이)")
print("=" * 60)

correct_pop = 0
total_pop = 0

for item in eval_data:
    candidates = item['candidates']
    true_idx = item['true_candidate_idx']

    # 인구 최대 후보 선택
    best_idx = max(range(len(candidates)),
                   key=lambda i: candidates[i]['population'])

    if best_idx == true_idx:
        correct_pop += 1
    total_pop += 1

if total_pop > 0:
    print(f"  Population Only: {correct_pop / total_pop * 100:.2f}% ({correct_pop}/{total_pop})")

# --- 5. State 보너스 포함 Gravity Model ---
print("\n" + "=" * 60)
print("Gravity Model + State 보너스 (GCN anchor)")
print("=" * 60)

import pandas as pd

# GeoNames에서 좌표->state 매핑
cities_file = os.path.join(os.path.dirname(__file__), '..', 'cities1000.txt')
if not os.path.exists(cities_file):
    cities_file = 'cities1000.txt'

if os.path.exists(cities_file):
    geo_cols = ['geonameid', 'name', 'asciiname', 'alternatenames',
                'latitude', 'longitude', 'feature class', 'feature code',
                'country code', 'cc2', 'admin1 code', 'admin2 code',
                'admin3 code', 'admin4 code', 'population', 'elevation',
                'dem', 'timezone', 'modification date']
    geo_df = pd.read_csv(cities_file, sep='\t', names=geo_cols, low_memory=False)
    us_cities_geo = geo_df[geo_df['country code'] == 'US'].dropna(subset=['latitude', 'longitude', 'admin1 code'])
    us_coords = us_cities_geo[['latitude', 'longitude']].values
    us_states = us_cities_geo['admin1 code'].values

    def get_state_from_gcn(lat, lon):
        diff = np.linalg.norm(us_coords - np.array([lat, lon]), axis=1)
        nearest_idx = np.argmin(diff)
        return str(us_states[nearest_idx]).strip()

    for beta in [1.0, 1.5, 2.0]:
        correct = 0
        total = 0

        for item in eval_with_gcn:
            uid = item['user_id']
            pred_lat, pred_lon = user_gcn_pred[uid]
            candidates = item['candidates']
            true_idx = item['true_candidate_idx']

            gcn_predicted_state = get_state_from_gcn(pred_lat, pred_lon)

            best_score = -1
            best_idx = -1

            for idx, cand in enumerate(candidates):
                dist = haversine(pred_lat, pred_lon, cand['lat'], cand['lon'])
                if dist < 1:
                    dist = 1

                pop = max(cand['population'], 1000)

                gcn_match = (cand['state'] == gcn_predicted_state)
                bonus = 5.0 if gcn_match else 1.0

                score = (pop / (dist ** beta)) * bonus

                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx == true_idx:
                correct += 1
            total += 1

        if total > 0:
            acc = correct / total * 100
            print(f"  beta={beta:.1f} + State bonus: {acc:.2f}% ({correct}/{total})")
else:
    print("  cities1000.txt를 찾을 수 없어 State 보너스 평가 건너뜀.")

print("\n" + "=" * 60)
print("Step 5 완료!")
