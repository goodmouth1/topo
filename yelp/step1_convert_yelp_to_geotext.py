#!/usr/bin/env python
"""
Step 1: Yelp -> GeoText 호환 포맷 변환
- user_info.{train,dev,test}.gz 생성 (user_id \t lat \t lon \t text)
- friendship_edges.tsv 생성
- 유저 좌표 = 리뷰한 비즈니스 좌표의 중앙값
"""

import json, gzip, random, os
import numpy as np
from collections import defaultdict

DATA_DIR = '.'  # Yelp JSON 파일들이 있는 디렉토리
OUT_DIR = 'yelp_geotext_format'

# 미국 주(state) 코드 목록 (캐나다 등 제외)
US_STATES = {
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
    'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
    'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
    'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY',
    'DC'
}

# --- 1. business 좌표 로드 (미국만) ---
print("[Step 1-1] business.json 로딩 (US only)...")
biz_loc = {}  # business_id -> {lat, lon, city, state}
skipped_non_us = 0
with open(f'{DATA_DIR}/yelp_academic_dataset_business.json', 'r') as f:
    for line in f:
        b = json.loads(line)
        state = b.get('state', '').strip()
        if state not in US_STATES:
            skipped_non_us += 1
            continue
        biz_loc[b['business_id']] = {
            'lat': b['latitude'], 'lon': b['longitude'],
            'city': b['city'], 'state': state
        }
print(f"  미국 비즈니스 수: {len(biz_loc)} (미국 외 제외: {skipped_non_us})")

# --- 2. 유저별 리뷰 텍스트 & 방문 좌표 수집 ---
print("[Step 1-2] review.json 로딩...")
user_texts = defaultdict(list)       # user_id -> [review_text, ...]
user_biz_coords = defaultdict(list)  # user_id -> [(lat, lon), ...]
user_businesses = defaultdict(list)  # user_id -> [business_id, ...]

with open(f'{DATA_DIR}/yelp_academic_dataset_review.json', 'r') as f:
    for line in f:
        r = json.loads(line)
        uid = r['user_id']
        bid = r['business_id']
        text = r.get('text', '').replace('\n', ' ').replace('\t', ' ')
        user_texts[uid].append(text)
        if bid in biz_loc:
            loc = biz_loc[bid]
            user_biz_coords[uid].append((loc['lat'], loc['lon']))
            user_businesses[uid].append(bid)

print(f"  리뷰 작성 유저 수: {len(user_texts)}")

# --- 3. 유저 좌표 결정 (리뷰한 비즈니스 좌표의 중앙값) ---
print("[Step 1-3] 유저 좌표 계산 중...")
user_locations = {}
for uid, coords in user_biz_coords.items():
    if len(coords) >= 3:  # 최소 3개 리뷰 필요
        lats = [c[0] for c in coords]
        lons = [c[1] for c in coords]
        user_locations[uid] = (np.median(lats), np.median(lons))

print(f"  좌표 계산 가능 유저 (리뷰 3+): {len(user_locations)}")

# --- 4. friendship 그래프에 있는 유저만 필터 ---
print("[Step 1-4] user.json에서 friendship 로딩...")
user_friends = {}
with open(f'{DATA_DIR}/yelp_academic_dataset_user.json', 'r') as f:
    for line in f:
        u = json.loads(line)
        uid = u['user_id']
        friends_str = u.get('friends', '')
        if friends_str and friends_str != 'None':
            friends = [fr.strip() for fr in friends_str.split(',') if fr.strip()]
        else:
            friends = []
        if uid in user_locations and len(friends) > 0:
            user_friends[uid] = friends

valid_users = set(user_friends.keys())
print(f"  위치+친구+리뷰 모두 있는 유저: {len(valid_users)}")

# --- 5. train/dev/test 분할 (80/10/10) ---
print("[Step 1-5] 데이터 분할 중 (80/10/10)...")
user_list = sorted(valid_users)
random.seed(42)
random.shuffle(user_list)

n = len(user_list)
train_end = int(n * 0.8)
dev_end = int(n * 0.9)

splits = {
    'train': user_list[:train_end],
    'dev': user_list[train_end:dev_end],
    'test': user_list[dev_end:]
}

# --- 6. GeoText 포맷으로 저장 ---
print("[Step 1-6] GeoText 포맷으로 저장 중...")
os.makedirs(OUT_DIR, exist_ok=True)

for split_name, users in splits.items():
    filepath = os.path.join(OUT_DIR, f'user_info.{split_name}.gz')
    with gzip.open(filepath, 'wt', encoding='utf-8') as f:
        for uid in users:
            lat, lon = user_locations[uid]
            all_text = ' '.join(user_texts[uid])
            all_text = all_text.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
            # C 파서 버퍼 오버플로 방지: 텍스트 50,000자로 제한 (TF-IDF에 충분)
            if len(all_text) > 50000:
                all_text = all_text[:50000]
            f.write(f"{uid}\t{lat}\t{lon}\t{all_text}\n")
    print(f"  {split_name}: {len(users)} users -> {filepath}")

# --- 7. friendship 엣지 파일 저장 ---
print("[Step 1-7] friendship edges 저장 중...")
edges_file = os.path.join(OUT_DIR, 'friendship_edges.tsv')
edge_count = 0
with open(edges_file, 'w') as f:
    for uid, friends in user_friends.items():
        if uid not in valid_users:
            continue
        for fid in friends:
            if fid in valid_users:
                f.write(f"{uid}\t{fid}\n")
                edge_count += 1

print(f"  Friendship edges: {edge_count} -> {edges_file}")

# --- 8. 메타데이터 저장 ---
meta = {
    'total_users': len(valid_users),
    'train': len(splits['train']),
    'dev': len(splits['dev']),
    'test': len(splits['test']),
    'total_edges': edge_count
}
with open(os.path.join(OUT_DIR, 'meta.json'), 'w') as f:
    json.dump(meta, f, indent=2)

print(f"\n메타데이터: {meta}")
print("Step 1 완료!")
