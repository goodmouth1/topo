#!/usr/bin/env python
"""
Step 0: Yelp 데이터 탐색 & 모호 도시명 확인
- 모호한 도시 수 확인 (2개 이상 state에 존재)
- friendship 그래프 밀도 확인
- 유저당 리뷰 수 확인
"""

import json
from collections import defaultdict, Counter

DATA_DIR = '.'  # Yelp JSON 파일들이 있는 디렉토리

# 1. business.json에서 도시명 수집
print("=" * 60)
print("[Step 0-1] business.json 분석 중...")
city_states = defaultdict(set)   # city_name -> {state1, state2, ...}
city_coords = defaultdict(list)  # city_name -> [{lat, lon, state, business_id}, ...]

with open(f'{DATA_DIR}/yelp_academic_dataset_business.json', 'r') as f:
    for line in f:
        biz = json.loads(line)
        city = biz.get('city', '').strip()
        state = biz.get('state', '').strip()
        lat = biz.get('latitude')
        lon = biz.get('longitude')
        if city and state and lat and lon:
            city_states[city].add(state)
            city_coords[city].append({
                'lat': lat, 'lon': lon,
                'state': state,
                'business_id': biz['business_id']
            })

# 2. 모호한 도시명 찾기 (2개 이상 state에 존재)
ambiguous = {city: states for city, states in city_states.items()
             if len(states) >= 2}

print(f"전체 도시 수: {len(city_states)}")
print(f"모호한 도시 수 (2+ states): {len(ambiguous)}")
print(f"\n모호한 도시 목록:")
for city, states in sorted(ambiguous.items(), key=lambda x: -len(x[1])):
    counts = Counter(e['state'] for e in city_coords[city])
    print(f"  {city}: {dict(counts)}")

# 3. 유저 수 & friendship 밀도 확인
print("\n" + "=" * 60)
print("[Step 0-2] user.json 분석 중...")
user_count = 0
has_friends = 0
total_friends = 0

with open(f'{DATA_DIR}/yelp_academic_dataset_user.json', 'r') as f:
    for line in f:
        user = json.loads(line)
        user_count += 1
        friends = user.get('friends', '')
        if friends and friends != 'None':
            friend_list = [f.strip() for f in friends.split(',') if f.strip()]
            if len(friend_list) > 0:
                has_friends += 1
                total_friends += len(friend_list)

print(f"전체 유저 수: {user_count}")
print(f"친구 있는 유저: {has_friends} ({has_friends/user_count*100:.1f}%)")
print(f"평균 친구 수: {total_friends/max(has_friends,1):.1f}")

# 4. 유저당 리뷰 수 확인
print("\n" + "=" * 60)
print("[Step 0-3] review.json 분석 중...")
reviews_per_user = Counter()
with open(f'{DATA_DIR}/yelp_academic_dataset_review.json', 'r') as f:
    for line in f:
        review = json.loads(line)
        reviews_per_user[review['user_id']] += 1

review_counts = list(reviews_per_user.values())
print(f"전체 리뷰 수: {sum(review_counts)}")
print(f"리뷰 작성 유저 수: {len(review_counts)}")
print(f"유저당 평균 리뷰: {sum(review_counts)/len(review_counts):.1f}")
print(f"유저당 중앙값 리뷰: {sorted(review_counts)[len(review_counts)//2]}")

print("\n" + "=" * 60)
print("Step 0 완료! 위 결과를 확인하고 실험 가능 여부를 판단하세요.")
print("- 모호한 도시가 30개 이상이면 좋음")
print("- friendship이 충분히 dense해야 함")
print("- 유저당 리뷰가 TF-IDF에 충분해야 함")
