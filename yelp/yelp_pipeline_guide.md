# Yelp GCN + Gravity Model 파이프라인 구축 가이드

## Claude Code에게: 이 문서를 읽고 Yelp 데이터셋 기반 지명 모호성 해소 실험을 구축해주세요.

---

## 1. 프로젝트 배경

석사 논문 주제: **GCN 기반 지명 모호성 해소 (Toponym Disambiguation)**

핵심 아이디어: 소셜 네트워크의 그래프 구조(GCN)로 유저 위치를 예측하고, 그 위치를 anchor로 삼아 Gravity Model로 모호한 지명을 해소한다.

### 이미 완성된 실험: Twitter (GeoText)
- 소셜 그래프: @mention 관계
- 텍스트 특징: Tweet TF-IDF
- GCN → 유저 위치 예측 → Gravity Model → 지명 해소
- 결과: 88.51% → 92.34% (Gravity Model 적용 후)
- 코드 위치: `/home/wang/master_thesis/` (서버)

### 새로 구축할 실험: Yelp
- 소셜 그래프: friendship 관계
- 텍스트 특징: Review TF-IDF
- 동일한 GCN → Gravity Model 프레임워크
- 목적: **플랫폼 독립적 일반화 가능성 검증**

---

## 2. 기존 GeoText 코드 구조 (참고용)

서버의 `/home/wang/master_thesis/` 디렉토리에 있는 핵심 파일들:

### 데이터 포맷 (GeoText)
```
data/
├── user_info.train.gz   # TSV: user \t lat \t lon \t text
├── user_info.dev.gz     # 같은 포맷
├── user_info.test.gz    # 같은 포맷
└── dump.pkl(.gz)        # 전처리된 캐시
```

각 행: `username \t latitude \t longitude \t 해당유저의모든트윗합침`

### 핵심 파일들

| 파일 | 역할 |
|------|------|
| `data.py` | DataLoader 클래스. 데이터 로딩, TF-IDF, @mention 그래프 구축 |
| `gcnmodel.py` | GraphConv 클래스 (Theano/Lasagne 기반 GCN) |
| `gcnmain.py` | GCN 훈련 메인. preprocess_data() → main() |
| `run_gravity_with_gcn_state.py` | GCN 예측 결과 + Gravity Model로 지명 해소 |
| `labeling/manual_labeled_data.pkl` | 수동 라벨링한 지명 정답 (235개 toponym) |
| `cities1000.txt` | GeoNames 데이터 (인구 1000+ 도시) |

### data.py DataLoader 핵심 흐름
```python
class DataLoader:
    def load_data(self):
        # user_info.{train,dev,test}.gz 읽기
        # 포맷: user \t lat \t lon \t text
        # user를 index로 설정

    def get_graph(self):
        # 텍스트에서 @mention 추출 (정규식)
        # user-user 간 edge 생성
        # networkx Graph 반환

    def tfidf(self):
        # sklearn TfidfVectorizer로 텍스트 벡터화
        # sparse matrix 반환

    def assignClasses(self):
        # 좌표를 bucket_size(300km) 기반 클래스로 이산화
        # 각 클래스의 중앙값 좌표 계산
```

### gcnmain.py 훈련 흐름
```python
def preprocess_data(data_home):
    dl = DataLoader(data_home)
    dl.load_data()        # 데이터 로딩
    dl.assignClasses()    # 좌표 → 클래스 이산화
    dl.tfidf()            # TF-IDF 벡터화
    dl.get_graph()        # 소셜 그래프 구축
    adj = nx.adjacency_matrix(dl.graph)  # 인접행렬
    # 정규화: D^{-1/2} A D^{-1/2}
    return (X, Y, adj, ...)

def main(data, args):
    clf = GraphConv(input_size, output_size, hidden_sizes, ...)
    clf.build_model(A)
    clf.fit(X, A, Y, train_indices, val_indices, ...)
    y_pred = clf.predict(X, A, test_indices)
    # 출력: gcn_1.0_percent_pred_256.pkl
    #   → (distances, latlon_true, latlon_pred)
```

### Gravity Model (run_gravity_with_gcn_state.py) 흐름
```python
# 1. GCN 예측 좌표 로드
# 2. 수동 라벨링 데이터 로드 (manual_labeled_data.pkl)
# 3. 각 toponym에 대해:
#    - GCN 예측 좌표 → anchor
#    - 후보 도시들에 대해 Score = (Population / Distance^1.5) * bonus
#    - bonus: GCN 예측 state와 후보 state 일치 여부
#    - 최고 점수 후보 선택
#    - 정답과 비교
```

---

## 3. Yelp 데이터셋 다운로드

### 다운로드
```bash
# Yelp Open Dataset 다운로드
# https://www.yelp.com/dataset 에서 다운로드 (약 9GB)
# 또는 Kaggle: https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset

# 압축 해제 후 다음 파일들 사용:
# yelp_academic_dataset_business.json
# yelp_academic_dataset_review.json  
# yelp_academic_dataset_user.json
```

### Yelp JSON 구조

**business.json** (한 줄에 하나의 JSON):
```json
{
  "business_id": "tnhfDv5Il8EaGSXZGiuQGg",
  "name": "Garaje",
  "city": "Phoenix",           // ← 이것이 toponym (지명)
  "state": "AZ",               // ← 평가 시 제거 (모호성 생성)
  "latitude": 33.4484,         // ← ground truth 좌표
  "longitude": -112.0740,
  "stars": 4.5,
  "review_count": 1198,
  "categories": "Mexican, Burgers, Gastropubs"
}
```

**user.json**:
```json
{
  "user_id": "Ha3iJu77CxlrFm-vQRs_8g",
  "name": "Joanne",
  "review_count": 108,
  "friends": "oMy_rEb0UBEmMlu-zcxnoQ, KfB_1r5vDzSTCWTfJPYs6Q, ...",
  // ← 쉼표 구분 user_id 리스트 = friendship 그래프!
  "elite": "2015,2016,2017"
}
```

**review.json**:
```json
{
  "review_id": "xQY8N_XvtGbearJ5X4QN1Q",
  "user_id": "Ha3iJu77CxlrFm-vQRs_8g",
  "business_id": "tnhfDv5Il8EaGSXZGiuQGg",
  "stars": 4,
  "text": "The food was amazing! Best tacos I've ever had...",
  "date": "2016-03-09"
}
```

---

## 4. Yelp 파이프라인 구축 단계

### 개요: 6단계

```
[Step 0] 데이터 탐색 & 모호 도시명 확인
[Step 1] Yelp → GeoText 포맷 변환
[Step 2] Friendship 그래프 구축
[Step 3] GCN 훈련 (기존 코드 재사용)
[Step 4] 지명 모호성 해소용 평가 데이터 구축
[Step 5] Gravity Model 평가 (기존 코드 적응)
```

---

### Step 0: 데이터 탐색 & 모호 도시명 확인 ⭐ (먼저 해야 함)

이 단계에서 실험 가능 여부를 판단한다.

```python
# step0_explore_yelp.py
import json
from collections import defaultdict, Counter

# 1. business.json에서 도시명 수집
city_states = defaultdict(set)  # city_name → {state1, state2, ...}
city_coords = defaultdict(list) # city_name → [(lat, lon, state, business_id), ...]

with open('yelp_academic_dataset_business.json', 'r') as f:
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
for city, states in sorted(ambiguous.items(), 
                            key=lambda x: -len(x[1])):
    counts = Counter(e['state'] for e in city_coords[city])
    print(f"  {city}: {dict(counts)}")

# 3. 유저 수 & friendship 밀도 확인
user_count = 0
has_friends = 0
total_friends = 0

with open('yelp_academic_dataset_user.json', 'r') as f:
    for line in f:
        user = json.loads(line)
        user_count += 1
        friends = user.get('friends', '')
        if friends and friends != 'None':
            friend_list = [f.strip() for f in friends.split(',') if f.strip()]
            if len(friend_list) > 0:
                has_friends += 1
                total_friends += len(friend_list)

print(f"\n전체 유저 수: {user_count}")
print(f"친구 있는 유저: {has_friends} ({has_friends/user_count*100:.1f}%)")
print(f"평균 친구 수: {total_friends/max(has_friends,1):.1f}")

# 4. 유저당 리뷰 수 확인
reviews_per_user = Counter()
with open('yelp_academic_dataset_review.json', 'r') as f:
    for line in f:
        review = json.loads(line)
        reviews_per_user[review['user_id']] += 1

review_counts = list(reviews_per_user.values())
print(f"\n전체 리뷰 수: {sum(review_counts)}")
print(f"리뷰 작성 유저 수: {len(review_counts)}")
print(f"유저당 평균 리뷰: {sum(review_counts)/len(review_counts):.1f}")
print(f"유저당 중앙값 리뷰: {sorted(review_counts)[len(review_counts)//2]}")
```

**⚠️ 중요: 이 스크립트의 결과를 먼저 확인할 것!**
- 모호한 도시가 충분한가? (최소 30개 이상이면 좋음)
- friendship 그래프가 충분히 dense한가?
- 유저당 리뷰가 TF-IDF에 충분한가?

---

### Step 1: Yelp → GeoText 호환 포맷 변환

기존 DataLoader를 최대한 재사용하기 위해 GeoText와 동일한 포맷으로 변환한다.

**목표 포맷**: `user_info.{train,dev,test}.gz`
```
user_id \t lat \t lon \t 유저의 모든 리뷰 텍스트 합침
```

유저의 좌표 = **해당 유저가 리뷰한 비즈니스들의 좌표 중앙값** (median lat, median lon)

```python
# step1_convert_yelp_to_geotext.py
import json, gzip, csv, random
import numpy as np
from collections import defaultdict

# --- 1. business 좌표 로드 ---
biz_loc = {}  # business_id → (lat, lon, city, state)
with open('yelp_academic_dataset_business.json', 'r') as f:
    for line in f:
        b = json.loads(line)
        biz_loc[b['business_id']] = {
            'lat': b['latitude'], 'lon': b['longitude'],
            'city': b['city'], 'state': b['state']
        }

# --- 2. 유저별 리뷰 텍스트 & 방문 좌표 수집 ---
user_texts = defaultdict(list)      # user_id → [review_text, ...]
user_biz_coords = defaultdict(list) # user_id → [(lat, lon), ...]
user_businesses = defaultdict(list)  # user_id → [business_id, ...]

with open('yelp_academic_dataset_review.json', 'r') as f:
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

# --- 3. 유저 좌표 결정 (리뷰한 비즈니스 좌표의 중앙값) ---
user_locations = {}
for uid, coords in user_biz_coords.items():
    if len(coords) >= 3:  # 최소 3개 리뷰 필요
        lats = [c[0] for c in coords]
        lons = [c[1] for c in coords]
        user_locations[uid] = (np.median(lats), np.median(lons))

# --- 4. friendship 그래프에 있는 유저만 필터 ---
user_friends = {}
with open('yelp_academic_dataset_user.json', 'r') as f:
    for line in f:
        u = json.loads(line)
        uid = u['user_id']
        friends_str = u.get('friends', '')
        if friends_str and friends_str != 'None':
            friends = [f.strip() for f in friends_str.split(',') if f.strip()]
        else:
            friends = []
        if uid in user_locations and len(friends) > 0:
            user_friends[uid] = friends

# 양방향 친구 관계만 유지 (선택적)
valid_users = set(user_friends.keys())
print(f"위치+친구+리뷰 모두 있는 유저: {len(valid_users)}")

# --- 5. train/dev/test 분할 (80/10/10) ---
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
import os
out_dir = 'yelp_geotext_format'
os.makedirs(out_dir, exist_ok=True)

for split_name, users in splits.items():
    filepath = os.path.join(out_dir, f'user_info.{split_name}.gz')
    with gzip.open(filepath, 'wt', encoding='utf-8') as f:
        for uid in users:
            lat, lon = user_locations[uid]
            # 모든 리뷰 합치기
            all_text = ' '.join(user_texts[uid])
            # 탭/줄바꿈 제거
            all_text = all_text.replace('\t', ' ').replace('\n', ' ')
            f.write(f"{uid}\t{lat}\t{lon}\t{all_text}\n")
    print(f"{split_name}: {len(users)} users → {filepath}")

# --- 7. friendship 엣지 파일 저장 (get_graph 대체용) ---
# 기존 data.py의 get_graph()는 @mention으로 그래프를 만들지만
# Yelp에서는 friendship으로 대체해야 함
edges_file = os.path.join(out_dir, 'friendship_edges.tsv')
edge_count = 0
with open(edges_file, 'w') as f:
    for uid, friends in user_friends.items():
        if uid not in valid_users:
            continue
        for fid in friends:
            if fid in valid_users:
                f.write(f"{uid}\t{fid}\n")
                edge_count += 1

print(f"Friendship edges: {edge_count} → {edges_file}")

# --- 8. 메타데이터 저장 ---
meta = {
    'total_users': len(valid_users),
    'train': len(splits['train']),
    'dev': len(splits['dev']),
    'test': len(splits['test']),
    'total_edges': edge_count
}
with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
    json.dump(meta, f, indent=2)
print(f"\n메타데이터: {meta}")
```

---

### Step 2: Friendship 그래프 구축 (data.py 수정)

기존 `data.py`의 `get_graph()`는 텍스트에서 @mention을 파싱해서 그래프를 만든다.
Yelp에서는 이를 **friendship_edges.tsv**에서 직접 로드하도록 수정해야 한다.

**방법 A: data.py를 상속해서 YelpDataLoader 만들기 (권장)**

```python
# yelp_data.py
import os
import networkx as nx
import logging
from data import DataLoader

class YelpDataLoader(DataLoader):
    """Yelp용 DataLoader. friendship 그래프를 @mention 대신 사용."""
    
    def get_graph(self):
        """friendship_edges.tsv에서 그래프 로드 (기존 @mention 파싱 대체)"""
        g = nx.Graph()
        nodes_list = (self.df_train.index.tolist() + 
                      self.df_dev.index.tolist() + 
                      self.df_test.index.tolist())
        node_id = {node: id for id, node in enumerate(nodes_list)}
        
        # self-loop 추가
        g.add_nodes_from(node_id.values())
        for node in node_id:
            g.add_edge(node_id[node], node_id[node])
        
        # friendship edges 로드
        edges_file = os.path.join(self.data_home, 'friendship_edges.tsv')
        edge_count = 0
        with open(edges_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    u, v = parts
                    if u in node_id and v in node_id:
                        g.add_edge(node_id[u], node_id[v])
                        edge_count += 1
        
        logging.info(f'Yelp friendship graph: {g.number_of_nodes()} nodes, '
                     f'{g.number_of_edges()} edges ({edge_count} from file)')
        self.graph = g
```

**방법 B: gcnmain.py에서 DataLoader 교체**

`gcnmain.py`의 `preprocess_data()` 함수에서:
```python
# 기존 (GeoText):
dl = DataLoader(data_home=data_home, ...)

# Yelp로 변경:
from yelp_data import YelpDataLoader
dl = YelpDataLoader(data_home=data_home, ...)
```

나머지 `load_data()`, `tfidf()`, `assignClasses()`는 **그대로 재사용** 가능.
(user_info.{train,dev,test}.gz 포맷이 동일하므로)

---

### Step 3: GCN 훈련

기존 명령어와 거의 동일:

```bash
# 기존 GeoText 훈련 명령어:
python gcnmain.py -d ./data -bucket 300 -hid 300 300 300 \
  -mindf 10 -reg 1e-6 -dropout 0.5 -cel 10 \
  -conv -highway -builddata

# Yelp 훈련 (디렉토리만 변경):
python gcnmain.py -d ./yelp_geotext_format -bucket 300 -hid 300 300 300 \
  -mindf 10 -reg 1e-6 -dropout 0.5 -cel 10 \
  -conv -highway -builddata
```

**주의사항**:
- `gcnmain.py`에서 `DataLoader` → `YelpDataLoader`로 교체했는지 확인
- Yelp 데이터가 훨씬 크므로 메모리 주의 (리뷰 텍스트가 트윗보다 길다)
- 필요시 `celebrity_threshold` 조정 (친구가 너무 많은 유저 필터)
- bucket_size도 조정 가능 (Yelp는 특정 metro area에 집중되어 있음)

**출력**: `gcn_1.0_percent_pred_256.pkl`
- (distances, latlon_true, latlon_pred) 형태
- latlon_pred = GCN이 예측한 각 유저의 좌표 ← 이것이 Gravity Model의 anchor

---

### Step 4: 지명 모호성 해소용 평가 데이터 구축 ⭐⭐⭐

이것이 Yelp 실험의 **핵심 혁신**이다.

GeoText에서는 235개 toponym을 **수작업으로 라벨링**했다.
Yelp에서는 **city 필드가 자동으로 정답을 제공**한다!

```python
# step4_build_eval_data.py
"""
Yelp 비즈니스의 city 필드를 toponym으로, 
실제 lat/lon을 ground truth로 사용하여 
평가 데이터를 자동 생성한다.

핵심 아이디어:
- business.city = "Portland" → 이것이 모호한 지명
- state 정보를 제거하면 Portland, OR인지 Portland, ME인지 모름
- GeoNames에서 "Portland" 검색 → 후보 리스트
- Gravity Model로 어떤 Portland인지 예측
- business.lat/lon과 비교하여 정답 확인
"""

import json, pickle
import pandas as pd
from collections import defaultdict

# --- 1. GeoNames 로드 ---
cols = ['geonameid', 'name', 'asciiname', 'alternatenames', 
        'latitude', 'longitude', 'feature class', 'feature code', 
        'country code', 'cc2', 'admin1 code', 'admin2 code', 
        'admin3 code', 'admin4 code', 'population', 'elevation', 
        'dem', 'timezone', 'modification date']

geo_df = pd.read_csv('cities1000.txt', sep='\t', names=cols, low_memory=False)
us_cities = geo_df[geo_df['country code'] == 'US'].copy()

# city name → 후보 리스트
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

# --- 2. Yelp 비즈니스 로드 & 모호 도시 필터 ---
city_states = defaultdict(set)
businesses = []

with open('yelp_academic_dataset_business.json', 'r') as f:
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

print(f"Yelp 내 모호 도시: {len(ambiguous_cities)}")
print(f"GeoNames에도 다중 후보 있는 도시: {len(truly_ambiguous)}")

# --- 3. 평가 데이터 구축 ---
# review와 연결하여 유저 정보 포함

# 유저별 리뷰한 비즈니스 매핑
user_reviews = defaultdict(list)  # user_id → [(business_id, text), ...]
with open('yelp_academic_dataset_review.json', 'r') as f:
    for line in f:
        r = json.loads(line)
        user_reviews[r['user_id']].append({
            'business_id': r['business_id'],
            'text': r.get('text', '')
        })

# test set 유저 로드 (GCN 예측이 있는 유저만 평가)
import gzip
test_users = set()
with gzip.open('yelp_geotext_format/user_info.test.gz', 'rt') as f:
    for line in f:
        parts = line.strip().split('\t')
        if parts:
            test_users.add(parts[0])

# 평가 데이터 생성
eval_data = []
for biz in businesses:
    city = biz['city'].strip()
    state = biz['state'].strip()
    
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
        continue
    
    # 이 비즈니스를 리뷰한 test 유저 찾기
    for uid, reviews in user_reviews.items():
        if uid not in test_users:
            continue
        for rev in reviews:
            if rev['business_id'] == biz['business_id']:
                eval_data.append({
                    'user_id': uid,
                    'business_id': biz['business_id'],
                    'city_name': city,          # toponym (모호한 지명)
                    'true_state': state,        # 정답 state (평가용)
                    'true_lat': biz['latitude'],
                    'true_lon': biz['longitude'],
                    'review_text': rev['text'],
                    'candidates': candidates,   # GeoNames 후보 리스트
                    'true_candidate_idx': true_idx,  # 정답 인덱스
                })

print(f"\n평가 데이터: {len(eval_data)}개 (toponym instances)")
print(f"고유 도시: {len(set(e['city_name'] for e in eval_data))}")
print(f"고유 유저: {len(set(e['user_id'] for e in eval_data))}")

# 저장
with open('yelp_eval_data.pkl', 'wb') as f:
    pickle.dump(eval_data, f)
print(f"저장: yelp_eval_data.pkl")
```

---

### Step 5: Gravity Model 평가

기존 `run_gravity_with_gcn_state.py`를 Yelp용으로 적응.

```python
# step5_yelp_gravity_eval.py
"""
GCN이 예측한 유저 좌표(anchor)와 Gravity Model로
Yelp 도시명을 해소한다.

Score = Population / Distance^β

β ∈ {0.5, 1.0, 1.5, 2.0, 2.5, 3.0} 테스트
"""

import sys, pickle, json
import numpy as np
import gzip

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

# --- 1. GCN 예측 좌표 로드 ---
gcn_file = sys.argv[1] if len(sys.argv) > 1 else 'gcn_1.0_percent_pred_256.pkl'
with open(gcn_file, 'rb') as f:
    distances, latlon_true, latlon_pred = pickle.load(f)

# test 유저 ID 목록 로드 (순서 매칭)
test_users = []
with gzip.open('yelp_geotext_format/user_info.test.gz', 'rt') as f:
    for line in f:
        parts = line.strip().split('\t')
        if parts:
            test_users.append(parts[0])

# user_id → GCN 예측 좌표
user_gcn_pred = {}
for i, uid in enumerate(test_users):
    if i < len(latlon_pred):
        user_gcn_pred[uid] = latlon_pred[i]  # [lat, lon]

print(f"GCN 예측 좌표: {len(user_gcn_pred)} users")

# --- 2. 평가 데이터 로드 ---
with open('yelp_eval_data.pkl', 'rb') as f:
    eval_data = pickle.load(f)

print(f"평가 데이터: {len(eval_data)} instances")

# --- 3. 다양한 β로 Gravity Model 평가 ---
for beta in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    correct = 0
    total = 0
    
    for item in eval_data:
        uid = item['user_id']
        
        # GCN 예측 좌표가 있는 유저만
        if uid not in user_gcn_pred:
            continue
        
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
        print(f"β={beta:.1f}: {acc:.2f}% ({correct}/{total})")

# --- 4. Baseline: Population Only (GCN 없이) ---
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

print(f"\nBaseline (Population Only): "
      f"{correct_pop/total_pop*100:.2f}% ({correct_pop}/{total_pop})")
```

---

## 5. 실행 순서 요약

```bash
# 0. Yelp 데이터 다운로드 & 압축 해제
cd /home/wang/master_thesis/yelp_experiment/
# (yelp_academic_dataset_*.json 파일들이 여기에)

# 1. 데이터 탐색 (실험 가능 여부 확인) ★ 반드시 먼저!
python step0_explore_yelp.py
# → 모호 도시 수, friendship 밀도, 리뷰 수 확인

# 2. GeoText 포맷 변환
python step1_convert_yelp_to_geotext.py
# → yelp_geotext_format/ 디렉토리 생성

# 3. GCN 훈련
# (yelp_data.py를 먼저 만들어서 같은 디렉토리에)
python gcnmain.py -d ./yelp_geotext_format -bucket 300 \
  -hid 300 300 300 -mindf 10 -reg 1e-6 -dropout 0.5 \
  -cel 10 -conv -highway -builddata
# → gcn_1.0_percent_pred_256.pkl 생성

# 4. 평가 데이터 구축
python step4_build_eval_data.py
# → yelp_eval_data.pkl 생성

# 5. Gravity Model 평가
python step5_yelp_gravity_eval.py gcn_1.0_percent_pred_256.pkl
# → β별 정확도 출력
```

---

## 6. GeoText vs Yelp 대응 관계 정리

| 구성요소 | GeoText (Twitter) | Yelp |
|---------|-------------------|------|
| 노드 | Twitter 유저 (9,475) | Yelp 유저 (~150K+) |
| 엣지 | @mention (텍스트 파싱) | friendship (user.json) |
| 텍스트 | Tweet (짧음) | Review (길음) |
| TF-IDF | Tweet 합침 | Review 합침 |
| 좌표 | 유저 프로필 좌표 | 리뷰한 비즈니스 중앙값 |
| Toponym | Tweet 내 지명 (NER) | business.city (구조화) |
| 후보 생성 | GeoNames 검색 | GeoNames 검색 |
| 정답 | 수동 라벨링 (235개) | business.lat/lon (자동!) |
| GCN | @mention 인접행렬 | friendship 인접행렬 |
| Gravity | Pop/Dist^β | Pop/Dist^β (동일) |

---

## 7. 주의사항 & 잠재적 문제

### 메모리
- Yelp 유저가 GeoText보다 10배+ 많음
- review 텍스트가 tweet보다 10배+ 길음
- TF-IDF sparse matrix가 매우 클 수 있음
- **해결**: `mindf` 높이기 (10→50), `celebrity_threshold` 높이기, 유저 수 제한

### 그래프 밀도
- Yelp friendship은 @mention보다 sparse할 수 있음
- GCN은 연결이 없으면 작동하지 않음
- **해결**: friendship이 0인 유저 제외, giant component만 사용

### 도시명 모호성
- Yelp가 11개 metro area에 집중되어 있어서 모호 도시가 적을 수 있음
- "Phoenix"는 거의 AZ만 있을 수 있음
- **해결**: Step 0의 결과를 보고 판단. 부족하면 GeoNames 대신 US Census의 모든 도시를 후보로 사용

### Theano/Lasagne
- 기존 코드가 Theano 기반이라 환경 설정 주의
- Python 3.8 이하 + Theano 1.0.5 + Lasagne 0.2 필요
- 기존 서버에 이미 설치되어 있으면 그대로 사용

---

## 8. 최종 논문 프레이밍

두 실험의 결과를 아래와 같이 프레이밍:

> "Twitter의 @mention 그래프뿐 아니라, Yelp의 friendship 그래프에서도 
> 동일한 GCN + Gravity 프레임워크가 작동함을 보임으로써, 
> **플랫폼에 독립적인 일반화 가능성**을 검증하였다."

**기대 결과 테이블**:

| 데이터셋 | Baseline (Pop Only) | GCN + Gravity |
|---------|--------------------:|-------------:|
| GeoText (Twitter) | 88.51% | 92.34% |
| Yelp | ??.??% | ??.??% |
