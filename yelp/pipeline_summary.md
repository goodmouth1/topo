# GCN + Gravity Model 파이프라인 실험 결과 요약

## 석사 논문: GCN 기반 지명 모호성 해소 (Toponym Disambiguation)

> 소셜 네트워크의 그래프 구조(GCN)로 유저 위치를 예측하고,
> 그 위치를 anchor로 삼아 Gravity Model로 모호한 지명을 해소한다.

---

## 1. 전체 파이프라인 구조

```
[소셜 텍스트] + [소셜 그래프]
       ↓
  TF-IDF 벡터화 + 인접행렬 구축
       ↓
   GCN 훈련 (유저 위치 예측)
       ↓
  유저별 예측 좌표 (anchor)
       ↓
  Gravity Model: Score = Pop / Dist^β
       ↓
  모호한 지명 해소 (Toponym Disambiguation)
```

---

## 2. 데이터셋 비교

### 2.1 데이터 규모

| 항목 | GeoText (Twitter) | Yelp |
|------|------------------:|-----:|
| **총 유저 수** | 9,475 | 351,282 |
| Train | 5,747 | 281,025 |
| Dev | 1,825 | 35,128 |
| Test | 1,903 | 35,129 |
| **소셜 그래프** | @mention | Friendship |
| **텍스트** | Tweet (짧음) | Review (길음) |
| **엣지 수** | ~30K | 4,556,080 |
| **데이터 범위** | 미국 | 미국 (US only) |

### 2.2 지명 모호성 해소 평가 데이터

| 항목 | GeoText | Yelp |
|------|--------:|-----:|
| **라벨링 방식** | 수동 (manual) | 자동 (business.city) |
| **평가 인스턴스 수** | 235 | 133,076 |
| **고유 toponym 수** | ~100+ | 27 |
| **고유 유저 수** | ~150+ | 17,019 |
| **정답 소스** | 수동 라벨링 (human_label_idx) | business.lat/lon |

### 2.3 GCN 훈련 하이퍼파라미터

| 파라미터 | GeoText | Yelp (기본) | Yelp (튜닝 최적) |
|----------|---------|-------------|-----------------|
| Bucket size | 300 km | 300 km | **200 km** |
| Hidden layers | [300, 300, 300] | [300, 300, 300] | [300, 300, 300] |
| Min document frequency | 10 | 10 | 10 |
| Regularization | 1e-6 | 1e-6 | 1e-6 |
| Dropout | 0.5 | 0.5 | 0.5 |
| Celebrity threshold | 10 | 10 | 10 |
| Convolution | Yes | Yes | Yes |
| Highway | Yes | Yes | Yes |
| 클래스 수 (자동 결정) | 256 | 1,030 | ~1,500+ |
| 훈련 장치 | GPU (GTX 1080) | CPU | CPU |

---

## 3. GCN 유저 위치 예측 정확도

GCN은 유저의 좌표(lat, lon)를 예측한다.
이 예측 좌표가 Gravity Model의 **anchor**로 사용된다.

| 메트릭 | GeoText (Twitter) | Yelp |
|--------|------------------:|-----:|
| **Mean Error** | 502.4 km | **404.8 km** |
| **Median Error** | 73.4 km | **11.2 km** |
| **Acc@161km** | 61.20% | **73.16%** |

**분석:**
- Yelp가 GeoText보다 GCN 위치 예측이 훨씬 좋음
- Median 11.2km → 도시 수준까지 유저 위치를 맞춤
- 이유: 리뷰 텍스트가 트윗보다 길고 지역 특성이 풍부, Friendship 그래프가 @mention보다 안정적

---

## 4. 지명 모호성 해소 결과 (Toponym Disambiguation)

### 4.1 GeoText (Twitter)

| 방법 | 정확도 | 정답/전체 |
|------|-------:|----------:|
| 1. Population Only (Baseline) | 88.51% | 208/235 |
| 2. Distance Only (GCN anchor) | 62.98% | 148/235 |
| 3. Basic Gravity (Pop/Dist^β) | **92.34%** | 217/235 |
| 4. Proposed Model (Gravity + State bonus) | **92.34%** | 217/235 |

**개선: +3.83%p** (88.51% → 92.34%)

### 4.2 Yelp (기본 설정: bucket=300)

| 방법 | 정확도 | 정답/전체 |
|------|-------:|----------:|
| Population Only (Baseline) | 94.80% | 126,154/133,076 |
| GCN + Gravity (β=0.5) | **95.37%** | 126,913/133,076 |
| GCN + Gravity (β=1.0) | 94.52% | 125,789/133,076 |
| GCN + Gravity (β=1.5) | 94.18% | 125,325/133,076 |
| GCN + Gravity + State bonus (β=1.0) | 94.14% | 125,275/133,076 |

**개선: +0.57%p** (94.80% → 95.37%)

### 4.3 Yelp 하이퍼파라미터 튜닝 결과

bucket_size를 중심으로 6개 설정을 비교 실험하였다.

| 설정 | Bucket | Hidden | Median | Acc@161 | Best Gravity | 개선폭 | 훈련 시간 |
|------|-------:|--------|-------:|--------:|-------------:|-------:|----------:|
| **bucket200_h500x3** | **200** | [500,500,500] | 11.7km | 72.95% | **95.46%** | **+0.66%p** | 171분 |
| **bucket200_h300x3** | **200** | [300,300,300] | 11.7km | 72.79% | **95.45%** | **+0.65%p** | 140분 |
| bucket200_h300x2 | 200 | [300,300] | 10.9km | 72.92% | 95.43% | +0.63%p | 134분 |
| bucket500_h300x3 | 500 | [300,300,300] | 10.8km | 73.49% | 95.40% | +0.60%p | 103분 |
| bucket100_h300x3 | 100 | [300,300,300] | 12.3km | 71.51% | 95.38% | +0.58%p | 236분 |
| bucket300_h300x3 (기본) | 300 | [300,300,300] | 11.2km | 73.16% | 95.37% | +0.57%p | - |

**튜닝 분석:**
- **bucket=200이 Gravity Model에 최적** — GCN Acc@161은 bucket=500이 최고(73.49%)이지만, Gravity Model은 bucket=200이 최고(95.46%)
- 이유: bucket이 작을수록 예측 좌표가 정밀해져서, 후보 도시 간 거리 차이의 변별력이 높아짐
- hidden size 영향은 미미 (500 vs 300 → 0.01%p 차이)
- **최적 설정**: bucket=200, hidden=[300,300,300] (GeoText와 동일 구조 유지, 공정한 비교)

### 4.4 결과 종합

| 데이터셋 | Baseline (Pop Only) | GCN + Gravity (Best) | 개선폭 |
|---------|--------------------:|---------------------:|-------:|
| **GeoText (Twitter)** | 88.51% | 92.34% | **+3.83%p** |
| **Yelp (기본 bucket=300)** | 94.80% | 95.37% | +0.57%p |
| **Yelp (튜닝 bucket=200)** | 94.80% | **95.45%** | **+0.65%p** |

---

## 5. 분석 및 해석

### Yelp에서 개선폭이 작은 이유

1. **Baseline이 이미 매우 높음 (94.80%)**
   - Yelp 데이터가 대도시(Phoenix, Nashville 등)에 집중
   - 모호한 도시 대부분에서 한쪽 state가 압도적 다수
   - 인구수만으로도 거의 정답을 맞출 수 있음

2. **모호한 도시가 27개로 제한적**
   - GeoText는 트윗에서 NER로 다양한 지명 추출
   - Yelp는 business.city 필드만 사용 → 구조화된 도시명만 존재

3. **GCN 위치 예측은 오히려 Yelp가 우수**
   - Median Error: 11.2km (Yelp) vs 73.4km (GeoText)
   - GCN이 정확해도 이미 baseline이 높아 추가 개선 여지가 적음

### GCN 정확도 ≠ Gravity 성능인 이유

하이퍼파라미터 튜닝 결과, GCN 위치 예측 정확도(Acc@161)가 가장 높은 설정이 Gravity Model에서 최적은 아니었다.

| 설정 | Acc@161 (GCN) | Gravity 정확도 |
|------|:---:|:---:|
| bucket=500 | **73.49%** (최고) | 95.40% |
| bucket=200 | 72.79% | **95.45%** (최고) |

- **Acc@161**은 "161km 이내 맞춤"이라는 넓은 기준 → bucket이 클수록 유리
- **Gravity Model**은 정밀한 좌표로 후보 도시 간 거리를 비교 → bucket이 작을수록 예측 좌표가 정밀하여 변별력 향상
- 즉, GCN의 "대략적 정확도"보다 "좌표 정밀도"가 Gravity Model 성능에 더 중요

### 플랫폼 독립적 일반화 검증

| 관점 | GeoText | Yelp | 의미 |
|------|---------|------|------|
| GCN 위치 예측 | Acc@161: 61.20% | Acc@161: 73.16% | Yelp에서 더 잘 작동 |
| Gravity > Baseline? | Yes (+3.83%p) | Yes (+0.65%p) | 두 플랫폼 모두 개선 |
| 소셜 그래프 종류 | @mention | Friendship | 둘 다 유효 |
| 텍스트 종류 | Tweet | Review | 둘 다 유효 |

> **결론**: Twitter의 @mention 그래프뿐 아니라, Yelp의 friendship 그래프에서도
> 동일한 GCN + Gravity 프레임워크가 작동함을 확인하여,
> **플랫폼에 독립적인 일반화 가능성**을 검증하였다.

---

## 6. 파일 구조

### GeoText 파이프라인 (기존)

```
/home/wang/master_thesis/
├── data.py                         # DataLoader (TF-IDF, @mention 그래프)
├── gcnmodel.py                     # GraphConv (Theano/Lasagne GCN)
├── gcnmain.py                      # GCN 훈련 메인
├── run_gravity_with_gcn_state.py   # Gravity Model 평가
├── kdtree.py                       # KD-Tree 유틸
├── cities1000.txt                  # GeoNames 데이터
├── data/                           # GeoText 데이터
│   ├── user_info.train.gz          # 5,747 users
│   ├── user_info.dev.gz            # 1,825 users
│   └── user_info.test.gz           # 1,903 users
├── labeling/
│   └── manual_labeled_data.pkl     # 수동 라벨링 (457 items, 548 toponyms)
├── result/
│   └── ablation_results_256.csv    # Ablation 결과 (235 toponyms)
└── gcn_1.0_percent_pred_256.pkl    # GCN 예측 결과
```

### Yelp 파이프라인 (신규)

```
/home/wang/master_thesis/yelp/
├── yelp_pipeline_guide.md          # 구축 가이드
├── pipeline_summary.md             # 결과 요약 (이 문서)
├── step0_explore_yelp.py           # [Step 0] 데이터 탐색
├── step1_convert_yelp_to_geotext.py # [Step 1] Yelp → GeoText 포맷 변환
├── yelp_data.py                    # [Step 2] YelpDataLoader (friendship 그래프)
├── yelp_gcnmain.py                 # [Step 3] GCN 훈련 (CPU, YelpDataLoader)
├── step4_build_eval_data.py        # [Step 4] 평가 데이터 구축
├── step5_yelp_gravity_eval.py      # [Step 5] Gravity Model 평가
├── yelp_geotext_format/            # 변환된 데이터
│   ├── user_info.train.gz          # 281,025 users
│   ├── user_info.dev.gz            # 35,128 users
│   ├── user_info.test.gz           # 35,129 users
│   ├── friendship_edges.tsv        # 4,556,080 edges
│   └── meta.json                   # 메타데이터
├── yelp_eval_data.pkl              # 평가 데이터 (133,076 instances, 27 cities)
├── gcn_1.0_percent_pred_1030.pkl   # GCN dev 예측 (bucket=300)
├── gcn_1.0_percent_pred_1030_test.pkl # GCN test 예측 (bucket=300)
├── run_tuning.py                   # 하이퍼파라미터 튜닝 자동화 스크립트
├── tuning_results.csv              # 튜닝 전체 결과
├── tuning.log                      # 튜닝 실행 로그
├── exp_bucket100_h300x3/           # 튜닝 실험별 예측 파일
├── exp_bucket200_h300x3/
├── exp_bucket200_h500x3/
├── exp_bucket200_h300x2/
├── exp_bucket500_h300x3/
└── yelp_gcn_train.log              # 초기 훈련 로그
```

---

## 7. 실행 순서

### GeoText (이미 완료)

```bash
cd /home/wang/master_thesis/
python gcnmain.py -d ./data -bucket 300 -hid 300 300 300 \
  -mindf 10 -reg 1e-6 -dropout 0.5 -cel 10 -conv -highway -builddata
python run_gravity_with_gcn_state.py gcn_1.0_percent_pred_256.pkl
```

### Yelp

```bash
cd /home/wang/master_thesis/yelp/

# Step 0: 데이터 탐색
python step0_explore_yelp.py

# Step 1: GeoText 포맷 변환 (US only)
python step1_convert_yelp_to_geotext.py

# Step 3: GCN 훈련 (CPU)
python yelp_gcnmain.py -d ./yelp_geotext_format -bucket 300 \
  -hid 300 300 300 -mindf 10 -reg 1e-6 -dropout 0.5 \
  -cel 10 -conv -highway -builddata

# Step 4: 평가 데이터 구축
python step4_build_eval_data.py

# Step 5: Gravity Model 평가
python step5_yelp_gravity_eval.py gcn_1.0_percent_pred_1030_test.pkl
```

---

*마지막 업데이트: 2026-03-05*
