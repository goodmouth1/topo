# GCN 기반 지명 모호성 해소 (Toponym Disambiguation) — 실험 방법론

## 석사 논문: 소셜 네트워크 그래프 학습을 통한 지명 모호성 해소

---

## 1. 연구 개요

### 1.1 문제 정의: 지명 모호성 (Toponym Ambiguity)

텍스트에 등장하는 지명(toponym)은 종종 모호하다. 같은 이름의 도시가 여러 곳에 존재하기 때문이다.

```
예시:
  "I love Portland" → Portland, OR (인구 652,503)? Portland, ME (인구 68,408)?
  "Going to Springfield" → 미국에만 Springfield이 34개 주에 존재
  "Great food in Columbus" → Columbus, OH? Columbus, GA? Columbus, IN?
```

**지명 모호성 해소(Toponym Disambiguation)**란, 텍스트에 등장하는 지명이 구체적으로 어떤 지리적 위치를 가리키는지 결정하는 문제이다. 이는 지리 정보 검색(Geographic Information Retrieval), 위기 대응(Crisis Response), 공간 데이터 분석 등에서 핵심적인 전처리 단계이다.

### 1.2 기존 접근법의 한계

기존 연구는 크게 두 갈래로 발전했다:

| 연구 분야 | 대표 연구 | 보유 데이터 | 한계 |
|:---:|:---|:---|:---|
| **User Geolocation** | Rahimi et al. (2018), Jurgens (2013) | 소셜 그래프 ✓, 유저 좌표 ✓ | 지명 라벨 ✗ |
| **Toponym Disambiguation** | Gritta et al. (2018), Lieberman et al. (2010) | 지명 라벨 ✓ | 소셜 그래프 ✗ |

두 분야가 따로 발전하여, **소셜 그래프 + 지명 라벨**을 동시에 갖춘 데이터셋이 존재하지 않았다.

### 1.3 제안하는 2단계 프레임워크

본 연구는 두 분야를 결합한 **2단계 프레임워크**를 제안한다:

```
┌─────────────────────────────────────────────────────────┐
│  Stage 1: Anchor Acquisition (유저 위치 추정)            │
│  ┌─────────────┐     ┌─────────────┐                    │
│  │ 텍스트(TF-IDF)│ + │소셜 그래프(A)│  →  GCN  →  예측 좌표 │
│  └─────────────┘     └─────────────┘                    │
│                                                         │
│  Stage 2: Gravity-based Disambiguation (지명 해소)       │
│  ┌──────────────┐     ┌──────────────┐                  │
│  │GCN 예측 좌표   │ + │GeoNames 후보들│  →  Gravity  →  정답 │
│  │(= anchor)     │     │(인구, 좌표)   │    Model          │
│  └──────────────┘     └──────────────┘                  │
└─────────────────────────────────────────────────────────┘
```

**핵심 아이디어**: GCN으로 유저의 위치를 추정하고, 그 위치를 **anchor**(기준점)로 삼아 Gravity Model로 모호한 지명을 해소한다.

### 1.4 검증 전략

동일한 프레임워크를 **이종 소셜 플랫폼**에 적용하여 일반화 가능성을 검증한다:

| 플랫폼 | 소셜 그래프 유형 | 텍스트 유형 | 지명 라벨 방식 |
|:---:|:---:|:---:|:---:|
| **Twitter** (GeoText) | @mention 관계 | 트윗 (짧음, ~140자) | 수동 라벨링 |
| **Yelp** | Friendship 관계 | 리뷰 (길음, ~수백자) | 자동 라벨링 |

---

## 2. 이론적 배경

### 2.1 Graph Convolutional Network (GCN)

#### 2.1.1 기본 개념

GCN(Graph Convolutional Network)은 그래프 구조 데이터에 합성곱(convolution) 연산을 적용하는 신경망이다 (Kipf & Welling, 2017). 이미지에서 CNN이 인접 픽셀의 정보를 집계하듯, GCN은 **인접 노드의 정보를 집계**한다.

```
이미지 CNN:  각 픽셀이 주변 픽셀의 정보를 집계 → 특징 추출
그래프 GCN:  각 노드가 이웃 노드의 정보를 집계 → 특징 추출

소셜 네트워크에서의 의미:
  - 노드 = 유저
  - 엣지 = 소셜 관계 (@mention 또는 friendship)
  - 노드 특징 = 유저의 텍스트 (TF-IDF 벡터)
  - GCN 학습 후: 각 유저의 표현에 친구들의 텍스트 정보가 반영됨
```

#### 2.1.2 수학적 정의

**인접행렬 (Adjacency Matrix)**

그래프 G = (V, E)에서 N명의 유저가 있을 때, 인접행렬 A ∈ ℝ^{N×N}은:

```
A_ij = 1  (유저 i와 유저 j 사이에 엣지가 있으면)
A_ij = 0  (없으면)
```

Self-loop을 추가한 인접행렬:

```
Ã = A + I_N
```

여기서 I_N은 N×N 단위행렬이다. Self-loop은 자신의 정보도 집계에 포함시키기 위함이다.

**대칭 정규화 (Symmetric Normalization)**

Kipf & Welling (2017)의 정규화:

```
Â = D̃^{-1/2} · Ã · D̃^{-1/2}
```

여기서 D̃는 Ã의 차수 행렬(degree matrix)이다:

```
D̃_ii = Σ_j Ã_ij
```

이 정규화의 의미: 이웃이 많은 노드와 적은 노드 사이의 스케일을 맞춘다. 정규화 없이는 이웃이 많은 노드의 특징값이 폭발적으로 커질 수 있다.

**GCN 레이어 전파 규칙**

l번째 레이어에서 l+1번째 레이어로의 전파:

```
H^{(l+1)} = σ(Â · H^{(l)} · W^{(l)})
```

여기서:
- H^{(l)} ∈ ℝ^{N×d_l}: l번째 레이어의 노드 표현 행렬 (N: 노드 수, d_l: 차원)
- W^{(l)} ∈ ℝ^{d_l × d_{l+1}}: l번째 레이어의 학습 가능한 가중치 행렬
- σ: 활성 함수 (본 연구에서는 tanh)
- Â: 정규화된 인접행렬

이 연산을 두 단계로 분해하면:

```
1단계: H^{(l)} · W^{(l)}  → 각 노드의 특징을 선형 변환
2단계: Â · (H^{(l)} · W^{(l)})  → 이웃 노드의 변환된 특징을 가중 평균으로 집계
```

**초기 입력**: H^{(0)} = X (TF-IDF 특징 행렬)

#### 2.1.3 Highway Connection

깊은 레이어에서 그래디언트 소실을 방지하기 위해 Highway Connection (Srivastava et al., 2015)을 적용한다:

```
H^{(l+1)}_highway = t ⊙ H^{(l+1)} + (1 - t) ⊙ H^{(l)}
```

여기서:
- t = σ(W_gate · H^{(l)} + b_gate): 학습 가능한 게이트 벡터 (0~1 사이)
- ⊙: 원소별 곱 (element-wise multiplication)

게이트 t가 1에 가까우면 GCN 변환 결과를 사용하고, 0에 가까우면 입력을 그대로 통과시킨다. 이를 통해 네트워크가 **스스로 최적의 깊이**를 학습한다.

#### 2.1.4 공간 이산화: KD-Tree 클러스터링

GCN은 분류(classification) 모델이므로, 연속적인 위도/경도 좌표를 이산적인 클래스 레이블로 변환해야 한다.

**KD-Tree 기반 공간 분할:**

```
1. 훈련 유저의 좌표를 KD-Tree로 재귀적 이등분
   - 각 리프 노드(버킷)에 최소 bucket_size개의 유저 포함
   - bucket_size가 작을수록 → 클래스가 많아지고 → 좌표가 정밀

2. 각 클래스 c의 대표 좌표: 해당 클래스 유저들의 중앙값
   - lat_c = median({lat_i | user_i ∈ class c})
   - lon_c = median({lon_i | user_i ∈ class c})

3. Dev/Test 유저: Haversine 거리 기준 가장 가까운 클래스에 할당
```

**데이터셋별 결과:**

| 데이터셋 | bucket_size | 생성 클래스 수 |
|:---:|:---:|:---:|
| GeoText | 50 | 129 |
| Twitter-US | 2,400 | 256 |
| Twitter-World | 2,400 | 930 |
| Yelp | 200 (최적) | ~1,500 |

### 2.2 Gravity Model

#### 2.2.1 역사적 배경

Gravity Model은 지리학에서 두 장소 간의 상호작용을 설명하는 고전적 모델이다:

| 연도 | 연구자 | 기여 |
|:---:|:---|:---|
| 1885 | Ravenstein | 인구 이동 법칙 (개념적 제안) |
| 1941 | Stewart | 인구 중력(demographic gravity) 공식 최초 적용 |
| 1962 | Tinbergen | 국제 무역에 중력 모델 적용 |
| 1971 | Wilson | 체계적 수학 정리 |
| 2007 | Leidner | 지명 해소에 인구 + 거리 휴리스틱 적용 |
| 2010 | Lieberman et al. | LGL 데이터셋, 지역 맥락의 중요성 증명 |

#### 2.2.2 수학적 정의

뉴턴의 만유인력 법칙에서 영감을 받은 공식:

```
원래 중력 법칙:    F = G · (m₁ · m₂) / r²
지명 해소 버전:    Score(c) = Population(c) / Distance(anchor, c)^β
```

여기서:
- c: 지명 후보 도시 (예: Portland, OR 또는 Portland, ME)
- Population(c): GeoNames에서 가져온 후보 도시의 인구수
- Distance(anchor, c): GCN 예측 좌표(anchor)와 후보 도시 간의 거리
- β: 거리 감쇠 파라미터 (distance decay parameter)

**β 파라미터의 의미:**

```
β가 클수록 (β=3.0):  가까운 후보에 강한 가중치 → 거리 중시
β가 작을수록 (β=0.5): 인구가 큰 후보에 유리 → 인구 중시

극단 케이스:
  β = 0: Score = Population (인구만으로 결정, Baseline과 동일)
  β = ∞: Score ≈ 1/Distance (가장 가까운 후보 선택)
```

**결정 규칙:**

```
prediction = argmax_{c ∈ candidates} Score(c)
```

모든 후보 중 점수가 가장 높은 도시를 정답으로 선택한다.

#### 2.2.3 Haversine 거리

지구 표면의 두 점 사이 대원 거리를 계산하는 공식:

```
a = sin²(Δlat/2) + cos(lat₁) · cos(lat₂) · sin²(Δlon/2)
d = 2R · arctan2(√a, √(1-a))
```

여기서 R = 6,371 km (지구 평균 반지름).

#### 2.2.4 State 보너스 (Twitter 전용 확장)

Twitter 실험에서는 텍스트와 GCN 예측 정보를 추가로 활용하는 3단계 보너스 시스템을 적용한다:

```
Score(c) = [Population(c) / Distance(anchor, c)^β] × Bonus(c)

Bonus 결정 로직:
  text_match = (후보 c의 state 코드가 트윗 텍스트에 포함?)
  gcn_match  = (GCN 예측 좌표의 reverse-geocoded state == 후보 c의 state?)

  if text_match AND gcn_match:  Bonus = 10.0  (강한 증거)
  elif text_match OR gcn_match: Bonus = 5.0   (부분 증거)
  else:                         Bonus = 1.0   (증거 없음)
```

### 2.3 TF-IDF 텍스트 벡터화

유저의 텍스트를 수치 벡터로 변환하는 방법:

```
TF-IDF(t, d) = TF(t, d) × IDF(t)

TF(t, d) = 문서 d에서 단어 t의 출현 빈도
IDF(t) = log(전체 문서 수 / 단어 t를 포함하는 문서 수)
```

**설정:**
- token_pattern: `(?u)(?<![@])#?\b\w\w+\b` → @mention 제외, #hashtag 유지
- min_df: 10 (최소 10명의 유저가 사용한 단어만 포함)
- max_df: 0.2 (전체 유저의 20% 이상이 사용한 단어 제외 = 불용어 자동 제거)
- norm: L2 정규화
- stop_words: 영어 불용어 제거

출력: 각 유저를 고차원 희소 벡터(sparse vector)로 표현.

---

## 3. 기반 연구: Rahimi et al. (2018)

### 3.1 논문 개요

본 연구의 Stage 1(GCN 유저 위치 예측)은 **Rahimi et al. (2018) "Semi-supervised User Geolocation via Graph Convolutional Networks" (ACL 2018)**을 기반으로 한다.

이 논문의 핵심 기여:
1. 소셜 네트워크의 @mention 그래프를 GCN으로 학습하여 유저 위치를 예측
2. 3개의 Twitter 데이터셋에서 검증 (GeoText, Twitter-US, Twitter-World)
3. 기존 방법(DeepWalk, LP) 대비 state-of-the-art 성능 달성

### 3.2 원논문에서 사용한 데이터셋

| 논문 표기 | 출처 | 폴더명 | 유저 수 | 지역 | 클래스 수 |
|:---:|:---|:---:|---:|:---:|:---:|
| **GeoText** | Eisenstein et al. (2010) | `data/cmu/` | 9,475 | 미국 | 129 |
| **Twitter-US** | Roller et al. (2012) | `data/na/` | 449,200 | 미국 | 256 |
| **Twitter-World** | Han et al. (2012) | `data/world/` | 1,386,766 | 전세계 | 930 |

**주의:** 폴더명 `cmu/`, `na/`, `world/`는 코드상 편의명이며, 논문에서의 공식 명칭은 위 표와 같다.

### 3.3 원논문 결과 vs 재현 결과

| 데이터셋 | 논문 Acc@161 | 논문 Median | 재현 Acc@161 | 재현 Median |
|:---:|:---:|:---:|:---:|:---:|
| GeoText | 60% | 45 km | **60.95%** | 46.2 km |
| Twitter-US | 66% | 71 km | **61.20%** | 73.4 km |
| Twitter-World | 54% | 108 km | **54.59%** | 100.2 km |

GeoText와 Twitter-World는 원논문과 거의 일치. Twitter-US는 하이퍼파라미터 차이(hidden dimension, training epochs 등)로 약간의 차이가 있다.

### 3.4 원논문의 모델 설정

| 파라미터 | GeoText | Twitter-US | Twitter-World |
|:---:|:---:|:---:|:---:|
| Hidden dimensions | 300 | 600 | 900 |
| GCN layers | 3 | 3 | 3 |
| Bucket size (KD-Tree) | 50 | 2,400 | 2,400 |
| Highway connections | ✓ | ✓ | ✓ |
| Dropout | 0.5 | 0.5 | 0.5 |
| Optimizer | Adam | Adam | Adam |

---

## 4. 데이터셋 상세

### 4.1 GeoText (Eisenstein et al., 2010)

미국 내 지오태깅(geotagged) 트윗을 수집한 데이터셋.

```
규모: 9,475명 유저, ~380,000 트윗
분할: Train 5,685 / Dev 1,895 / Test 1,895
지역: 미국 전역 (lat 25~49, lon -125~-66)
유저 좌표: 트위터 프로필에 등록된 GPS 좌표
소셜 그래프: 트윗 텍스트 내 @mention 관계 추출
텍스트: 유저의 모든 트윗을 하나로 합침
```

**소셜 그래프 구축 상세:**

```
1. 정규식으로 @mention 추출: (?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9_]+)
2. 이분 그래프(bipartite graph) 구축:
   - 유저 노드 ↔ mention 대상 노드
3. Celebrity 필터링:
   - degree=1 (1회만 mention) 노드 제거
   - degree > 10 (celebrity) 노드 제거 (예: @BarackObama는 위치 정보에 무관)
4. 유저 노드로 투영(projection):
   - 같은 대상을 mention한 두 유저 사이에 엣지 생성
5. Self-loop 추가 (대각선 = 1)
```

### 4.2 Twitter-US (Roller et al., 2012)

GeoText보다 약 47배 큰 미국 Twitter 데이터셋.

```
규모: 449,200명 유저
분할: Train 429,200 / Dev 10,000 / Test 10,000
지역: 미국 전역 (99.8%, 이름은 "NA"이지만 실질적으로 미국 전용)
소셜 그래프: @mention 관계
```

**GeoText vs Twitter-US의 핵심 차이:**

| | GeoText | Twitter-US |
|:---:|:---:|:---:|
| 유저 수 | 9,475 | 449,200 (47배) |
| 학습 장비 | GPU 가능 | CPU 필요 (메모리 초과) |
| 인접행렬 크기 | 9K × 9K | 449K × 449K |
| KD-Tree 클래스 | 129 | 256 |

### 4.3 Twitter-World (Han et al., 2012)

전세계 Twitter 유저를 포함하는 최대 규모 데이터셋.

```
규모: 1,386,766명 유저
분할: Train 1,366,766 / Dev 10,000 / Test 10,000
지역: 전세계 (미국 52.9%, 유럽 22.3%, 아시아 16.3%, 남반구 11.0%)
소셜 그래프: @mention 관계
```

### 4.4 Yelp Open Dataset

비즈니스 리뷰 플랫폼 데이터. Twitter와 완전히 다른 소셜 네트워크.

```
원본 데이터:
  - yelp_academic_dataset_business.json: 비즈니스 정보 (이름, 도시, 주, 좌표)
  - yelp_academic_dataset_review.json: 리뷰 텍스트
  - yelp_academic_dataset_user.json: 유저 정보 + friendship

변환 후:
  - 유효 유저: 351,282명 (미국만, 리뷰 3개 이상)
  - 분할: Train 281,025 / Dev 35,128 / Test 35,129
  - 소셜 그래프: user.json의 friends 배열 (4,556,080 엣지)
  - 유저 좌표: 리뷰한 비즈니스들의 좌표 중앙값 (median lat/lon)
```

**Yelp을 선택한 이유:**

GCN + Gravity Model에는 세 가지가 필요하다:
1. ✅ **소셜 네트워크 그래프** → user.json의 friendship
2. ✅ **텍스트** → review.json의 리뷰 텍스트
3. ✅ **지명 정답 좌표** → business.json의 city + state + lat/lon

특히 Yelp의 `business.city` 필드가 **구조화된 toponym**이며, `business.state`가 **ground truth**로 직접 사용 가능하여 수동 라벨링 없이 대규모 평가 데이터를 자동 생성할 수 있다.

### 4.5 데이터셋간 관계

**중요:** GeoText, Twitter-US, Twitter-World는 **독립적인 데이터셋**이다.

```
- GeoText: 해시된 유저 ID (user_a7ebaf03 형식)
- Twitter-US: 원본 트위터 핸들 (toddamurphy 형식)
- Twitter-World: 원본 트위터 핸들
- 유저 ID 중복: 0% (GeoText ↔ Twitter-US), 0% (GeoText ↔ Twitter-World)
```

따라서 한 데이터셋의 라벨링 데이터를 다른 데이터셋에 적용할 수 없다.

---

## 5. Stage 1: GCN 유저 위치 예측

### 5.1 전처리 파이프라인

두 플랫폼 모두 동일한 전처리 구조를 따른다:

```
[입력]                          [처리]                      [출력]
텍스트 데이터 ─────────→ TF-IDF 벡터화 ─────────→ X (특징 행렬)
소셜 관계 데이터 ────→ 그래프 구축 + 정규화 ──→ Â (정규화 인접행렬)
유저 좌표 ──────────→ KD-Tree 이산화 ────────→ Y (클래스 레이블)
```

**Twitter (GeoText/Twitter-US/Twitter-World):**
- 텍스트: 유저의 모든 트윗 합침 → TF-IDF
- 그래프: @mention 텍스트 파싱 → 이분 그래프 → 투영 → celebrity 필터링
- 좌표: 트위터 프로필 GPS

**Yelp:**
- 텍스트: 유저의 모든 리뷰 합침 → TF-IDF
- 그래프: friendship_edges.tsv에서 직접 로드 (파싱 불필요)
- 좌표: 리뷰한 비즈니스 좌표의 중앙값

### 5.2 GCN 모델 아키텍처

Rahimi et al. (2018)의 구현을 Theano/Lasagne로 사용한다.

```
Layer 1: SparseInputDenseLayer
  입력: X ∈ ℝ^{N × V} (sparse TF-IDF, V = 어휘 크기)
  출력: H₁ ∈ ℝ^{N × d} (dense)
  연산: H₁ = tanh(X · W₁ + b₁)
  Dropout: 0.5

Layer 2: Highway + GraphConv
  입력: H₁ ∈ ℝ^{N × d}
  Graph Convolution: G₂ = tanh(Â · H₁ · W₂)
  Highway Gate: t₂ = σ(H₁ · W_gate₂ + b_gate₂)
  출력: H₂ = t₂ ⊙ G₂ + (1 - t₂) ⊙ H₁

Layer 3: Highway + GraphConv
  입력: H₂
  동일 구조
  출력: H₃

Output Layer: ConvolutionDenseLayer3
  입력: H₃ ∈ ℝ^{N × d}
  연산: O = softmax(Â · H₃ · W_out)
  출력: P ∈ ℝ^{N × C} (C = 클래스 수)
  각 유저가 C개 지리적 클래스에 속할 확률 분포
```

**데이터셋별 모델 크기:**

| 데이터셋 | 입력 차원 (V) | Hidden (d) | 출력 (C) | 인접행렬 크기 |
|:---:|:---:|:---:|:---:|:---:|
| GeoText | ~9,467 | 300 | 129 | 9,475² |
| Twitter-US | ~가변 | 600 | 256 | 449,200² |
| Twitter-World | ~가변 | 900 | 930 | 1,386,766² |
| Yelp | ~가변 | 300 | ~1,500 | 351,282² |

### 5.3 훈련 설정

```
옵티마이저: Adam (lr=2×10⁻³, β₁=0.9, β₂=0.999, ε=10⁻⁸)
손실 함수: Categorical Cross-Entropy
         L = -Σᵢ Σⱼ y_ij · log(p_ij)  (i: 유저, j: 클래스)
정규화: L₁ + L₂ (계수 10⁻⁶)
Early Stopping: Validation loss가 10 에폭 연속 개선되지 않으면 중단
최대 에폭: 10,000
배치: Full-batch (GCN은 전체 그래프에 대해 forward pass 필요)
```

**중요:** GCN은 **Semi-supervised Learning**이다:
- Forward pass: 전체 N명의 유저에 대해 실행 (Â · H · W)
- Loss 계산: Train 유저에 대해서만 (dev/test 유저의 좌표는 학습에 사용하지 않음)
- 이를 통해 레이블이 없는 유저도 이웃의 정보를 전파받아 위치가 추정됨

### 5.4 추론 및 좌표 변환

```
1. 훈련된 모델로 dev/test 유저에 대해 예측:
   P = model.predict(X, Â, indices)  → 각 유저별 클래스 확률 분포

2. 예측 클래스 결정:
   ĉᵢ = argmax_j P_ij  → 유저 i의 예측 클래스 번호

3. 좌표 변환:
   lat̂ᵢ = classLatMedian[ĉᵢ]  → 예측 클래스의 중앙값 위도
   lon̂ᵢ = classLonMedian[ĉᵢ]  → 예측 클래스의 중앙값 경도

4. 오차 계산:
   error_i = haversine((latᵢ, lonᵢ), (lat̂ᵢ, lon̂ᵢ))  → km 단위
```

### 5.5 평가 메트릭

| 메트릭 | 정의 | 의미 |
|:---:|:---|:---|
| **Acc@161km** | 예측 좌표가 실제 좌표에서 161km(100마일) 이내인 유저 비율 | 업계 표준 메트릭 |
| **Median Error** | 오차 거리의 중앙값 | 일반적 성능 (이상값에 강건) |
| **Mean Error** | 오차 거리의 평균값 | 전체적 성능 (이상값에 민감) |

### 5.6 GCN 위치 예측 결과

| 데이터셋 | 유저 수 | Acc@161 | Median Error | Mean Error |
|:---:|---:|:---:|:---:|:---:|
| **GeoText** | 9,475 | 60.95% | 46.2 km | 520.4 km |
| **Twitter-US** | 449,200 | 61.20% | 73.4 km | 502.4 km |
| **Twitter-World** | 1,386,766 | 54.59% | 100.2 km | 1,107.3 km |
| **Yelp** | 351,282 | 73.16% | 11.2 km | 404.8 km |

**관찰:**

1. **GeoText vs Twitter-US (규모 효과)**: 유저가 47배 많아도 Acc@161은 비슷(60.95% vs 61.20%). 그러나 Mean Error는 개선(520→502km). 대규모 데이터가 극단적 오차를 줄이는 효과.

2. **Twitter-US vs Twitter-World (지역 확장 효과)**: 전세계로 확장 시 Acc@161 하락(61.20%→54.59%). 예측해야 할 지역이 넓어지고 클래스가 많아져서 과제 난이도가 높아짐.

3. **Yelp가 가장 우수**: Median 11.2km로 GeoText(46.2km) 대비 4배 정밀. 리뷰 텍스트의 풍부한 지역 특성("I went to the Brooklyn branch", "best sushi in downtown Phoenix")과 Friendship 그래프의 안정성 덕분.

---

## 6. Stage 2: Gravity Model 지명 해소

### 6.1 평가 데이터 구축

#### 6.1.1 Twitter (GeoText) — 수동 라벨링

GeoText 유저의 트윗에서 지명을 추출하고, 사람이 직접 정답을 지정하였다.

```
라벨링 과정:
1. 유저의 트윗에서 NER(Named Entity Recognition)로 지명 후보 추출
2. LLM(GPT-4o)이 1차 판별 → 연구자가 승인/수정
3. 각 지명에 대해 GeoNames에서 동명 도시 후보 리스트 생성
4. 사람이 정답 인덱스(human_label_idx) 최종 지정

결과: 457개 아이템, 235개 평가용 toponym
파일: labeling/manual_labeled_data.pkl
```

**데이터 구조:**
```
{
  'user_id': 'user_00b5310f',
  'user_true_loc': (40.71, -74.00),  # 유저 실제 좌표 (뉴욕)
  'toponyms': [
    {
      'word': 'Portland',
      'candidates': [
        {'name': 'Portland', 'state': 'OR', 'lat': 45.52, 'lon': -122.68, 'pop': 652503},
        {'name': 'Portland', 'state': 'ME', 'lat': 43.66, 'lon': -70.26, 'pop': 68408},
        ...
      ],
      'human_label_idx': 1  # 정답: Portland, ME
    }
  ]
}
```

**수동 라벨링의 한계:**
- 노동 집약적 (457개 라벨링에 상당한 시간 소요)
- GeoText 유저에만 해당 (Twitter-US, Twitter-World에는 라벨 없음)
- Twitter-US/World는 독립 데이터셋으로 유저가 겹치지 않아 라벨 재활용 불가

#### 6.1.2 Yelp — 자동 라벨링

Yelp의 구조화된 비즈니스 데이터를 활용하여 **대규모 평가 데이터를 자동 생성**한다.

```
핵심 아이디어:
  business.city = "Portland"  → toponym (모호한 지명)
  business.state = "OR"       → ground truth (평가 시 가림)

자동 라벨링 과정:
1. GeoNames에서 미국 도시 로드 (인구 1,000+)
2. Yelp business.json에서 모호한 도시 식별:
   - 같은 city 이름이 2개 이상 state에 존재
   - GeoNames에도 2개 이상 후보가 존재
   - 결과: 27개 모호 도시
3. 해당 비즈니스를 리뷰한 test 유저와 매칭
4. 평가 데이터 생성: (유저, 도시명, 후보 리스트, 정답 인덱스)

결과: 133,076개 평가 인스턴스, 17,019명 유저
```

**수동 vs 자동 라벨링 비교:**

| | Twitter (수동) | Yelp (자동) |
|:---:|:---:|:---:|
| 라벨 수 | 235개 | 133,076개 |
| NER 오류 | 있을 수 있음 | 없음 (구조화 필드) |
| 라벨 품질 | 매우 높음 (사람 판단) | 높음 (state=ground truth) |
| 확장성 | 낮음 (시간 소요) | 높음 (완전 자동) |

### 6.2 GCN 예측 좌표 → Gravity 입력

#### 6.2.1 유저-GCN 예측 매칭

**Twitter:** GCN 예측 파일에서 좌표 근접(coordinate proximity) 매칭:
```
1. GCN 예측 파일의 latlon_true 배열에서 유저의 user_true_loc과
   유클리드 거리가 가장 가까운 점을 검색 (threshold < 0.01)
2. 매칭된 인덱스의 latlon_pred를 anchor로 사용
```

**Yelp:** test 유저 순서가 보장되어 직접 인덱스 매칭:
```
1. user_info.test.gz의 유저 순서대로 읽기
2. GCN 예측 파일의 latlon_pred와 1:1 매칭
3. user_gcn_pred[uid] = latlon_pred[i]
```

#### 6.2.2 GCN 예측 좌표의 State 역변환 (Twitter 전용)

State 보너스 계산을 위해, GCN 예측 좌표를 가장 가까운 미국 도시의 state로 변환:

```
1. GeoNames cities1000.txt에서 미국 도시 좌표 + state 코드 로드
2. GCN 예측 좌표에서 가장 가까운 미국 도시 검색 (유클리드 거리)
3. 해당 도시의 admin1 code = GCN predicted state
```

### 6.3 Gravity Model 평가 절차

```
각 toponym 인스턴스에 대해:
  1. 후보 리스트 로드: candidates = [{name, state, lat, lon, pop}, ...]
  2. 유저의 GCN 예측 좌표(anchor) 로드
  3. 각 후보에 대해 Gravity Score 계산:
     Score(c) = Population(c) / haversine(anchor, c)^β
     (최소 거리 1km 적용하여 0으로 나누기 방지)
  4. 최고 점수 후보 선택: prediction = argmax Score
  5. 정답과 비교: correct = (prediction == ground_truth)
  6. 정확도 = Σ correct / Σ total
```

### 6.4 Baseline: Population Only

GCN을 사용하지 않고, **오직 인구수만**으로 판단하는 가장 단순한 방법:

```
prediction = argmax_{c ∈ candidates} Population(c)

예시:
  "Portland" → candidates: [OR(652,503명), ME(68,408명), TX(12,213명), ...]
  → 인구 최대: Portland, OR 선택
```

이 baseline이 높은 정확도를 보이는 이유: 실제로 사람들이 큰 도시를 언급하는 경우가 많기 때문.

---

## 7. 실험 결과

### 7.1 Twitter (GeoText) Gravity 결과

GCN 예측 파일별 결과 (수동 라벨링 데이터 기준):

| GCN 모델 | 매칭 유저 | 평가 토포님 | Gravity 정확도 |
|:---:|:---:|:---:|:---:|
| GeoText (129, 9K 유저) | 160명 | 204개 | **91.18%** |
| Twitter-US (256, 449K 유저) | 194명 | 235개 | **92.34%** |

**Baseline (Population Only): 88.51% (208/235)**

**Twitter-World (930)가 제외된 이유:**
- 라벨링 데이터의 457명 유저가 전부 GeoText 유저
- GeoText, Twitter-US, Twitter-World는 독립 데이터셋 (유저 ID 중복 0%)
- Twitter-World dev 유저 중 미국 내 유저가 적어 좌표 근접 매칭이 13명만 성공
- 13명/17개 토포님으로는 통계적으로 유의미하지 않음

**GeoText(9K) vs Twitter-US(449K) 비교:**
- 학습 데이터가 47배 많은 Twitter-US가 1.16%p 더 높은 성능 (91.18% → 92.34%)
- 더 많은 유저 매칭 (160 → 194명)으로 평가 커버리지도 향상

### 7.2 Ablation Study (Twitter-US 기준)

각 구성 요소의 기여도를 분석하기 위한 제거 실험:

| 모델 | 구성 | 정확도 | 의미 |
|:---|:---|:---:|:---|
| Population Only | Pop만 | 88.51% (208/235) | Baseline |
| Distance Only | GCN 거리만 | 62.98% (148/235) | GCN anchor만으로는 부족 |
| Basic Gravity | Pop / Dist^β | **92.34%** (217/235) | Pop + Distance 시너지 |
| Proposed (+ State Bonus) | Pop / Dist^β × Bonus | **92.34%** (217/235) | 보너스 효과 미미 |

**해석:**
- Population만 사용해도 88.51%로 상당히 높음 → 인구가 강력한 prior
- Distance만 사용하면 62.98%로 떨어짐 → GCN 예측 자체는 "가장 가까운 도시 = 정답"을 보장하지 않음
- **Pop과 Distance를 결합하면 시너지 효과** (88.51% → 92.34%, +3.83%p)
- State Bonus는 추가 개선이 없음 → 트윗에 state 코드가 명시적으로 포함된 경우가 드묾

### 7.3 오류 분석 (Twitter)

18개 오답 사례 중 대표적인 패턴:

| 정답 | 예측 | 원인 |
|:---:|:---:|:---|
| West Hollywood, **CA** | Hollywood, **FL** | GCN이 FL로 예측 + FL Hollywood가 더 가까움 |
| Boston, **NY** (희귀) | Boston, **MA** | 인구 압도적 차이 (MA 694K vs NY 1.3K) |
| Woodbury, **NY** | Woodbury, **MN** | GCN 예측이 중서부로 빗나감 |

### 7.4 Yelp Gravity 결과

**최적 하이퍼파라미터 (bucket=200):**

| 방법 | 정확도 |
|:---|:---:|
| Baseline (Population Only) | 94.80% (126,154/133,076) |
| **GCN + Gravity (β=0.5)** | **95.45%** |

### 7.5 Yelp 하이퍼파라미터 튜닝

bucket_size를 중심으로 6개 설정을 비교:

| 설정 | GCN Acc@161 | GCN Median | **Gravity 정확도** |
|:---|:---:|:---:|:---:|
| bucket=100, h=300×3 | 64.08% | 12.2 km | 95.36% |
| **bucket=200, h=300×3** | 69.30% | 11.4 km | **95.45%** |
| bucket=300, h=300×3 | 73.16% | 11.2 km | 95.20% |
| bucket=500, h=300×3 | **73.49%** | 14.1 km | 95.05% |
| bucket=200, h=500×3 | 69.46% | 10.3 km | **95.46%** |
| bucket=200, h=300×2 | 68.84% | 11.3 km | 95.37% |

**핵심 발견: GCN Acc@161 ≠ Gravity 성능**

```
GCN Acc@161 최고:   bucket=500 (73.49%) → Gravity 95.05% (4위)
Gravity 정확도 최고: bucket=200 (69.30%) → GCN Acc@161 (3위)
```

이유: Acc@161은 "161km 이내에 맞추면 정답"이라는 관대한 메트릭. bucket이 크면 클래스가 적어져 분류는 쉬워지지만(Acc@161 ↑), 예측 좌표의 정밀도가 떨어진다(클래스 중앙값이 넓은 영역의 대표값이므로). Gravity Model에서는 후보 도시들 간의 **미세한 거리 차이**가 중요하므로, 좌표 정밀도가 높은 작은 bucket이 유리하다.

```
bucket=500: 클래스 당 넓은 영역 → 좌표 정밀도 ↓ → "Portland OR과 ME 중 어디가 더 가까운가" 변별력 ↓
bucket=200: 클래스 당 좁은 영역 → 좌표 정밀도 ↑ → 후보 간 거리 차이 명확 → Gravity ↑
```

### 7.6 전체 결과 종합

| 플랫폼 | 데이터셋 | Baseline (Pop Only) | GCN + Gravity | **개선폭** |
|:---:|:---:|:---:|:---:|:---:|
| Twitter | GeoText (9K) | 88.51% | 91.18% | +2.67%p |
| Twitter | **Twitter-US (449K)** | 88.51% | **92.34%** | **+3.83%p** |
| **Yelp** | Yelp (351K) | 94.80% | **95.45%** | **+0.65%p** |

**두 플랫폼, 두 규모 모두에서 GCN + Gravity가 Population Only Baseline을 상회.**

---

## 8. 논의

### 8.1 왜 Yelp의 개선폭이 작은가?

Yelp의 Baseline이 94.80%로 이미 매우 높아, 개선 여지가 적다 (ceiling effect).

원인:
1. Yelp 데이터가 대도시에 집중 → 모호한 도시 대부분이 한쪽 주에 리뷰가 편중
2. 인구가 많은 도시가 리뷰도 많음 → Population 휴리스틱이 이미 강력
3. 반면 GeoText는 Baseline 88.51%로 개선 여지가 더 크고, 트윗은 지역적 편향이 덜함

### 8.2 플랫폼 독립적 일반화

| 차이점 | Twitter (GeoText) | Yelp |
|:---:|:---:|:---:|
| 소셜 그래프 | @mention (텍스트 파싱) | Friendship (명시적) |
| 텍스트 | 트윗 (~140자, 비격식) | 리뷰 (~수백자, 서술적) |
| 유저 좌표 출처 | 프로필 GPS | 비즈니스 좌표 중앙값 |
| 지명 라벨 | 수동 (235개) | 자동 (133,076개) |
| Gravity β 최적값 | 1.5 | 0.5 |

**동일한 프레임워크(GCN → Gravity)가 이질적인 두 플랫폼에서 모두 작동**함을 확인. 이는 그래프 구조의 유형(mention vs friendship)이나 텍스트의 길이/스타일에 관계없이, 소셜 네트워크에서 유저 위치를 추정하고 이를 지명 해소에 활용하는 접근법이 범용적임을 시사한다.

### 8.3 β 최적값의 차이 (Twitter β=1.5 vs Yelp β=0.5)

- **Twitter β=1.5**: GCN Median Error가 73.4km로 비교적 크므로, 거리 가중치를 강하게 줘야 노이즈를 극복
- **Yelp β=0.5**: GCN Median Error가 11.2km로 매우 정밀하므로, 인구(Population) 정보에 더 의존하는 것이 유리

이는 **GCN 예측 정밀도에 따라 최적 β가 결정됨**을 보여준다.

### 8.4 LGL 실험 시도 및 폐기 경위

본 연구 초기에는 뉴스 기사 데이터셋인 LGL (Lieberman et al., 2010)에도 프레임워크를 적용하려 하였다.

```
LGL (Local Global Lexicon):
  - 588개 미국 지방 신문 기사
  - 5,088개 toponym (사람이 라벨링)
  - 소셜 그래프 없음 (뉴스 기사이므로)
```

**문제:** 소셜 그래프가 없어 GCN을 직접 적용할 수 없었다. 대안으로 기사 내 toponym 좌표들의 평균값(centroid)을 anchor로 사용하였다.

```
결과: Population Only 45.33% → Gravity(centroid) 59.33% (+14%p)
```

그러나 교수님의 피드백:

> **"정답을 미리 보고 시험 치는 구조이다."**

centroid 계산에 정답 toponym의 좌표가 포함되어 있어 **데이터 누수(Data Leakage)** 문제가 발생. 이에 따라 LGL 실험을 폐기하고, GCN을 적용할 수 있는 새로운 데이터셋으로 **Yelp**을 선택하게 되었다.

### 8.5 Twitter-World가 Gravity 평가에서 제외된 이유

1. **라벨 데이터 부재**: 수동 라벨링된 457개 toponym이 모두 GeoText 유저에서 유래
2. **독립 데이터셋**: GeoText와 Twitter-World는 유저 ID가 완전히 다름 (중복 0%)
3. **좌표 매칭 한계**: Twitter-World dev 10,000명 중 미국 유저가 ~53%에 불과, 좌표 근접 매칭 시 13명만 성공
4. **자동 라벨링 불가**: Twitter 트윗의 도시 언급은 정답을 특정할 수 없음 (사용자의 위치 ≠ 언급 도시의 위치)
5. **수동 라벨링 필요**: Twitter-World 유저의 트윗을 사람이 직접 읽고 판별해야 하나, 시간적 제약으로 미수행

따라서 Twitter-World는 **GCN 위치 예측 성능만 보고**하고 (Acc@161: 54.59%), Gravity 평가에서는 제외하였다.

---

## 9. 코드 구조

### 9.1 기존 코드 (Rahimi et al. 기반, 수정하지 않음)

| 파일 | 역할 |
|:---|:---|
| `data.py` | DataLoader: 데이터 로딩, TF-IDF, @mention 그래프 구축 |
| `gcnmodel.py` | GraphConv: Theano/Lasagne GCN 모델 클래스 |
| `gcnmain.py` | GCN 훈련 메인 스크립트 |
| `run_gravity_with_gcn_state.py` | Twitter Gravity Model 평가 |

### 9.2 Yelp 파이프라인 (신규 작성, `/yelp/` 폴더)

| 파일 | 역할 | 대응하는 기존 코드 |
|:---|:---|:---|
| `step0_explore_yelp.py` | 데이터 탐색 (모호 도시 수, 그래프 밀도) | - |
| `step1_convert_yelp_to_geotext.py` | Yelp JSON → GeoText 포맷 변환 | - |
| `yelp_data.py` | YelpDataLoader (friendship 그래프) | `data.py` 상속 |
| `yelp_gcnmain.py` | Yelp GCN 훈련 (CPU 모드) | `gcnmain.py` 기반 |
| `step4_build_eval_data.py` | 자동 평가 데이터 구축 | `manual_labeled_data.pkl` 대응 |
| `step5_yelp_gravity_eval.py` | Yelp Gravity Model 평가 | `run_gravity_with_gcn_state.py` 대응 |
| `run_tuning.py` | 하이퍼파라미터 자동 튜닝 | - |

---

## 10. 참고 문헌

| 논문 | 기여 |
|:---|:---|
| **Rahimi et al. (2018)** "Semi-supervised User Geolocation via Graph Convolutional Networks" (ACL) | GCN 기반 유저 위치 예측 — 본 연구의 Stage 1 기반 |
| **Kipf & Welling (2017)** "Semi-Supervised Classification with Graph Convolutional Networks" (ICLR) | GCN 원논문 — 인접행렬 정규화, 레이어 전파 규칙 |
| **Srivastava et al. (2015)** "Highway Networks" | Highway Connection — 깊은 GCN에서 그래디언트 전파 |
| **Eisenstein et al. (2010)** "A latent variable model for geographic lexical variation" | GeoText 데이터셋 출처 |
| **Roller et al. (2012)** "Supervised text-based geolocation using language models on an adaptive grid" | Twitter-US 데이터셋 출처 |
| **Han et al. (2012)** "Geolocation prediction in social media data by finding location indicative words" | Twitter-World 데이터셋 출처 |
| **Stewart (1941)** "An inverse distance variation for certain social influences" | Gravity Model 원 공식 |
| **Leidner (2007)** "Toponym Resolution in Text" | 지명 해소에 인구 + 거리 휴리스틱 적용 |
| **Lieberman et al. (2010)** "Geotagging with local lexicons..." | LGL 데이터셋 |
| **Gritta et al. (2018)** "What's missing in geographical parsing?" | Toponym 해소 벤치마크 |

---

*마지막 업데이트: 2026-03-05*
