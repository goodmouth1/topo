# GCN 기반 지명 모호성 해소 연구 - 대화 정리

> 2026년 2월~3월 석사 논문 연구 과정에서 Claude와 나눈 대화를 정리한 문서

---

## 1. 연구 개요

**논문 주제**: GCN(Graph Convolutional Network) 기반 지명 모호성 해소 (Toponym Disambiguation)

**핵심 아이디어**: 소셜 네트워크의 그래프 구조를 GCN으로 학습하여 유저 위치를 예측하고, 그 위치를 anchor로 삼아 Gravity Model로 모호한 지명(예: Portland → OR? ME?)을 해소한다.

**2단계 프레임워크**:
- **Stage 1 (Anchor Acquisition)**: GCN으로 유저 위치 예측
- **Stage 2 (Gravity Model)**: Score = Population / Distance^β (β=1.5)

---

## 2. 완성된 Twitter (GeoText) 실험

### 데이터셋
- GeoText: 9,475명 유저, 380K 트윗
- @mention 관계로 소셜 그래프 구축
- 수동 라벨링: 235개 toponym

### 파이프라인
1. 유저의 모든 트윗 → TF-IDF 벡터화 (텍스트 특징)
2. @mention 관계 → 인접행렬 (그래프 구조)
3. GCN 훈련 → 유저 위치 예측 (anchor)
4. 트윗 내 지명 추출 → GeoNames에서 후보 검색
5. Gravity Model로 후보 점수 매기기 → 최고 점수 선택
6. 정답(수동 라벨)과 비교

### 결과
| 방법 | 정확도 |
|------|--------|
| Baseline (Population Only) | 88.51% |
| GCN + Gravity Model | **92.34%** |

### 코드 위치
서버: `/home/wang/master_thesis/`

| 파일 | 역할 |
|------|------|
| `data.py` | DataLoader (데이터 로딩, TF-IDF, @mention 그래프) |
| `gcnmodel.py` | GraphConv 클래스 (Theano/Lasagne GCN) |
| `gcnmain.py` | GCN 훈련 메인 |
| `run_gravity_with_gcn_state.py` | Gravity Model 평가 |
| `labeling/manual_labeled_data.pkl` | 수동 라벨링 데이터 |
| `cities1000.txt` | GeoNames 인구 1000+ 도시 |

---

## 3. Gravity Model 이론적 배경

### 역사
- 1885 Ravenstein: 인구 이동 법칙 (개념적)
- 1941 Stewart: 공식 적용
- 1962 Tinbergen: 무역 공식
- 1971 Wilson: 체계화

### 논문에서 사용하는 단순화 버전
```
Score = Population / Distance^β
```
- Population: GeoNames에서 가져온 도시 인구
- Distance: GCN 예측 좌표(anchor)와 후보 도시 간 Haversine 거리
- β: 거리 감쇠 파라미터 (Twitter 최적: 1.5)

### β 파라미터 의미
- β가 클수록 가까운 후보에 강한 가중치
- β가 작을수록 인구가 큰 후보에 유리
- 최적 β는 데이터셋마다 다름

---

## 4. LGL 외부 벤치마크 실험 (시도 → 폐기)

### 배경
논문의 일반화 가능성을 보여주기 위해 뉴스 데이터셋(LGL)에도 적용 시도.

### LGL 데이터셋
- Local Global Lexicon: 588개 뉴스 기사, 5,088개 toponym
- 86개 신문사에서 수집

### Centroid 방식
- GeoText에서는 GCN으로 anchor를 만들지만, LGL에는 소셜 그래프가 없음
- 대안: 기사 내 toponym 좌표들의 **평균값(centroid)**을 anchor로 사용
- 결과: Population Only 45.33% → Gravity(β=1.5) 59.33% (+14pp)

### 교수님 피드백 🔴
> "정답을 미리 보고 시험 치는 구조"

**문제점**: centroid 계산에 정답 좌표가 포함됨 → **데이터 누수 (Data Leakage)**
- 정답 toponym의 좌표로 anchor를 만들고
- 그 anchor로 다시 정답을 맞추는 구조
- 학술적으로 방어 불가

### 교수님 지시사항
1. LGL 실험 제거
2. GCN + Gravity에 집중
3. **GCN을 적용할 수 있는 다른 공개 데이터셋 찾기**

---

## 5. 데이터셋 탐색 과정

### 핵심 문제: 세 가지를 동시에 갖춘 데이터가 없다

GCN + Gravity를 돌리려면:
1. **소셜 네트워크 그래프** → GCN용
2. **텍스트** → TF-IDF 특징
3. **지명 정답 좌표** → 평가용

현실: 두 연구 커뮤니티가 따로 발전했음
- **User Geolocation** (Rahimi, Jurgens): 그래프 ✓, 좌표 ✓, 지명 라벨 ✗
- **Toponym Disambiguation** (Gritta, Hu): 지명 라벨 ✓, 그래프 ✗

### 검토한 Twitter 기반 Toponym 데이터셋

| 데이터셋 | Toponym 라벨 | 소셜 그래프 | 사용 가능? |
|---------|:-----------:|:---------:|:---------:|
| GeoCorpora (6K tweets) | ✓ | Tweet ID만 | ✗ (API 제한) |
| IDRISI-RE/D (20K+ tweets) | ✓ | user_id 삭제됨 | ✗ (익명화) |
| DLRGeoTweet (7,364 tweets) | ✓ | Tweet ID만 | ✗ (API 제한) |
| Geoparse Benchmark | ✓ | JSON 메타데이터 | △ (사이트 다운) |

**GeoCorpora/DLRGeoTweet로 소셜 그래프 구축 가능한가?**
- 이론적으로 Twitter API로 hydrate → @mention 추출 가능
- 실제로는:
  - Twitter API 비용: Basic $100/월, Pro $5,000/월 (Academic tier 2023년 폐지)
  - 10년 전 트윗 삭제율: 30~50%
  - **치명적 문제**: 6K~7K개 개별 트윗 → 유저 간 연결 거의 없음 → GCN 작동 불가

### 검토한 대안 SNS 플랫폼

**Bluesky**:
- BlueTempNet: 4M 유저, 235M 포스트, follow/block 네트워크
- 장점: 오픈 API, 탈중앙화 SNS 신규성
- 단점: toponym 라벨 없음, 처음부터 구축 필요

**Mastodon**:
- 400K+ 유저, 5.5M 링크, 1,700 인스턴스
- 단점: toponym 라벨 없음

### 검토한 LBSN (위치 기반 소셜 네트워크)

| 데이터셋 | 소셜 그래프 | 텍스트 | 좌표 | Toponym 라벨 |
|---------|:---------:|:-----:|:----:|:----------:|
| **Yelp** | ✓ friends | ✓ reviews | ✓ | ✓ (city!) |
| Foursquare Global | ✓ 600K edges | △ tips | ✓ | ✗ |
| Foursquare UMN | ✓ 27M edges | ✗ | ✓ | ✗ |
| Gowalla (SNAP) | ✓ 950K edges | ✗ | ✓ | ✗ |
| Brightkite (SNAP) | ✓ 214K edges | ✗ | ✓ | ✗ |

---

## 6. 돌파구: Yelp city 필드 활용 ⭐

### 핵심 발견

Yelp `business.json`의 구조:
```json
{
  "name": "Garaje",
  "city": "Portland",        ← 이것이 toponym (모호한 지명)
  "state": "AZ",             ← 이걸 제거하면 모호성 발생!
  "latitude": 33.4484,       ← ground truth 좌표
  "longitude": -112.0740
}
```

**아이디어**: state 정보를 제거하고 city 이름만 남기면, 자연스럽게 지명 모호성이 생긴다!
- "Portland" → Oregon? Maine? Texas? Indiana?
- GeoNames에서 후보 검색 → Gravity Model로 선택 → 실제 좌표로 평가

### 왜 이것이 혁신적인가

1. **수작업 라벨링 불필요**: city 필드 + 좌표가 이미 존재
2. **소셜 그래프 완비**: user.json의 friends 배열
3. **풍부한 텍스트**: review.json (트윗보다 훨씬 길고 정보량 많음)
4. **데이터 누수 없음**: state만 제거, GCN은 별도로 유저 위치 학습
5. **NER 오류 없음**: city 필드가 이미 구조화된 지명

### GeoText vs Yelp 파이프라인 비교

| 구성요소 | Twitter (GeoText) | Yelp |
|---------|-------------------|------|
| 노드 | Twitter 유저 (9,475) | Yelp 유저 (~150K+) |
| 엣지 | @mention (텍스트 파싱) | friendship (user.json) |
| 텍스트 | Tweet (짧음) | Review (길음) |
| TF-IDF | Tweet 합침 | Review 합침 |
| 좌표 | 유저 프로필 좌표 | 리뷰한 비즈니스 중앙값 |
| Toponym 소스 | Tweet 내 지명 (NER) | business.city (구조화) |
| 후보 생성 | GeoNames 검색 | GeoNames 검색 |
| 정답 | 수동 라벨링 (235개) | business.lat/lon (**자동!**) |
| GCN | @mention 인접행렬 | friendship 인접행렬 |
| Gravity | Pop/Dist^β | Pop/Dist^β (동일) |

**코드 변경 최소화**: DataLoader의 `get_graph()`만 @mention → friendship으로 교체하면 됨

---

## 7. 최종 실험 구성

### 실험 1: Twitter (GeoText) — 완성됨 ✅
- 소셜 그래프: @mention
- 텍스트: Tweet TF-IDF
- 결과: 88.51% → 92.34%

### 실험 2: Yelp — 구축 예정 🔨
- 소셜 그래프: friendship
- 텍스트: Review TF-IDF
- 동일 프레임워크 적용

### 논문 프레이밍
> "Twitter의 @mention 그래프뿐 아니라, Yelp의 friendship 그래프에서도
> 동일한 GCN + Gravity 프레임워크가 작동함을 보임으로써,
> **플랫폼에 독립적인 일반화 가능성**을 검증하였다."

이 구성의 강점:
- 마이크로블로그 + 리뷰 플랫폼 (서로 다른 도메인)
- @mention + friendship (서로 다른 그래프 유형)
- 짧은 텍스트 + 긴 텍스트 (서로 다른 정보량)
- LGL 데이터 누수 문제 완전 해결

---

## 8. Yelp 파이프라인 구축 계획

### 실행 순서

| 단계 | 파일 | 역할 |
|:---:|------|------|
| 0 | `step0_explore_yelp.py` | 데이터 탐색 (모호 도시 수, friendship 밀도, 리뷰 수) |
| 1 | `step1_convert_yelp_to_geotext.py` | Yelp → GeoText 포맷 변환 |
| 2 | `yelp_data.py` | YelpDataLoader (friendship 그래프 로드) |
| 3 | `yelp_gcnmain.py` | GCN 훈련 |
| 4 | `step4_build_eval_data.py` | 모호 도시 평가 데이터 자동 구축 |
| 5 | `step5_yelp_gravity_eval.py` | Gravity Model 평가 (β별 + State 보너스) |

### 필요한 데이터
- Yelp Open Dataset (공식 사이트 또는 Kaggle에서 다운로드, 약 9GB)
  - `yelp_academic_dataset_business.json`
  - `yelp_academic_dataset_review.json`
  - `yelp_academic_dataset_user.json`

### 확인 필요 사항 (Step 0에서 판단)
1. **모호한 도시 수**: 같은 city 이름이 다른 state에 존재하는 경우가 충분한가? (30개+)
2. **Friendship 밀도**: GCN이 작동할 만큼 그래프가 dense한가?
3. **유저당 리뷰 수**: TF-IDF feature가 충분한가?

### 잠재적 문제
- Yelp가 11개 metro area에 집중 → 모호 도시가 적을 수 있음
- Yelp 유저가 GeoText보다 10배+ → 메모리 주의
- Review 텍스트가 길어서 TF-IDF sparse matrix 크기 주의

---

## 9. 기술 세부사항 메모

### GCN 모델 (gcnmodel.py)
- Theano/Lasagne 기반
- Kipf (2016) GCN 구조
- Highway connections 사용
- 입력: TF-IDF sparse matrix + 정규화된 인접행렬
- 출력: 유저별 지역 클래스 예측 → 클래스 중앙값 좌표

### 좌표 이산화 (assignClasses)
- 미국 전역을 bucket_size(300km) 격자로 나눔
- 각 유저의 실제 좌표 → 가장 가까운 격자 클래스 할당
- GCN은 분류 문제로 훈련 (회귀가 아님)

### 인접행렬 정규화
- A_hat = D^{-1/2} * A * D^{-1/2}
- self-loop 추가 (대각선 = 1)

### Gravity Model 변형 (Twitter에서 사용)
```python
# 3단계 보너스 시스템
if text_match and gcn_match:
    bonus = 10.0  # 텍스트와 GCN 둘 다 같은 state 가리킴
elif text_match or gcn_match:
    bonus = 5.0   # 둘 중 하나
else:
    bonus = 1.0   # 증거 없음

score = (pop / dist^1.5) * bonus
```

---

## 10. 참고 논문

- **Rahimi et al. (2018)**: Semi-supervised User Geolocation via Graph Convolutional Networks — GCN 기반 유저 위치 예측의 원논문
- **Kipf & Welling (2017)**: Semi-Supervised Classification with Graph Convolutional Networks — GCN 원논문
- **Gritta et al. (2018)**: What's missing in geographical parsing? — Toponym 해소 벤치마크
- **Lieberman et al. (2010)**: Geotagging with local lexicons to build indexes for textually-specified spatial data — LGL 데이터셋
- **Stewart (1941)**: An inverse distance variation for certain social influences — Gravity Model 원논문

---

## 11. 타임라인

| 날짜 | 내용 |
|------|------|
| 2월 23일 | Ablation Study 설계, LGL 벤치마크 선정 |
| 2월 27일 | LGL 실험 완성 (45.33% → 59.33%), 논문 수식/방법론 문서화 |
| 3월 3일 | 교수님 피드백 (LGL centroid 데이터 누수 지적) |
| 3월 3일 | 대안 데이터셋 탐색 (Twitter, Bluesky, Mastodon, Foursquare...) |
| 3월 3일 | **Yelp city 필드 활용 아이디어 도출** |
| 3월 3일 | Yelp 파이프라인 설계 & Claude Code용 가이드 작성 |
| 3월 5일 | Yelp 데이터셋 다운로드 준비 |

---

*이 문서는 연구 진행 과정의 기록이며, Yelp 실험 결과에 따라 업데이트될 예정입니다.*
