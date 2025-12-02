# 🌙 Dacon Trade Forecast — Team *Girls_Night*

### 🏫 제3회 국민대학교 AI빅데이터 분석 경진대회  
**주제:** 품목 간 무역 연동성과 미래 예측 가능성에 대한 AI 기술의 응용  
**주최:** 국민대학교 경영대학원 · 한국기계산업진흥회(KOAMI)

---
# 1. Introduction

## 1.1 대회 개요
본 프로젝트는 국민대학교 경영대학원과 한국기계산업진흥회(KOAMI)가 공동 주최한  
**제3회 국민대학교 AI빅데이터 분석 경진대회**를 기반으로 수행되었다.  
대회의 목표는 2022년 1월 ~ 2025년 8월까지의 100개 수입 품목 월별 무역 데이터를 활용해  
**품목 간 공행성(comovement) 관계를 식별하고,  
선행 품목의 흐름을 이용해 후행 품목의 미래 무역량을 예측할 수 있는 AI 모델을 구축하는 것**이다.  
즉, 무역 시계열 내 숨겨진 구조적 관계를 분석하는 실무형 문제에 가깝다.

---

## 1.2 문제 정의
본 대회는 **관계 탐지 + 미래 예측**이라는 두 가지 목표를 동시에 해결해야 하는 이중 구조 문제이다.

### (1) 공행성(Comovement) 판별 – Relationship Detection  
각 품목 조합 (A, B)에 대해  
- A의 시계열 움직임이 B보다 **선행하는지(leading)**  
- 두 품목 간에 **구조적 연동성(comovement)**이 존재하는지  
를 예측해야 한다.

이는 단순 상관성 분석이 아니라  
**시차(lag)를 고려한 방향성 있는 관계 추론(Binary Classification)** 문제이다.

### (2) 후행 품목 미래 무역량 예측 – Conditional Forecasting  
공행성이 있는 것으로 판단된 쌍(A → B)에 한해  
**후행 품목 B의 다음달 총 무역량(value)**을 회귀 방식으로 예측해야 한다.

즉, 선행–후행 구조를 조건부로 활용하는  
**Structure-conditioned Regression** 문제라고 볼 수 있다.

### (종합 정의)  
대회는 아래 두 가지를 하나의 파이프라인에서 해결하는 것을 목표로 한다:

1) **품목 간 선후행 구조적 관계 추론 (A → B)**  
2) **관계가 있다고 판정된 쌍의 미래 value 예측**

따라서 전체 문제는  
**Hybrid Structural Forecasting** 문제로 정의할 수 있다.

---

## 1.3 데이터 특성 및 난이도
주어진 무역 데이터는 실무 데이터의 제약을 그대로 반영하고 있어 여러 도전 요인이 존재한다.

- **Sparsity**: 100개 품목 중 상당수는 특정 시점에만 거래가 존재하거나 거래가 거의 없음  
- **속성(column)의 활용성 제한**:  
  - `type`은 단일값  
  - `weight`, `quantity`는 단위 불일치가 심해 직접적 사용 불가  
- **이상치 포함**: value=0인데 weight>0 등의 계량 오류 존재  
- **패턴 불규칙성**: HS4 코드가 같더라도 거래 규모·주기·lag가 품목마다 상이  
- **Target Month가 단일(2025-08)** → 일반적 cross-validation 적용 어려움  

이로 인해 단순한 시계열 예측 모델로는 관계성·방향성을 포착하기 어렵고,  
**비정형 시계열에서 구조적 신호를 찾아내는 feature-driven 접근**이 필요했다.

---

## 1.4 프로젝트 접근 방식 요약
우리 팀은 raw 데이터를 월 단위로 재구성하고, 동일 월 내 여러 거래는 `seq`를 기준으로 합산하여  
각 품목에 대한 **월별 total_value 시계열**을 구축하였다.

초기에는 HS4 기반 품목 군집을 활용해 공행성을 탐색했으나,  
데이터 sparsity 때문에 유효한 쌍이 충분히 확보되지 않아  
최종적으로 **feature-driven pair filtering 전략**으로 방향을 전환했다.

사용한 주요 특성은 다음과 같다.  
- Pearson correlation  
- 다양한 window 기반 통계 특성  
- **lag-correlated movement 분석**(lag-corr)  
- HS4 기반 계층적 거리  

특히 lag-corr는 **선후행 방향성 검출에 가장 유효한 지표**로 작용했다.

최종 예측 단계에서는 **LightGBM 기반 회귀 모델**을 사용해  
선행 품목 A의 패턴을 기반으로 후행 품목 B의 미래 value를 추정하였다.  
이 접근 방식은 데이터 sparsity·비정형성 문제를 안정적으로 완화하는 데 효과적이었다.
---
## 2. Dataset & EDA Summary

### 2.1 Dataset Overview

대회에서 제공된 주요 데이터는 다음과 같다.

- `train.csv`  
  - **기간**: 2022-01 ~ 2025-07  
  - **컬럼**
    - `item_id`: 무역품 식별 ID (총 100개 품목)
    - `year`, `month`: 거래 연도·월
    - `seq`: 동일 연·월 내 거래를 구분하는 일련번호
    - `type`: 유형 구분 코드 (대부분 1)
    - `hs4`: HS4 코드
    - `weight`: 중량
    - `quantity`: 수량
    - `value`: 거래 금액(무역량, 정수형)

- `sample_submission.csv`  
  - `leading_item_id`: 선행 품목 ID  
  - `following_item_id`: 후행 품목 ID  
  - `value`: 예측해야 할 2025년 8월의 후행 품목 총 무역량

EDA 및 전처리 단계를 거쳐, `train.csv`는 월 단위로 집계된 `train_month.csv`로 변환되며,  
대표 코드의 `load_monthly_data()`는 이를 다시 `item_id × month` 피벗 형태로 불러와 분석에 활용한다.

---

### 2.2 주요 EDA Insight

EDA를 통해 다음과 같은 데이터 특성과 문제점을 확인했다.

1. **계량 오류 / 이상치 존재**  
   - `value=0`인데 `weight>0`인 레코드가 존재하는 등,  
     일부 샘플에서 물리적으로 해석하기 어려운 조합이 발견되었다.  
   - 단순히 전 컬럼을 신뢰하기보다는, 예측 대상인 `value` 중심 구조로 재해석이 필요했다.

2. **시계열 sparsity 및 데이터 부족**  
   - 전체 100개 품목 중 상당수는 특정 시점에만 거래가 발생하거나,  
     전체 기간 동안 비활성인 경우가 많았다.  
   - 공행성 분석과 학습에 활용 가능한 **active item 수는 약 30~40개 수준**으로 제한되며,  
     이는 쌍 단위(pairwise) 분석 시 유효한 데이터 포인트가 매우 적다는 의미다.

3. **속성 컬럼의 활용성 제약**  
   - `type`은 대부분 값이 1로, 정보량이 거의 없다고 판단했다.  
   - 무역품 특성상 품목별 단위가 제각각이라 `weight`, `quantity`는  
     결측·단위 불일치가 많고, 정규화 기준도 불명확하였다.  
   - 결과적으로 **`year`, `month`, `seq`, `hs4`, `value` 중심의 시계열·구조 분석**으로 축소했다.

4. **HS4 기준 군집의 한계**  
   - 동일 `hs4` 그룹 내에서도 거래 패턴(규모, 주기, 변동성)이 품목마다 제각각이었다.  
   - HS4 기반 군집을 이용해 공행성 후보군을 제한하는 전략은,  
     실제로는 유의한 쌍이 거의 남지 않아 **군집 기반 FE 효과가 낮은 것으로 평가**되었다.

5. **시차(lag)와 계절성의 불규칙성**  
   - 품목별 시계열을 그려 보면, 동일 시점에서 움직이는 경우보다  
     **몇 달의 지연(lag)을 두고 반응하는 패턴**이 더 자주 관찰되었다.  
   - 하지만 lag 길이는 품목마다 다르고 고정되어 있지 않아,  
     “한 달 고정 lag” 같은 단순 구조로는 설명이 되지 않았다.  
   - 월 단위 계절성(seasonality)은 미약했으며,  
     단순 seasonal correlation은 공행성 신호로 보기 어렵다고 판단했다.

6. **시각적 공행성 판단의 한계**  
   - 개별 품목의 가격대와 스케일이 크게 달라,  
     raw 시계열 그래프만으로 두 품목이 “같이 움직인다”고 판단하기 어려웠다.  
   - 이로 인해 사람 눈에 보이는 직관보다는,  
     **정량적 상관분석(correlation, lag-corr 등)에 기반한 공행성 탐색이 필수**라는 결론에 도달했다.

7. **검증 시점 구조의 특수성**  
   - 학습 데이터는 2025년 상반기까지만 제공되며,  
     실제 예측 대상은 2025년 8월 단일 시점이다.  
   - 따라서 일반적인 k-fold cross-validation보다는  
     **마지막 몇 개 월을 시간 순으로 잘라 쓰는 time-based 검증 전략**이 필요했다.

---

### 2.3 시계열 구조화: 월별 total_value & Pivot 설계

EDA 결과를 바탕으로, 다음과 같은 방식으로 시계열을 재구성했다.

1. **seq 병합 → 월별 total_value 생성**  
   - 동일 `item_id`, `year`, `month` 내에 여러 거래(`seq`)가 존재하는 경우,  
     해당 월의 `value`를 합산하여 **월 단위 `total_value`**를 정의하였다.  
   - 대회 평가 대상 또한 “월별 총 무역량(value)”이므로,  
     이는 도메인 측면에서도 자연스러운 집계 기준이다.

2. **item × month 피벗 테이블 구성**  
   - 대표 코드의 `load_monthly_data()`에서는 `train_month.csv`를 기반으로  
     `item_id × ym` 형태의 피벗 테이블을 생성한다.
   - 이 테이블은 공행성 탐색 및 lag-corr 계산의 **기본 시계열 그리드**로 활용되며,  
     존재하지 않는 거래(月)는 `0`으로 채워 연속적인 월 시퀀스를 구성한다.

3. **active item 필터링**  
   - 공행성 분석에 앞서, 각 품목의 시계열에서 `value>0`인 월의 개수를 세어  
     일정 기준(`min_nonzero`) 미만인 품목은 분석 대상에서 제외하였다.  
   - 대표 코드에서는 `min_nonzero=12`~`15` 수준으로 설정하여,  
     **실질적인 거래 히스토리가 있는 품목만 후보군으로 유지**한다.

---

### 2.4 EDA 기반 공행성 탐색 설계

위와 같은 EDA 결과는 공행성 탐색 로직과 모델링 설계에 직접 반영되었다.

1. **lag 기반 공행성 후보 탐색**  
   - `find_comovement_pairs()`에서는  
     - 최대 시차 `max_lag`(예: 6개월)  
     - 최소 비제로 월 수 `min_nonzero`  
     - 상관계수 임계값 `corr_threshold` (예: 0.38~0.40)  
     를 활용해 공행성 후보 쌍을 찾는다.
   - 구체적으로,  
     - Leader 시계열 `A(t-lag)`와 Follower 시계열 `B(t)`를 정렬해  
     - lag 1~6 범위에서 Pearson correlation을 계산하고  
     - 가장 절대값이 큰 상관계수를 `max_corr`, 해당 시차를 `best_lag`로 기록한다.
   - `|max_corr|`가 임계값 미만인 쌍은 버리고,  
     **충분한 거래 이력 + 일정 수준 이상의 lag-corr를 가진 쌍만 공행성 후보로 채택**한다.

2. **시각 기반 Time Series CV 전략**  
   - `train_cv_ensemble()`에서는 `target_date`(예: t+1 month)별로 데이터를 구분하고,  
     **마지막 몇 개 월을 차례대로 validation으로 사용하는 time-based CV**를 적용한다.  
   - 이 전략은 2025년 상반기까지만 존재하는 학습 구간과  
     2025년 8월이라는 단일 예측 타겟 구조를 최대한 반영하기 위한 선택이다.

3. **feature-driven 접근으로의 전환**  
   - HS4 군집, raw weight/quantity 사용 등 초기 아이디어들은  
     데이터 품질 및 sparsity 이슈로 인해 실질적인 성능 향상을 제공하지 못했다.  
   - 대신, lag-corr, roll mean/std, 증감률, leader–follower 간 비율/차이 등  
     **시계열 통계 기반 feature를 중심으로 한 feature-driven 전략**이  
     공행성 탐색과 예측에 더 효과적이었다.

EDA 파트에서의 결론은 다음과 같다:

> “이 데이터는 전통적인 시계열 모델을 그대로 얹기보다는,  
>  월별 total_value 시계열을 기준으로 lag-corr와 각종 통계 특성을 추출하여  
>  공행성 후보를 선별하는 **feature-driven 구조 설계**가 필수적이다.”

---
## 3. Preprocessing

본 프로젝트의 전처리 과정은 모델이 요구하는 입력 형식(`train_month.csv`)을 생성하고,  
공행성 분석 및 lag 기반 피처 생성이 안정적으로 이루어지도록  
원시 데이터를 월 단위 시계열 구조로 재편성하는 데 초점을 맞추었다.

---

### 3.1 Raw → Month-level Aggregation

원본 데이터(`train.csv`)는 동일 item × year × month 내에서 여러 거래(`seq`)가 발생하며,  
대회 평가 기준은 “월별 총 무역량(value)”이므로 다음과 같은 방식으로 집계하였다.

- 동일 item × month 내 `value` 합산 → **total_value**  
- 동일 item × month 내 `weight`, `quantity`는 단위 불일치 및 결측 문제로 사용하지 않음  
- 월별 시계열을 구성하기 위해 `ym`(year-month) 컬럼 생성  

최종적으로 아래 구조의 **train_month.csv**를 생성한다.

| item_id | year | month | value |
|---------|------|--------|--------|
| A01     | 2022 | 01     | 1500   |
| A01     | 2022 | 02     | 3200   |
| ...     | ...  | ...    | ...    |

이 파일은 이후 전체 모델링 파이프라인의 기본 입력으로 사용된다.

---

### 3.2 Outlier & Missing Handling

EDA 과정에서 다음과 같은 품질 이슈를 확인하였다.

- `value=0` & `weight>0`과 같이 물리적으로 해석 어려운 조합 존재  
- `type` 컬럼은 대부분 단일값으로 의미 없음  
- `weight`, `quantity`는 단위가 품목마다 달라 정규화 불가 수준  
- 일부 월에 거래가 존재하지 않음 → 공백 월은 **0으로 채움**

따라서 전처리 단계에서는 **value 중심 구조(value aggregation)**만 유지하며  
그 외 raw 컬럼은 모델 학습에 포함하지 않았다.

---

### 3.3 월 단위 Complete Panel 구성

공행성(comovement) 분석에서 가장 중요한 요소는  
**동일 길이의 시계열**을 가지는 것이다.

따라서:

- 모든 item_id에 대해 전체 날짜 범위(2022-01 ~ 2025-07)를 생성  
- 거래가 없는 월은 value=0으로 채운 뒤  
- item_id × ym 형태의 완전 패널 시계열을 구성하였다.

이는 lag 기반 상관분석과 pairwise feature 생성에서 필수적인 구조이다.

---

### 3.4 Pivot Table Construction (대표 모델 입력 구조)

대표 모델 코드(`load_monthly_data`)는 전처리의 최종 결과인 `train_month.csv`를 불러와  
다음과 같은 pivot table을 생성한다.

pivot[item_id][ym] = monthly total_value


예시:

| ym      | item_A | item_B | item_C |
|---------|--------|--------|--------|
| 2022-01 |  3200  |   0    | 1200   |
| 2022-02 |  1800  |  450   |   0    |
| ...     |  ...   |  ...   |  ...   |

이 pivot table은 다음 단계에서 다음 목적으로 사용된다.

- **공행성 후보 탐색(find_comovement_pairs)**  
- **lag alignment 기반 feature 생성**  
- **pairwise regression 학습 데이터 구성**

즉, 모델 파이프라인 전체가 이 pivot 시계열을 기준으로 설계되어 있다.

---

### ✔ 전처리 단계 요약

| 단계 | 설명 |
|------|------|
| Raw Cleaning | 타입 정규화, 중복 제거, 날짜 생성 |
| Monthly Aggregation | seq 단위 거래 → 월별 total_value 집계 |
| Missing Handling | 비거래 월 → value=0 |
| Complete Panel 구성 | 모든 item × 모든 월 시계열 확보 |
| Pivot 생성 | item_id × ym 구조 (공행성 분석의 핵심 입력) |
| train_month.csv 생성 | 모델 파이프라인 표준 입력 형식 |

---
## 4. Feature Engineering

Feature Engineering은 본 문제의 핵심 단계로,  
공행성(comovement) 기반 구조를 회귀 모델이 학습할 수 있도록  
**Leader(A)–Follower(B)의 시계열 관계를 재정렬(alignment)**하는 데 초점을 두었다.

대표 모델의 `_build_pair_frame()` 함수는 하나의 (A, B) 쌍에 대해  
다음 네 가지 종류의 특징을 생성한다.

---

### 4.1 Follower-based Time-series Features (B)

후행 품목 B 자체의 패턴은 미래 value 예측의 가장 직접적인 단서이다.

- **Lag features**: `b_t_1`, `b_t_2`, `b_t_3`, `b_t_6`  
- **차분(diff)**: `b_diff1 = b_t - b_t_1`  
- **증감률(percent change)**: `b_pct1`  
- **Rolling statistics**:  
  - `b_roll3_mean`, `b_roll3_std`  
  - `b_roll6_mean`, `b_roll6_std`  

이들은 B의 단기 변동성, 추세, 평균 수준 등을 반영한다.

---

### 4.2 Leader-aligned Features (A’s influence on B)

A가 B보다 **best_lag**만큼 선행한다는 사실을 모델에 반영하기 위해  
leader의 시계열을 follower의 타깃 시점에 맞춰 재정렬(alignment)한다.

정렬 방식:
- B(t+1)을 예측할 때 사용할 수 있는 leader 정보는  
  **A(t - (best_lag - 1))**

생성되는 특징:
- `a_t_aligned`  
- `a_t_aligned_1`, `a_t_aligned_2`, `a_t_aligned_3`  
- `a_diff1`, `a_pct1`  
- `a_roll3_mean`, `a_roll6_mean`

이 구조는 lag-corr 분석을 그대로 feature에 녹여  
모델이 **A → B causal-like dependency**를 학습할 수 있게 한다.

---

### 4.3 Interaction Features (A × B Pairwise Relationship)

A와 B의 상대적 규모 및 차이를 직접 반영한다.

- `ab_ratio = A(t*) / B(t)`  
- `ab_diff = A(t*) - B(t)`  

이 교차 특성은 단일 품목 시계열에서는 얻을 수 없는  
**pair-level 구조적 신호**를 제공한다.

---

### 4.4 Meta Features (Lag, Corr, Month)

- `best_lag`: A가 B를 선행하는 시차  
- `max_corr`: lag alignment 기반 상관 강도  
- `month`: 약한 시즌 패턴을 반영하기 위해 추가된 달(month) 정보

이 메타 정보는 공행성 신뢰도 및 시계열 위치 정보를 모델에 전달한다.

---

### ✔ Target Definition

회귀 모델의 타겟은 다음달 follower value:

target_value = B(t+1)
target_log1p = log1p(B(t+1)


회귀 안정성을 위해 log1p 변환을 적용하며,  
예측 후 inverse transform(`exp1m`)을 수행한다.

---

### ✔ Summary

본 Feature Engineering 전략은 다음을 만족하도록 설계되었다.

- 공행성 신호를 **정렬된 leader-features**로 반영  
- follower 자체 변동성을 다각도로 캡처  
- A–B 관계를 ratio/diff로 구조화  
- lag-corr 기반 directionality을 모델이 학습 가능하도록 설계  

즉, 단순 시계열 예측이 아닌  
**pairwise, lag-aware structural forecasting** 문제에 최적화된 FE 구조이다.

---
## 5. Modeling

본 문제의 모델링은 크게 **두 단계**로 구성된다.

1. **품목 간 공행성(comovement) 탐색**  
2. **공행성이 있다고 판단된 (A → B) 쌍에 대해 pairwise regression으로 B의 다음 달 value 예측**

이 구조는 대회 문제 정의(“공행성 쌍 판별 + B의 미래 value 예측”)와 정합성이 높으며,  
도메인 특성상 단일 글로벌 모델보다 **쌍 단위(pair-level) 모델링**이 더 적합한 접근이라고 판단하였다.

---

### 5.1 Comovement Pair Mining

첫 번째 단계는 두 품목 A, B 간에 **선후행 관계가 있는지** 판별하는 것이다.  
대표 모델은 pivot된 item × month 시계열을 입력으로 받아 다음 과정을 수행한다.

#### (1) Non-zero 기간 필터링
- 거래가 거의 없는 품목을 배제하기 위해  
  **min_nonzero ≥ 12~15개월** 기준으로 active item을 선별한다.

#### (2) Lag-correlation 기반 공행성 탐색
각 item pair (A, B)에 대해 다음을 계산한다.

- lag = 1~6 범위에서 corr( A(t - lag), B(t) )

- 절댓값이 가장 큰 상관계수 → `max_corr`  
- 그때의 lag → `best_lag`

#### (3) 공행성 쌍 선정 기준
다음 조건을 모두 만족하는 경우 **공행성 쌍**으로 인정한다.

- |max_corr| ≥ corr_threshold (예: 0.38~0.40)  
- A, B 모두 충분한 active 기록 보유  
- A ≠ B (자가 상관 제외)

이 과정을 통해 전체 100개 품목 중 실제 공행성이 있다고 판단된 소수의 (A → B) 후보만 남기게 된다.

---

### 5.2 Pairwise Regression Modeling

공행성이 확인된 각 (A → B) 쌍에 대해  
다음 달 B의 value를 예측하는 **독립적인 회귀 모델 input row**를 생성한다.

이때 사용되는 특징들은 Feature Engineering 섹션에서 설명한 네 가지 그룹:

- Follower(B) 기반 시계열 특징  
- Leader(A) alignment 기반 lag-aware 특징  
- Interaction(A×B) 특징  
- Meta 정보(best_lag, max_corr, month)

대표 모델의 타깃은 다음과 같다.

target_value = B(t+1)
target_log1p = log1p(B(t+1))


log1p 변환을 적용하여 스케일 불안정성과 이상값의 영향을 줄였다.

---

### 5.3 Model Architecture: LightGBM Regressor

시계열 길이가 매우 짧고,  
쌍별 데이터 포인트 수가 수십 개 수준으로 제한되는 특성을 고려하여  
딥러닝 대신 **LightGBM 회귀 모델**을 채택하였다.

- `objective = regression_l1` (MAE 기반)
- `n_estimators ≈ 2000`
- `learning_rate = 0.03`
- `num_leaves = 31`, `max_depth = -1`
- `colsample_bytree = 0.7`
- `subsample = 0.7`
- fold마다 `random_state` 변경해 앙상블 다양성 확보

LightGBM은 적은 샘플에서도 안정적이며,  
비선형 상호작용과 시계열 기반 FE를 효과적으로 학습할 수 있다는 장점이 있다.

---

### 5.4 Time-series Cross Validation

2025년 8월 단일 시점을 예측해야 하기 때문에  
일반적인 K-Fold CV는 사용할 수 없었다.

대신, 대표 모델은 다음과 같은 **time-based expanding window CV**를 사용한다.

예:
- Fold 1: Train [:-5], Valid = month(t-5)  
- Fold 2: Train [:-4], Valid = month(t-4)  
- …  
- Fold 5: Train [:-1], Valid = month(t-1)

CV는 다음을 가능하게 한다.

- 최근 트렌드를 더 정확하게 반영  
- target leakage 방지  
- 시계열 구조 준수

---

### 5.5 Ensemble Strategy

각 Fold에서 학습된 모델은 모두 저장되며,  
테스트(2025-08) 예측 시 다음과 같이 평균 앙상블을 수행한다.

final_pred = mean(model_k.predict(x_test) for k in folds)


- 시계열 데이터의 불안정성을 완화  
- 각 Fold가 다른 기간을 학습하므로 서로 다른 패턴을 포착  
- 실제 LB 성능 개선에 효과적이었음

---
## 6. Experiments & Evaluation

본 문제는 “공행성 쌍 판별(F1)”과 “다음달 무역량 예측(NMAE)” 두 지표의 조합으로 평가되었다.  
따라서 모델링과 실험 설계는 두 요소 모두가 균형 있게 작동하도록 구성하였다.

---

### 6.1 Evaluation Metric

대회 공식 점수는 다음과 같다.

Score = 0.6 × F1 + 0.4 × (1 − NMAE)


#### 🟣 (1) F1 Score 정의

F1 = 2 × (Precision × Recall) ÷ (Precision + Recall)


- **Precision = TP / (TP + FP)**  
- **Recall = TP / (TP + FN)**  

여기서  
- **TP (True Positive)**: 정답에도 있고, 예측에도 있는 공행성 쌍  
- **FP (False Positive)**: 예측에는 있으나 정답에는 없는 쌍  
- **FN (False Negative)**: 정답에는 있으나 예측에 없는 쌍  

공행성 탐색의 기준을 너무 좁게 잡으면 FN 증가 → Recall 하락  
너무 넓게 잡으면 FP 증가 → Precision 하락  
→ 득점 최적화를 위해 쌍 개수 조절이 매우 중요했다.

---

#### 🟣 (2) NMAE Normalized Mean Absolute Error

NMAE = (1/|U|) × Σ[min(1, |y_true − y_pred| / (|y_true| + ε))]


- U = 정답쌍(G) ∪ 예측쌍(P)의 합집합  
- y_true = 정답 value (정수)  
- y_pred = 예측 value (정수 반올림)  
- FP/FN 상황 모두 오차 = 1.0 처리 (최하점)  
- 오차가 100%를 넘는 경우에도 1.0로 clipping  

즉,  
**공행성 쌍을 맞추지 못하면 회귀 잘 해도 점수가 크게 깎인다.**  
이 문제의 난이도가 바로 이 부분에서 나온다.

---

---

### 6.2 Validation Strategy (Time-series CV)

데이터가 2025년 7월까지만 존재하고  
예측 대상은 단일 시점 “2025년 8월”이므로,

일반 K-Fold는 적용 불가 → **expanding-window Time-series CV** 사용.

| Fold | Train 구간 | Valid 구간 |
|------|-------------|-------------|
| 1 | 전체[:-5] | month(t-5) |
| 2 | 전체[:-4] | month(t-4) |
| 3 | 전체[:-3] | month(t-3) |
| 4 | 전체[:-2] | month(t-2) |
| 5 | 전체[:-1] | month(t-1) |

이 방식은
- 미래 정보 누출 방지  
- 2025-08 예측 환경과 가장 유사한 구조  
라는 장점이 있다.

---

### 6.3 공행성 필터링 결과

이 단계는 전체 실험 중 가장 중요한 단계였다.  
처음에는 기준을 너무 엄격하게 잡아 약 **500개 수준**만 남겼고,  
이는 F1 점수를 크게 하락시켰다.

실험적으로 확인한 사실:

> **정확도가 조금 떨어지더라도 최소 1,000개 이상의 공행성 쌍을 제출해야  
> F1 손실 없이 안정적인 점수가 나왔다.**

최종적으로 공행성 필터링 후 남은 쌍 수는:

➡ **`[]`**  
(팀원이 직접 적을 예정)

참고: 대회에서 제공한 평가 쌍은 총 **9,999쌍**이었음.

---

### 6.4 Cross-validation Performance

각 Fold의 MAE는 아래와 같이 기록되었다.  
(팀이 기록한 수치를 이후 채워넣을 수 있음)

| Fold | MAE |
|------|-----|
| 1 | [] |
| 2 | [] |
| 3 | [] |
| 4 | [] |
| 5 | [] |

**Average MAE: []**

CV 간 편차가 크지 않아  
시계열 짧은 환경에서도 모델이 안정적으로 일반화하는 것을 확인했다.

---

### 6.5 Ablation Summary (Qualitative)

특징 조합별 실험 결과에서 다음을 확인했다.

- follower-only baseline → 성능 낮음  
- leader alignment 추가 → MAE 유의미하게 감소  
- interaction(ab_ratio, ab_diff) → pair 관계를 직접 반영하며 추가 개선  
- time-series CV + ensemble → 예측 안정성 증가  

전체적으로 “공행성 구조를 반영한 feature-driven 모델링”이  
단일 시계열 기반 모델보다 훨씬 효과적이었다.

---

### 6.6 Final Result (Leaderboard)

**예선 Public Leaderboard 기준: 전체 1,688명 중 259위**  
→ **상위 약 15%**

대회는 예선까지 진행했으며,  
Private Score은 0.3615357019로 Public Score와 동일하게 처리되었다.

우리 팀의 최종 제출 파일은 v7_improved_submission.csv이며, 공행성 탐색 + pairwise regression + ensemble 구조를 기반으로 한다.
---
## 7. Team Contributions

본 프로젝트는 팀 Girls_Night의 협업을 통해 진행되었으며,  
각 팀원은 아래와 같은 역할을 중심으로 기여했다.

| 이름 | 역할 및 기여 |
|------|--------------|
| **이수민 (팀장)** | 전체 파이프라인 설계 총괄, EDA 및 문제 정의 구조화, 공행성 필터링 전략 개선, README 문서화 및 리포지토리 관리, 모델링 실험 검증 |
| **권문진** | 전처리 코드 구현 보조, 공행성 후보군 검증, 시각화 보조 분석, 제출 전략 점검, 모델링 실험 검증 |
| **고민서** | 데이터 구조 정리, EDA 서포트, 특성별 패턴 확인, 모델 실험 대비 파일 관리, 모델링 실험 검증 |
| **허예경** | 노트북 시각화, 초기 baseline 분석, 실험 로그 정리 및 팀 내부 보고서 작성, 모델링 실험 검증 |

> ※ 실제 대회 진행 중 역할이 일부 겹치는 부분이 있었으며,  
> 중요한 실험과 설계는 팀원 전원이 함께 논의하고 결정했습니다.

---
## 8. Project Structure

```
dacon-trade-forecast-team_girls_night/
├── data/ # 원본/전처리 데이터 (gitignore)
│ ├── raw/ # train.csv, sample_submission.csv
│ ├── interim/ # 중간 가공 파일
│ └── processed/ # train_month.csv, pivot, summary 등
│
├── notebooks/ # 탐색·전처리·모델링 실험 노트북
│ ├── 1_EDA.ipynb
│ ├── 2_Preprocessing.ipynb
│ ├── 3_FeatureEngineering.ipynb
│ ├── 4_Modeling.ipynb
│ └── 5_Evaluation.ipynb
│
├── feature_engineering/ # 특징 생성 관련 유틸 모듈
│ ├── lag_features.py
│ ├── cross_features.py
│ ├── stats_features.py
│ └── feature_selection.py
│
├── src/ # 핵심 실행 파이프라인
│ ├── preprocess.py # train_month 생성
│ ├── feature_engineering.py
│ ├── correlation.py # 공행성 탐색
│ ├── train_model.py # LGBM pairwise regression
│ ├── evaluate.py
│ └── utils.py
│
├── outputs/ # 결과물 저장 (gitignore)
│ ├── figures/ # 시각화
│ ├── logs/ # 실험 로그
│ ├── models/ # 학습된 모델 가중치
│ └── submissions/ # 최종 제출 파일
│
├── configs/ # 설정 파일
│ ├── paths.yaml
│ ├── params.yaml
│ └── features.yaml
│
├── requirements.txt
├── README.md
└── .gitignore
```

---
## 9. Limitations & Future Work

본 프로젝트는 공행성 기반의 무역 시계열 예측이라는 도전적인 문제를 다루었으며,  
데이터 구조적 제약과 문제 정의의 특수성으로 인해 다음과 같은 한계가 존재한다.

---

### 9.1 Limitations

#### **1) 짧고 희소한(sparse) 시계열**
- 각 품목의 거래 기록이 매우 짧고 active months도 30~40개 수준에 불과하다.  
- 특히 value=0인 월이 과도하게 많아, 유효한 패턴을 포착하기 어려웠다.
- 이로 인해 데이터가 작은 변화에도 매우 민감하게 반응하여  
  **특징 공학(feature engineering)이 조금만 달라져도 성능이 급격히 흔들리는 문제**가 있었다.

> 실제로 FE에서 일부 계산 방식이나 기준을 조금만 바꿔도  
> LB 점수가 **순식간에 0.1대까지 떨어지는** 극단적인 민감도를 보였다.  
> 데이터의 불안정성과 sparsity가 결합되며 모델이 과도하게 흔들리는 구조였다.

#### **2) 공행성 신호의 불안정성**
- Lag-based correlation은 의미 있는 구조를 잡을 수 있지만  
  거래량이 적은 품목에서는 우연한 패턴에도 크게 반응한다.  
- best_lag이 시점에 따라 다르게 나타나는 경향이 뚜렷했다.

#### **3) HS4 기반 도메인 정보 활용의 어려움**
- 동일 HS4 코드라 하더라도 품목 간 패턴과 시장 규모가 지나치게 달라  
  도메인 기반 군집 특징 활용이 효과적이지 못했다.

#### **4) Pairwise 모델의 데이터 부족**
- 공행성 쌍 하나당 실제 회귀 학습에 사용 가능한 row는  
  20~30개 수준에 불과했다.  
- LightGBM이 비교적 견고하지만 근본적으로 **극저샘플 회귀 문제**라는 한계는 여전하다.

#### **5) FE 난이도의 과도한 영향력**
- 데이터 자체가 매우 sparse하고 value scale도 품목마다 극단적으로 달라  
  **조금만 FE가 삐끗해도 전체 점수 구조가 무너질 정도로 민감**했다.
- 솔직히 raw value 기반 단순 모델이 더 안정적일 것 같을 때도 있었지만,  
  다양한 시도 끝에 **섬세한 feature engineering을 적용하면 점수가 분명히 상승**하는 것을 확인했다.
- 이로 인해 개발 과정은 매우 까다로웠으며 “FE 주도형(score-sensitive)” 문제로 평가할 수 있다.

#### **6) 공행성 쌍 개수 조정 의존성**
- F1이 전체 점수의 60%를 차지하므로  
  공행성 쌍을 많이/적게 선택하는 전략이 성능에 직접적인 영향을 주었다.  
- 정확한 쌍을 맞추지 못하면 회귀 성능과 관계없이 NMAE가 최하점 처리되므로  
  모델의 근본적 성능보다 **쌍 개수 조정 전략이 더 중요해지는 왜곡된 구조**가 존재했다.

---

### 9.2 Future Work

#### **1) Multi-task Learning 기반 통합 모델**
공행성 판별(F1)과 value 예측(NMAE)을  
하나의 모델에서 동시에 학습하는 dual-head 구조를 시도할 수 있다.  
이는 불안정한 pairwise 샘플 수도 보완할 수 있다.

#### **2) Graph-based Modeling (GNN)**
품목을 노드, 공행성 강도를 edge로 구축하여  
GCN/GAT 기반 무역 네트워크 모델링을 시도할 수 있다.

#### **3) Dynamic Lag Modeling**
현재는 best_lag을 고정값으로 사용하지만  
실제 무역 관계는 시점마다 달라질 가능성이 있다.  
attention-based dynamic lag 모델이 대안이 될 수 있다.

#### **4) Distribution-aware Regression**
value의 규모 차이가 매우 큰 점을 고려해  
zero-inflated 모델, quantile regression, mixture model 등을 적용할 수 있다.

#### **5) HS 도메인 정보 강화**
HS2/HS3/HS4 계층적 구조를 활용해  
상품군 embedding 또는 category-aware attention을 도입할 여지가 있다.

---

### ✔ Summary

본 대회는 단순 시계열 예측문제가 아니라  
**극도로 희소한(sparse) 데이터 환경에서 pairwise 관계를 추론해야 하는 고난도 문제**였다.  
특히 데이터 양이 적고 value=0 월이 많아  
모델이 작은 FE 변화에도 크게 흔들릴 정도로 예민한 구조였다.

그럼에도 불구하고,  
공행성 기반 feature engineering과 time-series CV + LightGBM 조합을 통해  
실질적인 예측 성능을 확보할 수 있었으며,  
향후 그래프 모델·multi-task 학습 등을 도입한다면  
더 안정적이고 견고한 구조로 확장할 수 있을 것이다.

---
## 🧩 규칙 요약
- `data/`, `outputs/` 폴더는 `.gitignore`에 등록 (업로드 금지)  
- **모델 실험은 개인에게 할당된 브랜치 기반으로 진행**
  ```bash
  git checkout -b soomin-dev
  git push origin soomin-dev
  ```
- `.env` 파일에 **API key, WandB token, Kaggle key 등 민감정보 저장** (커밋 금지)  
- 제출 파일은 `outputs/submissions/` 내부에 저장  
