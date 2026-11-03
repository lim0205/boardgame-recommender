# Memory-based CF / Hybrid 구현 (담당 B)

이 문서는 `boardrec/memory_cf.py`, `boardrec/hybrid.py`의 구현 방식과 Validation 실험 방법을 정리합니다. **Test split은 튜닝에 사용하지 않습니다.**

## 1. Memory-based Collaborative Filtering — `boardrec/memory_cf.py`

`MemoryCFRecommender(kind, similarity, k, min_support=3, shrinkage=10, item_reg=10)`

| 설정 | 후보 | 의미 |
|---|---|---|
| `kind` | `user`, `item` | User-based kNN / Item-based kNN |
| `similarity` | `cosine`, `pearson` | 유사도 함수 |
| `k` | 10, 20, 40, 80 | 예측에 쓰는 이웃 수 |

**예측식.** SVD와 같은 Train 전용 기준선 위에 이웃들의 잔차 가중평균을 더합니다.

```text
b_ui = mean_u + item_bias_i                      (item_bias: 평균 대비 편차를 item_reg로 shrink)
User-based : r̂_ui = b_ui + Σ_v s(u,v)·(r_vi − b_vi) / Σ_v s(u,v)   (i를 평가한 사용자 중 유사도 상위 k명)
Item-based : r̂_ui = b_ui + Σ_j s(i,j)·(r_uj − b_uj) / Σ_j s(i,j)   (u가 평가한 게임 중 유사도 상위 k개)
```

- 유사도는 **함께 평가한 항목(co-rated)만**으로 계산합니다. 평가하지 않은 칸은 0점이 아닙니다.
  - `cosine`: 원래 1~10 평점의 코사인
  - `pearson`: 평균을 뺀 평점의 코사인 (User-based는 사용자 평균, Item-based는 게임 평균으로 중심화)
- 함께 평가한 수가 `min_support` 미만이면 유사도 0, `shrinkage`로 `n_common / (n_common + shrinkage)`를 곱해 근거가 적은 유사도를 줄입니다.
- 양수 유사도만 사용합니다. 이웃이 없으면 기준선 `b_ui`로 예측합니다.
- **Fallback:** Train에 없는 사용자는 `global_mean + item_bias` (SVD와 동일). Train 평점이 없는 게임은 item_bias 0.
- **인기도 미반영:** item_bias는 평점 *편차*의 평균이며 평가 수(인기도)가 아닙니다. 추천 점수에 popularity 항은 없습니다.
- 구현은 dense 행렬 + BLAS로 전체 유사도 행렬을 미리 계산합니다. 4,997명 기준 User-based 유사도 행렬은 약 100MB(float32)이고, 실험 중 최대 메모리는 약 1GB였습니다(동일 규모 합성 데이터로 확인).

## 2. Hybrid — `boardrec/hybrid.py`

```text
hybrid(u, i) = alpha · Content(u, i) + (1 − alpha) · CF(u, i)
```

- Content: 기존 `ContentBasedRecommender(feature_set="both")` (Validation RMSE로 이미 선택된 설정)
- CF 후보: **SVD** (10 factors, residual_scale 1.0 — 기존 Validation 선택값)와 **Memory-based CF** (위 실험의 선택값)
- `alpha ∈ {0.0, 0.1, …, 1.0}`. `alpha=1`은 순수 Content, `alpha=0`은 순수 CF이므로 단독 모델 결과도 같은 표에 나옵니다.
- 두 모델 모두 Train에만 fit, 같은 1~10 척도, 같은 후보 catalog(교집합)를 사용하므로 Coverage 비교가 유지됩니다.
- **선택 기준:** (CF 모델, alpha) 조합 중 Validation RMSE 최소 — 팀 기존 규칙과 동일. HitRate@10과 Coverage@10은 같은 250명으로 함께 보고합니다.
- `CachedPredictions`는 alpha 반복 시 각 구성 모델의 예측을 재사용하기 위한 캐시이며 결과값은 직접 호출과 동일합니다.

## 3. 실행 방법

`data/processed`가 준비되어 있어야 합니다 (`scripts/run_bgg.py`로 생성).

```bash
pip install -e ".[dev]"
python -m pytest -q

# 1) Memory-based CF: kind × similarity × k 비교 (Validation만 사용)
python scripts/run_memory_cf.py --data-dir data/processed --out outputs/bgg_memory_cf --max-hit-users 250

# 2) Hybrid: 위에서 선택된 Memory CF 설정과 SVD(10, 1.0)를 각각 Content와 alpha 결합
python scripts/run_hybrid.py --data-dir data/processed --out outputs/bgg_hybrid \
    --svd-factors 10 --svd-scale 1.0 \
    --memory-cf-results outputs/bgg_memory_cf/results.json --max-hit-users 250
```

출력 (집계값만, 개인 평점 없음):

- `outputs/bgg_memory_cf/validation_metrics.csv`, `results.json`, `validation_metrics.svg`
- `outputs/bgg_hybrid/validation_metrics.csv`, `results.json`, `validation_alpha.svg`

5,000명 × 1,000게임 규모 합성 데이터 기준 소요 시간: Memory CF 8개 설정 약 5분(16개 설정 약 10분), Hybrid 22개 조합 약 40초.

### 최종 Test (1회)

Validation 실험이 모두 끝난 뒤에 한 번만 실행합니다. 설정은 Validation 결과 파일에서 읽어오며, 이 단계에서는 아무것도 다시 고르지 않습니다.

```bash
python scripts/run_final_test.py --data-dir data/processed --out outputs/final_test
```

평가 대상: Content, SVD, Memory CF, Hybrid(Content+SVD), Hybrid(Content+Memory CF). Hybrid의 alpha는 Validation RMSE 기준으로 **0 < alpha < 1** 범위에서 고릅니다. alpha=0과 1은 각각 순수 CF와 순수 Content라서 이미 따로 보고되기 때문입니다. 결과는 `outputs/final_test/test_metrics.csv`, `results.json`에 저장됩니다.

## 4. 해석 시 유의점

- 기존 문서와 같이 Hit Rate@10은 held-out 양성(8점 이상)만 정답으로 보는 Offline 지표이며, 평가하지 않은 게임은 부정이 아닙니다.
- RMSE 기준으로 선택한 alpha와 Hit Rate가 가장 높은 alpha는 다를 수 있습니다. 보고서에는 alpha별 전체 표를 함께 제시하는 것을 권장합니다.
- 최종 Test 평가는 선택된 하나의 설정으로 한 번만 수행합니다. 기존 Test는 이미 탐색 단계에서 열람되었으므로 "완전히 새로운 Test"라고 표현하지 않습니다.

## 5. 실제 BGG 결과 (2026-11-03)

데이터: 기존과 동일한 전처리(게임 1,000 · 사용자 4,997 · seed 42). `run_bgg.py`로 재생성한 Content Test 값이 기존 기록(RMSE 1.245727, Hit Rate 3.2%)과 일치하여 split이 동일함을 확인했습니다. RMSE/MAE는 전체 held-out 평점(Validation 36,212 · Test 36,003), Hit Rate/Coverage는 같은 규칙의 250명 표본입니다.

### Memory-based CF 설정 비교 (Validation, 상위 5개)

| kind | similarity | k | RMSE | MAE | Hit Rate@10 |
|---|---|---:|---:|---:|---:|
| **item** | **pearson** | **40** | **1.0960** | **0.8293** | 15.6% |
| item | pearson | 80 | 1.0972 | 0.8302 | 15.2% |
| item | pearson | 20 | 1.0984 | 0.8317 | 15.2% |
| user | pearson | 80 | 1.1052 | 0.8364 | 16.8% |
| item | cosine | 80 | 1.1059 | 0.8361 | 14.8% |

Pearson(평균 중심화)이 Cosine보다 일관되게 좋았고, Item-based가 User-based보다 RMSE가 낮았습니다. 전체 16개: [memory_cf_validation_metrics.csv](../examples/real_bgg_memory_cf_hybrid/memory_cf_validation_metrics.csv)

### Hybrid alpha (Validation)

| CF | alpha | RMSE | Hit Rate@10 | Coverage@10 |
|---|---:|---:|---:|---:|
| SVD | 0.0 (순수 SVD) | 1.1109 | 18.8% | 4.8% |
| SVD | 0.1 | 1.1108 | 18.8% | 5.2% |
| SVD | 0.5 | 1.1403 | 17.2% | 16.4% |
| Memory CF | 0.0 (순수 Memory CF) | 1.0960 | 15.6% | 7.5% |
| Memory CF | 0.1 | 1.0982 | 14.4% | 8.0% |
| Memory CF | 0.5 | 1.1353 | 13.2% | 17.5% |
| — | 1.0 (순수 Content) | 1.2384 | 4.0% | 34.3% |

Content 비중을 높일수록 RMSE와 Hit Rate는 나빠지고 Coverage는 올라갑니다. 0~1 전체에서 RMSE 최소는 alpha=0(순수 Memory CF)이므로, 최종 Hybrid는 0 < alpha < 1 범위에서 Validation RMSE로 선택했습니다(두 CF 모두 alpha=0.1). 전체 표: [hybrid_validation_metrics.csv](../examples/real_bgg_memory_cf_hybrid/hybrid_validation_metrics.csv)

### 최종 Test (Validation에서 고정한 설정, 1회 평가)

| 모델 | RMSE ↓ | MAE ↓ | Hit Rate@10 ↑ | Coverage@10 ↑ |
|---|---:|---:|---:|---:|
| Content (Mechanics+Themes) | 1.2457 | 0.9399 | 3.2% (8/250) | **31.5%** |
| SVD (k=10) | 1.1157 | 0.8447 | **10.8% (27/250)** | 4.2% |
| Memory CF (item, pearson, k=40) | **1.0989** | **0.8296** | 6.8% (17/250) | 8.1% |
| Hybrid: 0.1·Content + 0.9·SVD | 1.1161 | 0.8449 | **10.8%** | 4.6% |
| Hybrid: 0.1·Content + 0.9·Memory CF | 1.1019 | 0.8317 | 6.8% | 8.7% |

### 해석

- **평점 예측:** Memory CF(Item-based Pearson)가 Validation과 Test 모두에서 RMSE와 MAE가 가장 낮습니다. 순위가 두 split에서 동일하여 안정적입니다.
- **Top-10 적중:** SVD가 가장 높습니다. 다만 Test Hit Rate는 모든 CF 모델에서 Validation보다 크게 낮아졌습니다(SVD 18.8% → 10.8%). 250명 표본에서 1명이 0.4%p이고 10% 근처의 표준오차가 약 ±1.9%p이므로, 모델 간 몇 %p 차이는 표본 변동일 수 있습니다.
- **Coverage:** Content가 압도적으로 넓고, CF 모델은 소수 게임에 추천이 집중됩니다.
- **Hybrid:** 이 데이터에서 Content 신호가 CF보다 약해서, Hybrid는 정확도를 높이기보다 **정확도와 추천 다양성 사이의 trade-off를 조절하는 수단**으로 작동했습니다. alpha=0.1은 CF와 거의 같고, alpha=0.5 근처에서 Hit Rate를 약간 내주면 Coverage가 약 3배가 됩니다.
- Test는 이전 탐색 단계에서 이미 열람된 적이 있으므로 완전히 새로운 holdout은 아닙니다.
