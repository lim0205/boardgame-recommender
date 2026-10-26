# Board Game Recommender — ML Term Project

Python 프로젝트: **데이터 전처리 + Content-based Filtering + 공통 Evaluation**.
두 명이 독립적으로 작업한 모델을 같은 데이터/평가 규칙으로 연결하도록 설계했습니다.

> **상태:** 인공 데이터 및 사용자가 제공한 BGG ZIP으로 **실제 실험 완료(4,997명 / 362,787평점)**. 자세한 숫자와 제약: [실제 데이터 실험 보고서](docs/ACTUAL_BGG_RUN.md).
> **데이터 사용 주의:** BGG XML API 약관에는 사이트/API 데이터의 AI 학습 목적 사용을 제한하는 조항이 있습니다. Kaggle 게시 여부와 무관하게 실제 데이터 이용 허가 범위를 먼저 확인하세요. 이 저장소에는 BGG 데이터가 없습니다.

## 최신 실제 데이터 추천 순위 개선 (2026-10-26)

[Popularity Baseline + Content/Popularity Ranking 실제 BGG 실험](docs/RANKING_IMPROVEMENT.md).
Validation Hit Rate@10(1,000명): 인기순 33.9%, 콘텐츠 10%+인기도 90% 32.2%, 순수 콘텐츠 6.1%.
같은 Test 1,000명: 인기순 30.6%, 콘텐츠 10% 혼합 29.9%, 순수 콘텐츠 4.7%.

**성능 최고인 인기순(α=0)은 개인화 추천이 아닙니다.** α=0.1은 추천 다양성을 높이는 절충안입니다. 새로운 순위 점수는 평점 1~10이 아니므로 RMSE 계산에 사용하지 않습니다. Test 분할은 이전 실험에서 이미 살펴본 적이 있어 결과는 탐색적 비교로 표기합니다.

```bash
python scripts/run_ranking.py --prepared-dir data/processed --out outputs/bgg_ranking --max-users 1000 --rights-confirmed
```

## 담당 범위

| 파트 | 코드 | 담당 |
|---|---|---|
| CSV 로딩, 데이터 정제/필터링, 사용자별 80/10/10 분할 | `boardrec/preprocessing.py` | A (본인) |
| Mechanics/Themes 기반 Content-based Filtering | `boardrec/content.py` | A (본인) |
| RMSE / MAE / Hit Rate@10 | `boardrec/evaluation.py` | A (본인) |
| 특징 조합 비교, validation tuning, test, 그래프 | `boardrec/experiments.py` | A (본인) |
| 공통 API 연결 어댑터 | `boardrec/adapters.py` | A (본인) |
| Popularity Baseline + Content/Popularity Ranking + Validation tuning | `boardrec/ranking.py`, `boardrec/ranking_experiments.py` | A (본인) |
| Memory-based CF + SVD + Hybrid | `boardrec/memory_cf.py`, `boardrec/svd.py`, `boardrec/hybrid.py` (추가 예정) | B (팀원) |

## 요구 사항

- Python 3.10 이상
- `pip install -e ".[dev]"`
- `python -m pytest -q` 로 테스트

## 먼저 작동 확인하기 — 외부 데이터 필요 없음

```bash
python -m boardrec.cli demo --top-games 35 --min-user-ratings 8 --max-users 25
```

무작위로 **생성한 인공 데이터**를 사용하여 전처리→모델 평가까지 실행합니다. 생성된 `outputs/demo/results.json`, `metrics.csv`, `validation_rmse.png`, `example_top10.csv`는 예제 실험 결과이며 실제 추천 성능이 아닙니다.

## 실제 CSV 실행 방법 (사용 허가가 확인된 경우만)

원본 ZIP을 직접 준비한 경우 권장 실행 방법 (`max-users=5000`, 약 4GB 이상의 메모리 권장):

```bash
python scripts/run_bgg.py --archive "data/raw/archive (1).zip" --rights-confirmed --max-users 5000
```

3-pass streaming 경로가 필요하다면 `--low-memory`를 추가하세요. 아래의 수동 2단계 명령도 유지되지만 대용량 CSV에서는 느립니다.


원본 4개 CSV 파일을 `data/raw/`에 직접 배치합니다. 원본은 Git에 포함되지 않습니다.

```text
data/raw/
  user_ratings.csv
  games.csv
  mechanics.csv
  themes.csv
```

전처리:

```bash
python -m boardrec.cli prepare --ratings data/raw/user_ratings.csv --games data/raw/games.csv --mechanics data/raw/mechanics.csv --themes data/raw/themes.csv --top-games 1000 --min-user-ratings 20 --max-users 20000 --seed 42
```

모델 실험:

```bash
python -m boardrec.cli run --data-dir data/processed --out outputs --max-hit-users 500
```

수량은 예시이며 사전에 정한 기준대로 실험한 뒤 보고서에 최종 값과 선택 이유를 남기세요. 무작위 시드는 **42로 고정**되어 누구나 같은 분할을 복원할 수 있습니다.

## 준비된 입력/출력 규격

`data/processed` 폴더의 세 평점 파일은 전부 동일한 컬럼을 사용합니다.

| 파일 | 열 | 용도 |
|---|---|---|
| `train.csv` | `user_id`, `game_id`, `rating` | 유사도와 모델 학습만 |
| `val.csv` | 동일 | 하이퍼파라미터 / feature / α 선택 |
| `test.csv` | 동일 | 선택된 설정의 최종 평가, 한 번만 |
| `games.csv` | `game_id` + 원본 속성 | 표시용 게임 정보 |
| `mechanics.csv` | `game_id` + binary features | 콘텐츠 특징 |
| `themes.csv` | `game_id` + binary features | 콘텐츠 특징 |
| `manifest.json` | 건수, 시드, 필터 기준 | 재현성 기록 |

원본에 `Username`/`BGGId`/`Rating`처럼 이름이 달라도 위 세 열로 통일해 저장합니다. **ID는 전부 문자열(str)** 입니다. `0`은 미관측 평점이 아닌 0/1 특징에만 사용합니다. 전체 `users × games` 밀집 배열은 만들지 않습니다.

## 내용 기반 추천 방식

1. `mechanics.csv` / `themes.csv`의 0/1 컬럼을 조합하여 sparse feature matrix를 만듭니다.
2. 각 게임 벡터를 L2 정규화해 **Cosine Similarity**를 계산합니다.
3. 각 사용자에 대해 **훈련 데이터에 있는 평점**만 이용해 평균 및 취향을 구합니다.
4. 예측할 게임과 평가한 게임들의 유사도를 가중치로 사용하여 사용자 평균에서의 **평점 편차**를 평균합니다.
5. 학습용 평점이 있는 게임을 제외하고 예상 평점 상위 10개를 추천합니다.

수식 (평점이 존재하는 학습 게임 집합을 `I(u)`라 표기):

```text
μ_u = mean(r_ui for i in I(u))
s(i,j) = cosine(feature_i, feature_j)

r_hat(u,j) = μ_u + Σ_(i in I(u)) [s(j,i) × (r_ui - μ_u)] / Σ_(i in I(u)) s(j,i)
```

분모가 0이면 `μ_u`, 사용자 기록 자체가 없으면 `global_train_mean`을 반환합니다. 마지막 값은 1~10으로 clip합니다. **신규 사용자에게 반환하는 전체 평균은 개인화 추천이 아닙니다.** 별도 온보딩 설문 없이는 사용자 Cold Start가 해결되지 않습니다.

## 공통 평가

- **RMSE**: `sqrt(mean((prediction-rating)^2))` — 낮을수록 좋음.
- **MAE**: `mean(abs(prediction-rating))` — 낮을수록 좋음.
- **Hit Rate@10**: test(또는 validation)에서 8점 이상인 항목이 하나 이상 있는 **평가 가능한 사용자**에 대해, 학습에서 관측되지 않은 게임 전체를 대상으로 Top-10을 만든 뒤 숨긴 긍정 항목이 하나라도 있는 사용자 비율. 높을수록 좋음.
- 일반적인 `Precision@10`이 아닙니다. **평점이 없는 게임을 음성 레이블(싫어함)로 취급하지 않습니다.** 이 방식도 숨겨진 긍정 피드백이 매우 적다는 평가 편향이 있고 실제 사용자 만족도와 같지 않습니다.
- Hit Rate의 `eligible_users`, `evaluated_users`도 반드시 함께 보고하세요. 계산비용을 줄이기 위해 기본 최대 500명만 사용하며 **시드 고정 무작위 추출**입니다.
- 특징 조합(`mechanics`, `themes`, `both`)은 `val`의 RMSE가 최소인 설정으로 고르고, 오직 그 설정만 `test`에 평가합니다.

## 팀원이 붙이는 법

`docs/TEAM_HANDOFF.md`에 필요한 파일/함수/라이브러리 예시가 있습니다. 핵심은 두 줄입니다.

```python
model.fit(train, games, mechanics, themes)    # Content-based 전용
model.predict("user_id", "game_id")           # 모든 모델의 공통 규격 -> float 1..10
```

SVD 또는 Memory-CF에서 `predict_many`, `recommend`를 직접 구현하기 어렵다면 `boardrec.adapters.PredictAdapter`로 `predict()`만 연결하세요. Hybrid에는 Content-based와 CF 두 모델의 예측 점수를 같은 1~10 스케일로 사용하고 α는 **validation에서만 선택**합니다.

## 실제 실험 결과

실제 BGG 파일로 5,000명을 표본 추출해 실험했고, 마지막에 4,997명이 남았습니다. Validation 최저 RMSE 모델은 Mechanics+Themes였으며, Test RMSE **1.2457**, Test MAE **0.9399**, sampled Hit Rate@10 **0.032**였습니다. 상세한 실험 조건/한계 및 그래프는 [docs/ACTUAL_BGG_RUN.md](docs/ACTUAL_BGG_RUN.md)와 [examples/real_bgg_5000/](examples/real_bgg_5000/)에 있습니다. 원본 데이터, 개인 식별이 가능한 계정명/평점은 저장소에 업로드하지 않습니다.

## 파일 목록

```text
boardrec/
  cli.py             # prepare / run / demo 명령어
  preprocessing.py   # 메모리 절약형 3-pass CSV 읽기
  bgg_fast.py        # BGG ZIP 실제 데이터용 1-pass 고속 전처리
  bgg_io.py          # ZIP 및 CSV 구조 검사
  content.py         # ContentBasedRecommender
  evaluation.py      # rating_metrics, hit_rate_at_k, evaluate
  experiments.py     # feature 비교 및 결과 저장
  adapters.py        # 팀원 모델 predict 함수 연결
  demo_data.py       # 전적으로 synthetic인 샘플 생성
examples/
  teammate_integration.py   # 통합 예제: 실제 CF 대신 단순 baseline

docs/TEAM_HANDOFF.md        # 팀원에게 전달할 계약

tests/                     # 자동 단위/통합 테스트
```

## 라이브러리 및 데이터 참조

- Scikit-learn — cosine-compatible sparse L2 normalization: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html
- BGG XML API 문서: https://boardgamegeek.com/wiki/page/BGG_XML_API2
- BGG XML API 이용약관: https://boardgamegeek.com/wiki/page/XML_API_Terms_of_Use
- 예시로 기획한 Kaggle 출처: https://www.kaggle.com/datasets/threnjen/board-games-database-from-boardgamegeek (실제 데이터/학습 권한 사전 확인 필수)

추가 자료: [`docs/GITHUB_PUBLISH.md`](docs/GITHUB_PUBLISH.md)에는 팀원 공유/게시 방법을 적어뒀습니다. [`examples/synthetic_results/`](examples/synthetic_results/)에는 실제 데이터가 아닌 **가상 데이터 실행 결과**를 첨부했습니다.

## 제한사항

- 최초 게임 선별에서 전체 평점 파일의 **평가 개수**를 사용하므로 split 후 holdout의 항목 존재 여부가 데이터셋 선택에 간접적으로 반영될 수 있습니다. 엄격한 inductive 평가라면 catalog 선정 규칙을 학습 데이터만으로 다시 설계해야 합니다.
- 2026-10-26 기준 실제 BGG ZIP의 헤더/형태와 연결하여 4,997명 표본 실험을 수행했습니다. 다만 권한 문제는 별도이며 20,000명 실행은 완료되지 않았습니다.
- 트레이닝 기록이 없거나 feature overlap이 없는 데이터의 평점은 mean fallback을 사용합니다.
- Top-K는 **학습에서 미평가한 게임**이라는 뜻이며, 실제 플레이하지 않은 게임과 같다고 단정하지 않습니다.
- 제안 시스템/콘텐츠 기반은 학습 데이터를 사용하는 것을 전제하므로 BGG 권리자로부터의 허가 문제를 해결하기 전에는 synthetic demo만 사용하세요.
