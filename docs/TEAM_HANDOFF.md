# 팀원 인수인계: Collaborative Filtering / SVD / Hybrid 구현 가이드

## 1. 먼저 실행할 명령

```bash
pip install -e ".[dev]"
python -m boardrec.cli demo --top-games 35 --min-user-ratings 8 --max-users 25
python -m pytest -q
```

먼저 `data/demo/processed/`의 샘플 split으로 개발 후, 합법적으로 사용 가능한 실제 데이터를 준비한 다음 같은 포맷으로 바꾸세요.

## 2. 같은 데이터만 사용

```python
from boardrec.preprocessing import load_prepared
train, val, test, games, mechanics, themes = load_prepared("data/processed")
```

- `train`, `val`, `test`: user_id(str), game_id(str), rating(float; 1~10)
- games/mechanics/themes: game_id(str) + 나머지 게임 속성
- 평가할 모델이 달라도 **정확히 같은 split**이 필요합니다.
- 훈련과 사용자 프로필/평균/유사도를 만들 때 val/test 평점을 사용하지 마세요.

## 3. 반드시 맞춰야 하는 인터페이스

```python
class MyCollaborativeRecommender:
    def fit(self, train):
        ...
        return self

    def predict(self, user_id: str, game_id: str) -> float:
        """1~10점 예상 평점을 float으로 반환."""
        ...
```

비교 함수와 평가 함수는 `predict_many()` / `recommend()`도 사용합니다. 복잡하게 직접 만들지 않아도 다음 어댑터로 해결할 수 있습니다.

```python
from boardrec.adapters import PredictAdapter
from boardrec.evaluation import evaluate

base = MyCollaborativeRecommender().fit(train)
recommender = PredictAdapter(
    predict_fn=base.predict,
    train=train,
    catalog_game_ids=games.game_id,
)
validation_scores = evaluate(recommender, val, max_hit_users=500)
print(validation_scores)
```

SVD Surprise의 `predict()` 결과는 보통 prediction 객체입니다. 이를 `float(surprise_model.predict(uid=str(user_id), iid=str(game_id)).est)`로 변환하여 어댑터에 넘기면 됩니다. SVD가 학습되지 않은 게임/사용자를 만나는 경우 어떤 fallback이 적용되는지 보고서에 명시하세요.

`PredictAdapter`는 쉽게 통합할 수 있도록 개별 예측을 반복합니다. 규모가 큰 실험에서는 효율적인 `predict_many()`를 CF 모델에 직접 구현할 수 있습니다.

## 4. Hybrid 연결 규칙

```python
# Only after validating each model separately:
content_score = content_model.predict(user_id, game_id)
cf_score = cf_model.predict(user_id, game_id)
hybrid_score = alpha * content_score + (1.0 - alpha) * cf_score
```

- 두 모델 모두 1~10점 척도 사용.
- `alpha in {0.0, 0.1, ..., 1.0}` 등의 후보를 **val에서 비교**.
- 선택된 하나의 alpha를 test에 한 번 평가.
- Content-based의 특성 선택 또한 train/val만 사용.

## 5. 보고서/발표에서 비교할 실험

| 모델 | 바꿔볼 설정 | 기본 평가 |
|---|---|---|
| Memory-based CF | User/Item-based, Cosine/Pearson, k | RMSE, MAE, Hit Rate@10, Catalog Coverage@10 |
| SVD | n_factors, reg_all, lr_all, n_epochs | 동일 |
| Content-based | mechanics / themes / both | 동일 |
| Hybrid | CF 선택 + alpha | 동일 |

**Popularity는 최종 추천 점수에 추가하지 않습니다.** 네 가지 평가 지표를 모든 모델에 동일하게 적용합니다.

모델이 많은 경우 연산 시간이 오래 걸리므로 baseline과 작은 데이터로 기능 검증을 한 뒤 실험 규모를 늘리세요. 비교 시 데이터 필터, user split, seed, 평가 사용자 표본을 고정하세요.

## 6. PR 또는 파일로 합치기

**Private 저장소:** 팀원은 먼저 저장소 소유자에게 Collaborator로 초대받아야 push할 수 있습니다. GitHub의 Repository Settings → Collaborators에서 Write 권한으로 초대하세요.

아래 파일에 팀원 코드를 추가하길 권장합니다.

```text
boardrec/memory_cf.py
boardrec/svd.py
boardrec/hybrid.py
```

공통 `preprocessing.py`와 `evaluation.py`는 수정 전에 변경 이유를 공유하세요. 기능 추가 시 `tests/`에 대응 테스트도 같이 작성하세요. 데이터 원본, 대규모 가공 데이터, GitHub 토큰 등은 절대로 커밋하지 마세요.

## 7. 알려진 평가 제약

오프라인 held-out 10%는 한 사용자가 평가한 게임 수가 20개라면 2개 수준이어서, 추천 10개에 해당 게임이 나타날 확률이 낮을 수 있습니다. Hit Rate@10은 평가 가능한 사용자에게만 정의되며, 순위 추천의 실제 효과는 클릭/플레이 피드백 없이 확정할 수 없습니다.
