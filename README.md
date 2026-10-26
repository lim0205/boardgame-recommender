# Board Game Recommender — Machine Learning Term Project

실제 BoardGameGeek(BGG) 평점과 게임 메타데이터를 이용하는 보드게임 추천 시스템 프로젝트입니다. **현재 구현된 추천기는 순수 Content-based Filtering이며, 추천 점수에 Popularity를 혼합하지 않습니다.**

## 프로젝트 현황

- **구현 완료 (담당 A):** BGG CSV 전처리, Mechanics/Themes 기반 Content-based Filtering, 공통 평가, 실험 실행, 테스트
- **팀원 구현 예정 (담당 B):** Memory-based Collaborative Filtering, SVD (Model-based CF), 최종 Hybrid (Content + CF)
- **실제 데이터 실험:** 게임 1,000개 · 사용자 4,997명 · 평점 362,787건. 최종 비교 표본이 아닌, 우선 완료된 5,000명 규모 실험입니다.
- **원본 데이터는 저장소에 없음.** 원본 평점/Username/개인별 학습 데이터는 커밋하지 않습니다.

## 파일 구조

```text
boardrec/
  preprocessing.py     # CSV 정제, 게임/사용자 필터, 사용자별 데이터 분할
  bgg_fast.py          # 실제 대용량 BGG CSV 고속 전처리
  bgg_io.py            # ZIP 압축 해제 / 파일·컬럼 확인
  content.py           # Content-based: Mechanics/Themes + Cosine Similarity
  evaluation.py        # RMSE, MAE, Hit Rate@10, Catalog Coverage@10
  experiments.py       # Validation 모델 선택, 평가·그래프
  adapters.py          # 팀원 모델 공통 인터페이스 연결
  cli.py               # 실행 명령
  demo_data.py         # 기능 검증용 Synthetic data 생성
  memory_cf.py         # 팀원이 추가 예정 (아직 없음)
  svd.py               # 팀원이 추가 예정 (아직 없음)
  hybrid.py            # 팀원이 추가 예정 (아직 없음)
scripts/run_bgg.py     # 실제 BGG ZIP 실행
tests/                 # 자동 테스트
docs/TEAM_HANDOFF.md   # 팀원 개발·통합 가이드
docs/EVALUATION_POLICY.md  # 최종 평가 기준
docs/ACTUAL_BGG_RUN.md # 실제 데이터 실험 기록
examples/real_bgg_5000/   # 1차 실험 집계 결과
examples/real_bgg_official_1000/ # Validation 집계 결과
```

## PPT 계획에 따른 전처리

1. 전체 데이터에서 평점이 많은 **상위 게임 1,000개** 선정
2. 이 게임들에 대해 **평점 20개 이상** 기록한 사용자만 남김
3. 너무 크면 사용자 무작위 표본 추출 (seed=42)
4. 중복 사용자-게임 평점 정리 후 **사용자별 Train 80% / Validation 10% / Test 10%** 분할

**주의:** 기획 PPT의 사용자 **1만~2만 명**은 실험 규모 후보였습니다. 현재 실제 실행 완료는 **최대 5,000명 추출 → 정제 후 4,997명**이며, 1만~2만 명 실험이 완료된 것은 아닙니다. 게임 선별은 분할 전 전체 평점 수를 기준으로 하므로 선택 편향이 존재하며, **추천 점수에 인기도를 넣는 것과는 다릅니다.**

## 설치 및 실행

Python 3.10 이상.

```bash
pip install -e ".[dev]"
python -m pytest -q
python -m boardrec.cli demo --top-games 35 --min-user-ratings 8 --max-users 25
```

실제 BGG 데이터 사용권한을 확인하고, ZIP을 `data/raw/`에 준비한 경우:

```bash
python scripts/run_bgg.py --archive "data/raw/archive (1).zip" --rights-confirmed --top-games 1000 --min-user-ratings 20 --max-users 5000 --seed 42
```

사용권한 확인 없이 `--rights-confirmed`를 사용하면 안 됩니다. Kaggle에서 내려받을 수 있다는 사실 자체가 머신러닝 학습 허가를 의미하지는 않습니다. RAM이 부족하면 `--low-memory`를 추가합니다.

입력/출력 형식: `train.csv`, `val.csv`, `test.csv`는 공통으로 `user_id, game_id, rating` (IDs are strings, ratings 1–10)을 사용합니다. `data/raw`, `data/processed`, `outputs`는 Git에서 제외합니다.

## 최종 평가 지표 — 4개

| Metric | 목적 |
|---|---|
| **RMSE** ↓ | 평점 예측 오차 (큰 오차에 민감) |
| **MAE** ↓ | 평점 예측 평균 절대 오차 |
| **Hit Rate@10** ↑ | Validation/Test에서 실제 8점 이상을 준 게임이 Top-10에 하나 이상 포함된 사용자 비율 |
| **Catalog Coverage@10** ↑ | 평가 대상 사용자 전체의 Top-10에 등장한 서로 다른 게임 수 / 공통 후보 게임 수 |

게임에 평점이 없다는 것은 **싫어한다는 뜻이 아니므로**, Hit Rate는 관측된 선호 항목에 대한 Offline 지표입니다. 평가 대상 사용자 수와 후보 게임 수를 함께 명시합니다. 상세 원칙: [Evaluation Policy](docs/EVALUATION_POLICY.md).

현재 순수 Content-based (Mechanics + Themes) **Validation** 결과: RMSE **1.2384**, MAE **0.9390**, Hit Rate@10 **6.1%**, Catalog Coverage@10 **50.4%**. 마지막 두 지표는 1,000명 표본입니다. **Test 결과와 혼동하지 마세요.**

## 팀원 작업

1. 저장소를 **clone**한 뒤 `pip install -e ".[dev]"` 실행.
2. `boardrec/memory_cf.py`, `boardrec/svd.py`, `boardrec/hybrid.py` 파일을 추가.
3. `predict(user_id, game_id) -> float (1~10)` 인터페이스와 공통 전처리 데이터를 재사용.
4. 같은 Validation 데이터로 설정을 고른 후 **RMSE / MAE / Hit Rate@10 / Coverage@10**으로 비교하고, Test를 튜닝에 사용하지 않음.
5. 새 모델마다 `tests/`에 테스트를 추가하고 Pull Request로 합치는 것을 권장.

구체적인 연결 예제는 [TEAM_HANDOFF](docs/TEAM_HANDOFF.md) 참조. **Private 저장소이므로 팀원은 GitHub `Settings → Collaborators`에서 따로 초대해야 push할 수 있습니다.**

### BGG 데이터 이용 유의

BGG 사이트/API 데이터에는 AI 학습 관련 이용 제한이 있습니다. 데이터 제공 및 사용 권한을 별도로 확인하세요. 원본/사용자별 평가 로그를 GitHub에 올리지 마세요.
