# Board Game Recommender — Machine Learning Term Project

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
  memory_cf.py          # 구현 완료: User/Item-based kNN CF
  memory_cf_experiments.py # Memory CF Validation 실험
  svd.py                # 구현 완료: Truncated SVD
  svd_experiments.py    # 구현 완료: SVD Validation 실험
  hybrid.py             # 구현 완료: alpha·Content + (1−alpha)·CF
  hybrid_experiments.py # Hybrid alpha Validation 실험
  final_test.py         # Validation에서 고정한 설정의 Test 1회 평가
scripts/run_bgg.py     # 실제 BGG ZIP 실행
scripts/run_memory_cf.py, scripts/run_hybrid.py # Memory CF / Hybrid Validation 실행
scripts/run_final_test.py # 선택된 모델들의 최종 Test 1회 평가
tests/                 # 자동 테스트
docs/TEAM_HANDOFF.md   # 팀원 개발·통합 가이드
docs/EVALUATION_POLICY.md  # 최종 평가 기준
docs/MEMORY_CF_HYBRID.md   # Memory CF / Hybrid 구현 및 실행 방법
docs/ACTUAL_BGG_RUN.md # 실제 데이터 실험 기록
examples/real_bgg_5000/   # 1차 실험 집계 결과
examples/real_bgg_official_1000/ # Validation 집계 결과
```

## PPT 계획에 따른 전처리

1. 전체 데이터에서 평점이 많은 **상위 게임 1,000개** 선정
2. 이 게임들에 대해 **평점 20개 이상** 기록한 사용자만 남김
3. 너무 크면 사용자 무작위 표본 추출 (seed=42)
4. 중복 사용자-게임 평점 정리 후 **사용자별 Train 80% / Validation 10% / Test 10%** 분할


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

## External APIs and parameter reference

See [External Python APIs, model parameters, and reproducible commands](docs/EXTERNAL_APIS.md) for the source-code explanations required by the project specification.
