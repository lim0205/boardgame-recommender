# BGG 원본 데이터 연결과 실험

현재 저장소의 12개 자동 테스트는 가상 CSV 또는 테스트 fixture 기반이다. **실제 BGG 데이터를 내려받거나 실제 데이터로 학습·평가한 결과는 없다.**

## 원본과 권한

기획 출처: [Kaggle: Board Games Database from BoardGameGeek (threnjen)](https://www.kaggle.com/datasets/threnjen/board-games-database-from-boardgamegeek).

BGG 약관에는 AI 모델 훈련 목적 데이터 이용 제한이 있으므로, 학습 사용 권한을 먼저 확인한다. Kaggle 공개 다운로드와 ML 학습 허가는 별개다. 이 저장소에 원본 CSV나 사용자별 평점 기록을 커밋하지 않는다.

## 준비 단계 (권한 확인 후)

1. 권한 있는 원본 데이터 ZIP을 확보한다. 원본 이름이 무엇이든 `--archive` 경로로 지정할 수 있다.
2. `pip install -e ".[dev]"`
3. `python -m pytest -q`
4. `python scripts/run_bgg.py --archive data/raw/bgg_dataset.zip --rights-confirmed`

또는 직접 압축 해제한 네 개의 CSV로:

```bash
python scripts/run_bgg.py --csv-dir data/raw --rights-confirmed
```

필수 CSV: `games.csv`, `user_ratings.csv`, `mechanics.csv`, `themes.csv`. 이 파일명·컬럼은 기획한 데이터셋 버전 기준이다. 다른 버전인 경우 실제 헤더를 비교하고 매핑을 수정한다.

## 처리/출력

`data/processed/`에 train/val/test (80/10/10) CSV 및 `manifest.json` 출력.
`outputs/bgg/`에 검증 지표, 그래프, 최종 테스트 지표 및 Top-10 예시 출력.

선호도 비교용 실험은 validation에서 `mechanics`, `themes`, `both` 중 최소 RMSE를 고르고, 선택된 모델만 test에서 평가한다. 정보 누출과 차가운 시작(Cold Start), 테스트 후보 제한은 README 제한사항 참조.

`--rights-confirmed`는 데이터 사용자의 확인 플래그이며 실제 이용 허가를 자동으로 부여하지 않는다.

## 확인 완료

- ZIP에 필수 CSV가 누락/중복일 때 오류 처리
- BGGId, Username, Rating 헤더 검사
- 대용량 평점 데이터를 메모리 한꺼번에 올리지 않도록 CSV chunk 읽기 (전처리 3-pass)
- 가상 데이터/fixture 기반 테스트 12개 통과

## 미완료

- 실제 CSV 획득 및 권한 확인
- 실제 CSV 데이터 품질/라이브러리 호환성 검증
- 실제 BGG 기반 실험 성능 및 그래프 생성
