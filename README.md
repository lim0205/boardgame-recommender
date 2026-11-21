# Board Game Recommendation System

Machine Learning Term Project — Team 4

This project compares several ways to recommend board games from BoardGameGeek ratings. Given a user's rating history, each model estimates ratings for other games and returns a Top-10 list. The implementation is a Python experiment pipeline, not a website.

## Models

| Method | Implementation | Chosen setting |
|---|---|---|
| Content-based Filtering | Game mechanics and themes, cosine similarity | Mechanics + Themes |
| Model-based CF | Truncated SVD on rating residuals | 10 factors, residual scale 1.0 |
| Memory-based CF | User/item kNN with cosine or Pearson similarity | Item-based, Pearson, k=40 |
| Hybrid Filtering | Weighted average of Content-based and CF predictions | 10% Content + 90% CF |

Parameters were selected using Validation RMSE. The final Test results use those fixed settings.

## Final Test results

| Model | RMSE ↓ | MAE ↓ | Hit Rate@10 ↑ | Recall@10 ↑ | Coverage@10 ↑ |
|---|---:|---:|---:|---:|---:|
| Content-based | 1.2457 | 0.9399 | 3.2% | 0.97% | 31.5% |
| SVD | 1.1157 | 0.8447 | **10.8%** | **4.13%** | 4.2% |
| Memory-based CF | **1.0989** | **0.8296** | 6.8% | 3.14% | 8.1% |
| Hybrid + SVD | 1.1161 | 0.8449 | **10.8%** | 4.11% | 4.6% |
| Hybrid + Memory CF | 1.1019 | 0.8317 | 6.8% | 3.09% | 8.7% |

Memory-based CF had the lowest rating prediction error. SVD had the highest Recall@10 and tied with Hybrid + SVD in Hit Rate@10. Content-based Filtering covered the largest number of different games.

All models were evaluated on the same Test split. RMSE and MAE use 36,003 observed ratings. Ranking metrics use 250 users sampled with seed 42 from users with at least one Test rating of 8 or higher. **Recall@10 is the average of each user's recall (Macro Recall)**, not the pooled number of retrieved games divided by all relevant games. Coverage uses a common catalog of 1,000 games.

The full-precision results are available in [final_test_metrics.csv](examples/real_bgg_memory_cf_hybrid/final_test_metrics.csv) and [final_test_results.json](examples/real_bgg_memory_cf_hybrid/final_test_results.json). Validation tables and plots are also included under `examples/`.

## Data and preprocessing

We used the [Board Games Database from BoardGameGeek](https://www.kaggle.com/datasets/threnjen/board-games-database-from-boardgamegeek) dataset.

1. Keep the 1,000 games with the most rating records.
2. Keep users with at least 20 ratings among those games.
3. Sample up to 5,000 users (seed 42).
4. Average duplicate user–game ratings, then recheck the user threshold.
5. Split each user's ratings into Train / Validation / Test (about 80% / 10% / 10%).

The final dataset has **4,997 users, 1,000 games and 362,787 ratings**. The exact split counts are 290,572 Train, 36,212 Validation and 36,003 Test.

Missing ratings are not treated as zero-star ratings. User profiles, similarity matrices and latent factors use Training data only. The game catalog was chosen by rating count, but **no popularity term** was added to the recommendation scores.

## Code structure

| Path | Role |
|---|---|
| `boardrec/bgg_io.py`, `bgg_fast.py`, `preprocessing.py` | BGG data loading, cleaning and splitting |
| `boardrec/content.py` | Content-based model |
| `boardrec/svd.py` | Truncated SVD |
| `boardrec/memory_cf.py` | Neighborhood-based CF |
| `boardrec/hybrid.py` | Combined predictions |
| `boardrec/evaluation.py` | RMSE, MAE, Hit Rate@10, Recall@10 and Coverage@10 |
| `boardrec/*_experiments.py`, `final_test.py` | Model selection and final comparison |
| `scripts/` | Commands for running the experiments |
| `tests/` | Automated tests |

The code has comments at the main calculation and data-handling steps. For additional details about library functions and parameters, see [External APIs and Parameters](docs/EXTERNAL_APIS.md). The definitions of the five evaluation metrics are in [Evaluation Protocol](docs/EVALUATION_POLICY.md).

## Running the project

Python 3.10 or later is required. From the repository folder:

```bash
pip install -e ".[dev]"
python -m pytest -q
```

A small synthetic example can be run without the BGG dataset:

```bash
python -m boardrec.cli demo --top-games 35 --min-user-ratings 8 --max-users 25
```

To run the BGG experiment, obtain the original dataset and verify permission to use it. Place the ZIP at `data/raw/archive (1).zip`:

```bash
python scripts/run_bgg.py --archive "data/raw/archive (1).zip" --rights-confirmed --top-games 1000 --min-user-ratings 20 --max-users 5000 --seed 42
python scripts/run_svd.py --data-dir data/processed --out outputs/bgg_svd --max-hit-users 250
python scripts/run_memory_cf.py --data-dir data/processed --out outputs/bgg_memory_cf --max-hit-users 250
python scripts/run_hybrid.py --data-dir data/processed --out outputs/bgg_hybrid --svd-factors 10 --svd-scale 1.0 --memory-cf-results outputs/bgg_memory_cf/results.json --max-hit-users 250
python scripts/run_final_test.py --data-dir data/processed --out outputs/final_test --max-hit-users 250
```

The final command saves `test_metrics.csv` and `results.json` under `outputs/final_test/`, including Recall@10. Original rating data, user histories and locally processed files are not included in this repository.

## Notes

These are offline results based on observed ratings, not measurements of user satisfaction. Unrated games are unknown, and evaluating only the most-rated 1,000 games limits generalization. Some Test data had already been inspected during earlier experiments, so this is not a completely untouched holdout.

Dataset: [BoardGameGeek dataset on Kaggle](https://www.kaggle.com/datasets/threnjen/board-games-database-from-boardgamegeek). SVD implementation reference: [SciPy `svds`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.svds.html).
