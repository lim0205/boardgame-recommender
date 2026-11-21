# Board-Game Recommendation System

**Machine Learning Term Project · Team 4**

A reproducible Python-based recommender for BoardGameGeek (BGG) ratings. It predicts 1–10 ratings and ranks ten board games for each user. Four families of methods were implemented and compared: **Content-based Filtering**, **Model-based CF (Truncated SVD)**, **Memory-based CF (item/user kNN)** and **Hybrid Filtering**.

## Final Test results

All five configurations used the same processed data, evaluation rules and candidate catalog. Models and hyperparameters were chosen using Validation RMSE, not Test Recall.

| Model | RMSE ↓ | MAE ↓ | Hit Rate@10 ↑ | **Macro Recall@10 ↑** | Coverage@10 ↑ |
|---|---:|---:|---:|---:|---:|
| Content-based (mechanics + themes) | 1.2457 | 0.9399 | 3.2% | 0.97% | **31.5%** |
| Truncated SVD (10 factors) | 1.1157 | 0.8447 | **10.8%** | **4.13%** | 4.2% |
| Memory CF (item, Pearson, k=40) | **1.0989** | **0.8296** | 6.8% | 3.14% | 8.1% |
| Hybrid (0.1 Content + 0.9 SVD) | 1.1161 | 0.8449 | **10.8%** | 4.11% | 4.6% |
| Hybrid (0.1 Content + 0.9 Memory) | 1.1019 | 0.8317 | 6.8% | 3.09% | 8.7% |

**Interpretation:** Memory-based CF achieved the lowest prediction error; SVD had the highest Macro Recall@10 and tied for the highest Hit Rate@10; Content-based recommended the broadest set of games. The hybrids slightly increased coverage without improving prediction RMSE.

- **Rating evaluation:** all 36,003 observed Test ratings for RMSE/MAE.
- **Ranking evaluation:** the same 250 eligible users sampled with seed 42 from 4,460 Test users with at least one held-out rating ≥ 8.
- **Recall@10:** compute the fraction of each user's observed Test-relevant games retrieved among their Top 10, then average over 250 users (**macro**, not pooled/micro).
- **Hit Rate@10:** fraction of those 250 users for whom the Top 10 contains ≥1 observed Test-relevant game.
- **Catalog Coverage@10:** distinct recommended games across those users divided by the shared 1,000-game catalog.

[Final Test CSV](examples/real_bgg_memory_cf_hybrid/final_test_metrics.csv) · [Final Test JSON](examples/real_bgg_memory_cf_hybrid/final_test_results.json) · [Validation model/alpha comparisons](examples/real_bgg_memory_cf_hybrid/) · [SVD Validation comparisons](examples/real_bgg_svd_5000/)

## Dataset and experiment design

- Source: [Board Games Database from BoardGameGeek (Kaggle)](https://www.kaggle.com/datasets/threnjen/board-games-database-from-boardgamegeek).
- Valid original observations: 18,942,144 (from 18,942,215 rows). From the 1,000 most frequently rated games, select users with ≥20 ratings, sample up to 5,000 (seed=42), then deduplicate and recheck eligibility.
- Final sample: **4,997 users, 1,000 games, 362,787 ratings**. Per-user Train/Validation/Test: **290,572 / 36,212 / 36,003**.
- Only Training data inform user profiles, similarity, item biases and latent factors. Missing interactions are **unlabeled**, not observed 0-star ratings. Popularity is used for catalog selection only, **not** as a score component.
- SVD uses SciPy sparse `svds` (not Surprise SGD); Memory CF uses co-rated Pearson similarity with minimum support 3, shrinkage 10 and k=40; both Hybrid variants use content weight α=0.1.

## Software layout

| Location | Purpose |
|---|---|
| `boardrec/preprocessing.py`, `boardrec/bgg_fast.py`, `boardrec/bgg_io.py` | Authorized BGG input, cleaning, sampling and data splits |
| `boardrec/content.py`, `boardrec/svd.py`, `boardrec/memory_cf.py`, `boardrec/hybrid.py` | Four recommender families |
| `boardrec/evaluation.py` | RMSE, MAE, Hit Rate@10, **Macro Recall@10**, Catalog Coverage@10 |
| `boardrec/*_experiments.py`, `boardrec/final_test.py` | Validation experiments and final Test comparison |
| `scripts/` | Runnable entrypoints |
| `tests/` | Synthetic-data and metric correctness tests |
| `examples/` | Aggregate-only saved evaluation results/plots |

See [Evaluation protocol](docs/EVALUATION_POLICY.md) and [External functions / model parameters](docs/EXTERNAL_APIS.md) for precise definitions, nontrivial APIs and experiment details.

## Install, test and run

Requires **Python 3.10+**. From the repository root:

```bash
pip install -e ".[dev]"
python -m pytest -q
# Synthetic-data smoke test; does not need BGG input:
python -m boardrec.cli demo --top-games 35 --min-user-ratings 8 --max-users 25
```

To reproduce the real-data experiments, obtain any necessary dataset permissions and download the BGG archive **yourself**; no original ratings or usernames are provided in this repository. Place the archive at `data/raw/archive (1).zip` and run:

```bash
python scripts/run_bgg.py --archive "data/raw/archive (1).zip" --rights-confirmed --top-games 1000 --min-user-ratings 20 --max-users 5000 --seed 42
python scripts/run_svd.py --data-dir data/processed --out outputs/bgg_svd --max-hit-users 250
python scripts/run_memory_cf.py --data-dir data/processed --out outputs/bgg_memory_cf --max-hit-users 250
python scripts/run_hybrid.py --data-dir data/processed --out outputs/bgg_hybrid --svd-factors 10 --svd-scale 1.0 --memory-cf-results outputs/bgg_memory_cf/results.json --max-hit-users 250
python scripts/run_final_test.py --data-dir data/processed --out outputs/final_test --max-hit-users 250
```

The final script writes `outputs/final_test/test_metrics.csv` and `results.json`, both including `recall@10`. The raw and processed data directories, along with local outputs, are excluded from version control. If RAM is limited, see `python scripts/run_bgg.py --help` for `--low-memory`.

## Scope and limitations

This is an **offline coursework study**, not an online user trial. Sparse observed positives underestimate unknown preferences; 250-user ranking metrics are sample-sensitive. Selecting the most-rated 1,000 games introduces popularity-related catalog bias. Earlier exploratory analyses had viewed Test information; the reported Test results are therefore **not a pristine untouched independent holdout**. Recall was added later to the *same* fixed Test predictions without retraining or reselecting models.

**Data and code attribution:** Download availability is not a grant of machine-learning training or redistribution rights; check the applicable BGG/Kaggle terms. No raw BGG CSV, usernames or per-user histories are committed. The implementation relies on [SciPy `svds` documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.svds.html); adapted source code from external projects must be credited at its actual use site. Making a GitHub repository public does not grant rights to the third-party dataset.
