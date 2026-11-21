# External APIs, model parameters, and reproducibility

This document accompanies the commented Python source. It explains nontrivial libraries, classes, methods, and configuration parameters used in the board-game recommendation experiment. The GitHub repository is a Python command-line pipeline; no web front-end is claimed.

## Repository modules

- `boardrec/bgg_io.py`: unpack/check authorized BGG CSV files and their required columns.
- `boardrec/bgg_fast.py`, `preprocessing.py`: read/clean ratings, select the top-1,000-item catalog, sample users with a fixed seed, average duplicate labels, and split each user into Train/Validation/Test.
- `boardrec/content.py`: binary mechanics/themes features, L2 normalization, on-demand cosine similarity, rating prediction.
- `boardrec/svd.py`: sparse residual matrix and Truncated SVD rating prediction.
- `boardrec/memory_cf.py`: User/Item neighborhood CF, masked co-rating similarity, support threshold, shrinkage and top-k weighting.
- `boardrec/hybrid.py`: weighted predictions from already-fitted Content and CF models.
- `boardrec/evaluation.py`: rating metrics (all observed holdout labels) and Top-10 metrics (same eligible user sample across models).
- `boardrec/*_experiments.py`: Validation hyperparameter comparisons and aggregate plots; `boardrec/final_test.py` performs final evaluation on selected configurations.
- `scripts/`: reproducible entrypoints; `tests/`: automated regression tests; `examples/`: aggregate output CSV, JSON and figures.

## Custom classes and key parameters

| Class / method | Parameters and purpose |
| --- | --- |
| `PrepConfig` | `top_games` = catalog size; `min_user_ratings` = activity threshold *within selected catalog*; `max_users` = sample cap; `seed` = reproducible RNG; `chunksize` = rows per low-memory CSV batch; `train_ratio`, `val_ratio` = per-user holdout fractions. |
| `ContentBasedRecommender(feature_set)` | `feature_set` selects `mechanics`, `themes`, or `both`; the fitted model uses only Training ratings and game features. |
| `SVDRecommender(n_factors, item_reg, residual_scale, seed)` | `n_factors` = truncated rank; `item_reg` = regularization of per-item rating offsets; `residual_scale` = strength of reconstructed residual; `seed` = sparse-SVD RNG seed. |
| `MemoryCFRecommender(kind, similarity, k, min_support, shrinkage, item_reg)` | `kind` = `user` or `item`; `similarity` = `cosine` or `pearson`; `k` = neighbors; `min_support` = minimum overlap; `shrinkage` = discount of poorly supported similarities; `item_reg` = regularization of item bias. |
| `HybridRecommender(content_model, cf_model, alpha, train)` | The models are already fitted; `alpha` is the Content weight, `1-alpha` the CF weight; `train` identifies games previously rated and excluded from recommendation. |
| `fit(train,...)` | Fits to Training records only; for Content also accepts `games`, `mechanics`, `themes`. |
| `predict(user_id, game_id)` / `predict_many(user_id, game_ids)` | Predict one or multiple 1–10 ratings. |
| `recommend(user_id, n=10)` | Sort candidate items by predicted score and return Top-`n` without Training-rated games. |
| `rating_metrics(model, holdout)` | Returns RMSE and MAE for **all observed** holdout rating rows. |
| `hit_rate_at_k(model, holdout, k=10, relevant_threshold=8, max_users=250)` | Computes Hit Rate@k and catalog coverage over a deterministic sample of users with at least one held-out relevant game. |

## External libraries and nontrivial API calls

| Library / callable | Explanation of used parameters |
| --- | --- |
| `pandas.read_csv(path, usecols=..., dtype=..., chunksize=...)` | Reads selected rating columns, sets compact dtypes, or streams rows by chunks. |
| `pandas.DataFrame.groupby(keys)`, `.mean()` | Groups ratings by user/game; merges duplicate user-game labels by mean. |
| `pandas.to_numeric(values, errors='coerce')` | Converts rating strings to numeric values, turning invalid text into missing values. |
| `numpy.random.default_rng(seed)`, `.choice(..., replace=False)`, `.permutation(...)` | Repeatable sampling without replacement and per-user random data partitioning. |
| `sklearn.preprocessing.normalize(X, norm='l2', axis=1)` | L2-normalizes the game feature vector of each row; their dot products provide cosine similarity. |
| `scipy.sparse.csr_matrix((values, (row_idx, col_idx)), shape=...)` | Stores observed matrix coordinates sparsely to reduce RAM. Unobserved residuals are not observed zero ratings. |
| `scipy.sparse.linalg.svds(A, k, which='LM', random_state=seed)` | Computes top-k singular components of the sparse residual matrix; `LM` denotes largest-magnitude singular values. Not equivalent to Surprise SGD/Funk SVD. |
| `numpy.argpartition(a, kth, axis=...)` | Efficient partial selection of top neighbors without full sorting. |
| `numpy.divide(numerator, denominator, out=..., where=...)` | Safe division only where denominator is non-zero; otherwise keeps fallback values. |
| `numpy.clip(values, 1, 10)` | Enforces the available rating scale. |
| `matplotlib.pyplot.subplots(figsize=...)`, `Figure.savefig(path)` | Builds and exports aggregate validation plots. |
| `pytest` | Runs local unit tests, including preprocessing and evaluation correctness tests. |

These are the main externally provided calls that determine the models' behavior. Standard-library functions and other trivial wrappers are documented in the source as appropriate.

## Reproducible installation and outputs

Requires Python >= 3.10 and permitted access to the original input dataset.

```bash
pip install -e ".[dev]"
python -m pytest -q
python scripts/run_bgg.py --archive "data/raw/archive (1).zip" --rights-confirmed --top-games 1000 --min-user-ratings 20 --max-users 5000 --seed 42
python scripts/run_svd.py --data-dir data/processed --out outputs/bgg_svd --max-hit-users 250
python scripts/run_memory_cf.py --data-dir data/processed --out outputs/bgg_memory_cf --max-hit-users 250
```

For the Hybrid and final Test sequence, see `docs/MEMORY_CF_HYBRID.md`. The final aggregate Test output is included in `examples/real_bgg_memory_cf_hybrid/final_test_metrics.csv`. It records five configurations with Test RMSE, MAE, Hit Rate@10 and Coverage@10. Validation figures and per-model comparison CSVs are also retained in `examples/`.

**Citations / provenance:** The dataset source is Kaggle's Board Games Database from BoardGameGeek. The core sparse-SVD method uses SciPy `svds` ([official documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.svds.html)). If any code was copied or adapted from a blog, a repository, or Kaggle, cite the exact source in the relevant source file. The presence or absence of copying cannot be established from a code review alone.

**Data restrictions:** Do not commit user histories, usernames or original BGG files to GitHub. Download availability does not grant ML-training or redistribution rights; confirm permitted use. The repository is private, so provide the grader explicit access or a separate code archive.
