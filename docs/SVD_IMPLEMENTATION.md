# Model-based Collaborative Filtering — Truncated SVD

This file documents the **real BGG Validation** experiment for the team's second independently implemented recommender. It is NOT a final test-set comparison.

## Method

- Classical **Truncated SVD** on a sparse residual matrix of **observed ratings only**. Missing ratings are represented as zero **residual**, not zero-star ratings.
- For each observed interaction, subtract the train-only user mean and a train-only shrunken item rating bias; learn a rank-k reconstruction via SciPy `svds`.
- Prediction: `user_mean + item_bias + residual_scale * (user_factors @ game_factors)` clipped to 1–10.
- Cold start: unseen user falls back to training global mean plus item bias; unseen (known-catalog) game has no latent contribution. This is a truncated-SVD approach, **not Surprise's SGD-based Funk SVD**.
- **No explicit popularity component or popularity weight** is added. Item bias reflects observed rating deviations rather than counts alone. The candidate catalog still uses the **top 1,000 most-rated games**, per PPT preprocessing.

## Preprocessing (same as PPT)

- User-uploaded BGG CSV dataset; top 1,000 rated games; user must have >=20 ratings among selected games.
- 5,000 users sampled using seed 42; after duplicate resolution, **4,997 users / 1,000 games / 362,787 ratings**.
- Same per-user 80/10/10: Train **290,572**, Validation **36,212**, Test **36,003**.
- Hyperparameters tuned exclusively with Validation. Test split **not read for SVD metrics**. Earlier exploratory work already inspected the old Test, so it should not be called a fresh unseen test in the final report.

## Hyperparameter comparison

Train item-bias regularization `item_reg=10`, `seed=42`; compare `n_factors in {10,20,40}` and `residual_scale in {0.5,1.0}`. **Select minimum Validation RMSE**, report MAE, Hit Rate@10 and Catalog Coverage@10 alongside it.

The selected configuration: **10 factors, residual_scale=1.0**.

| Metric (Validation) | Content-based (Mechanics+Themes) | SVD |
|---|---:|---:|
| RMSE ↓ (all 36,212 ratings) | 1.2384 | **1.1109** |
| MAE ↓ (all 36,212 ratings) | 0.9390 | **0.8418** |
| Hit Rate@10 ↑ (same 250 eligible users) | 4.0% (10/250) | **18.8% (47/250)** |
| Catalog Coverage@10 ↑ (same 250 users / 1,000-game catalog) | **34.3%** (343 games) | 4.8% (48 games) |

SVD predicts observed ratings more accurately and finds more held-out liked games, but **recommendations concentrate on far fewer distinct games**. Stronger Hit Rate does not prove higher satisfaction: unrated games are unlabeled, not negatives.

### Commands

```bash
pip install -e ".[dev]"
python -m pytest -q
# If the authorized BGG data has been prepared at data/processed:
python scripts/run_svd.py --data-dir data/processed --out outputs/bgg_svd --max-hit-users 250
```

For real data prep, see `scripts/run_bgg.py`. The SVD script does **not** repeat preprocessing and requires existing `train.csv`, `val.csv`, `test.csv`, `games.csv`, `mechanics.csv`, `themes.csv`.

## Deliverables

- `boardrec/svd.py`: SVD model with `fit`, `predict`, `predict_many`, `recommend`.
- `boardrec/svd_experiments.py`, `scripts/run_svd.py`: Validation tuning and outputs.
- `tests/test_svd.py`, `tests/test_svd_experiments.py`: model, API, and no-test-leakage tests.
- `examples/real_bgg_svd_5000/`: aggregate-only Validation CSV/JSON; no usernames, per-user ratings or raw dataset.

**Data rights:** Kaggle availability does not automatically grant permission to train on BGG data. Confirm original data rights before reusing the actual-data workflow.