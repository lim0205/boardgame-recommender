# Actual BGG dataset experiment — 2026-10-26

This experiment uses the real **user-provided** `archive (1).zip`, not synthetic data. It is an offline coursework comparison, not a measured production recommendation success rate.

## Source and privacy

The archive contains `games.csv`, `user_ratings.csv`, `mechanics.csv`, `themes.csv`, and supporting BGG files.

- Original rating table: **18,942,215** records
- Valid ratings used for selection: **18,942,144**
- Original distinct rated game IDs: **21,925**
- Required rating headers: `BGGId, Rating, Username`
- No user ratings, usernames, user histories, or original CSVs are included in this repository.
- **Data rights:** Being able to download a Kaggle ZIP does not grant model-training permission. Verify the BGG terms and obtain any necessary permission before reuse or redistribution. The local CLI requires `--rights-confirmed` but that flag does not grant permission.

## Preprocessing specification

Selection and splits use `seed=42`.

1. Select the **1,000** most-rated games, intersecting with game metadata.
2. Eligible users must have at least **20** ratings among those games.
3. Sample **5,000** users at random from **140,755** eligible users.
4. Deduplicate `(user_id, game_id)` using mean rating, leaving **4,997 users**, **1,000 games**, and **362,787 ratings**.
5. Split each user's observations into Train **290,572**, Validation **36,212**, Test **36,003**.

## Models and evaluation

Compare feature sets `mechanics`, `themes`, and `both` using binary game attributes and cosine similarity. User history comes **only from training**. Predict 1–10 using similarity-weighted deviation from training user mean. Validation RMSE chooses the feature variant. The final test split is used only to evaluate the winner.

| Features | Split | RMSE | MAE | sampled Hit Rate@10 |
|---|---|---:|---:|---:|
| Mechanics | Validation | 1.242693 | 0.942983 | 5.6% |
| Themes | Validation | 1.304937 | 0.983749 | 6.4% |
| Mechanics + Themes | Validation | **1.238384** | **0.939017** | 4.0% |
| **Mechanics + Themes** | Test | **1.245727** | **0.939926** | **3.2%** |

Hit Rate@10 was evaluated on a fixed seeded sample of **250 eligible users** (rating ≥8 is relevant). For test, **8/250** eligible sampled users had at least one held-out relevant game in their Top-10 list. Unrated items are unknown, not negatives. The sparse held-out sample means this number is not a real user satisfaction rate or online conversion rate.

Results are published as aggregate-only CSV/JSON/SVG under `examples/real_bgg_5000/`.

## Reproducing on a machine with rights to the ZIP

```bash
pip install -e ".[dev]"
python -m pytest -q
python scripts/run_bgg.py --archive "data/raw/archive (1).zip" --rights-confirmed --top-games 1000 --min-user-ratings 20 --max-users 5000 --seed 42 --max-hit-users 250
```

The fast one-pass typed CSV read needs sufficient RAM (~4GB recommended); use `--low-memory` for the slower chunked preparation. Result files `data/processed/*` and `outputs/*` are gitignored because they may include user information.

**Status:** 5,000-user sample finished and its metrics were reproduced. The attempt to scale this run to 20,000 users was interrupted while saving intermediate CSVs; **no** 20,000-user metrics have been claimed.

## Technical caveat

The item popularity cutoff is selected from the full original rating file before per-user split, which can reflect held-out item activity. It is a fixed-catalog offline comparison rather than a strict inductive generalization benchmark.
