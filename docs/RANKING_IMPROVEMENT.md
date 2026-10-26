# BGG Ranking Improvement — Actual Data, 26 October 2026

This document reports an **offline experiment** on the user-uploaded BoardGameGeek/Kaggle ZIP. There are no raw usernames or individual rating records in GitHub. Verify usage authorization independently; publishing a Kaggle dataset does not automatically authorize AI training.

## Dataset and split (unchanged)

- 4,997 users, 1,000 games, 362,787 valid deduplicated ratings
- Train: 290,572; Validation: 36,212; Test: 36,003
- A relevant held-out item is a game rated at least 8/10 by that user
- Evaluation: Hit Rate@10 among 1,000 randomly sampled eligible users per split, random seed 42
- **Important:** This project already inspected the test set before the ranking improvement. Therefore these test comparisons are exploratory, not a pristine unbiased final estimate. Future work should construct a fresh untouched holdout or use a new time-based split.

## Methods

1. **Pure Content:** predict item ratings using user mean + weighted rating deviations and binary mechanics/theme cosine similarity.
2. **Popularity baseline:** recommend games with the most ratings in **Train** only, excluding the user's Train-rated items. This is **not personalized**.
3. **Content + Popularity Ranking:** rank the candidate games twice (predicted content rating and log1p(Train rating count)), convert each ranking to percentile [0,1], then combine them:

   `rank(u,g) = α * content_rank_percentile(u,g) + (1 - α) * popularity_rank_percentile(g)`

   Alpha = 1 is content-only; alpha = 0 is popularity-only. Both ranking components use *only Train* statistics. The content recommender's predicted rating remains on the original 1–10 scale; **blended ranking scores are not predicted ratings**.

4. Tune 33 combinations of feature sets {mechanics, themes, both} and α ∈ {0.0, 0.1, ..., 1.0} using Validation Hit Rate@10. Prefer more content weight if scores tie. Report number of **distinct recommended games** to monitor popularity concentration.

## Validation results (n=1,000 eligible users)

| Ranking model | α for content | Hit Rate@10 | Distinct recommended games |
|---|---:|---:|---:|
| **Popularity only (winner)** | 0.0 | **33.9%** (339/1,000) | 54 |
| Both mechanics/themes + popularity | 0.1 | 32.2% (322/1,000) | 98 |
| Both + popularity | 0.5 | 22.2% (222/1,000) | 196 |
| Both content only | 1.0 | 6.1% (61/1,000) | 504 |

[Full validation grid](../examples/real_bgg_ranking_1000/validation_ranking_grid.csv) · [Validation plot](../examples/real_bgg_ranking_1000/validation_ranking.svg)

**Tradeoff rule, set using Validation only:** preserve Hit Rate within 2 percentage points of the best variant, require at least 1.5x as many distinct recommended games, and select the largest content weight satisfying both. This selected **both features, α=0.1**.

## Exploratory Test comparison (n=1,000 eligible users)

| Model | Test Hit Rate@10 | Distinct recommended games |
|---|---:|---:|
| Content only, both features | **4.7%** (47/1,000) | 506 |
| Popularity only | **30.6%** (306/1,000) | 52 |
| Both content 10% + popularity 90% | **29.9%** (299/1,000) | 94 |

The prior **3.2%** result used just 250 test users, so it must **not** be compared directly with these 1,000-user results as a causal or matched performance gain. The matched 1,000-user content-only reference is **4.7%**.

### Interpretation

- Optimizing Hit Rate strongly favors mainstream games. Popularity is powerful because **observed** positive ratings are concentrated on frequently rated games. This result does not prove users prefer popularity-only recommendations.
- α=0.1 substantially broadens catalog coverage (94 vs 52 unique Test recommendations) while giving up 0.7 percentage points of measured Test Hit Rate.
- RMSE and MAE of the content **rating predictor** do not change when ranking alone changes. Previously measured Content-both Test RMSE ≈ **1.2457**, MAE ≈ **0.9399**. The blended 0–1 ranking score is not a rating and must not be evaluated as an RMSE prediction.
- Offline Hit Rate depends on a sparse set of observed held-out positives. An unrated item is **unknown**, not disliked. The metric is not live click-through rate or user satisfaction.
- This is a popularity-aware content ranker, **not** the coursework Hybrid CF + Content model: the teammate will still implement Memory-based CF, SVD and CF + Content hybrid.

## Reproduce

Obtain the original data with appropriate permission. Using the previously generated prepared files:

```bash
pip install -e ".[dev]"
python -m pytest -q
python scripts/run_ranking.py --prepared-dir data/processed --out outputs/bgg_ranking --max-users 1000 --rights-confirmed
```

The prepared files under `data/processed/` are not committed to GitHub. To regenerate them from an authorized archive, see [BGG data instructions](BGG_REAL_DATA.md) and keep the same filters `--top-games 1000 --min-user-ratings 20 --max-users 5000 --seed 42`.

## For the teammate

`boardrec/ranking.py` contains reusable `PopularityRanker` and `BlendedRanker`; both implement `recommend(user_id,n=10)`, and `BlendedRanker` delegates `predict(user_id, game_id)` to Content-based. For **true CF + Content Hybrid**, the teammate should separately implement the combination of 1–10 rating predictions, tune α on Validation, and report both rating metrics and ranking metrics. Do not confuse the ranking α here with that hybrid α.
