# BGG recommendation evaluation beyond Hit Rate@10

Actual user-uploaded BGG CSVs; preprocessed **4,997 users / 1,000 games / 362,787 ratings**; evaluation uses **1,000 fixed, seeded validation users** among 4,479 with at least one held-out rating >=8. Top-K = 10. The split was 80% Train / 10% Validation / 10% Test per user. **No new test comparison was performed for this study.** The earlier experiments had already inspected Test, so a new untouched Test split would be needed for confirmatory performance claims.

## Why Hit Rate alone is misleading

Offline user-game ratings are *missing not at random*: popular games are rated more often. Treating every unrated item as a negative is incorrect. An offline metric scored against held-out ratings disproportionately favors familiar popular games. Our earlier recommendation with 90% popularity was a valid popularity-oriented baseline, **not a good default personalized recommendation model**.

## New evaluation metrics

- **Observed NDCG@10**: order-aware; test ratings >=8 contribute gain `min(3, rating - 7)` and receive a log2 discount by position. Unrated candidate labels are **unknown**, so the score is calculated against sparse *observed* positives only.
- **Observed Recall@10**: among each user's held-out ratings >=8, fraction recovered in the 10 recommendations; mean over users. Missing ratings remain unlabeled.
- **Catalog Coverage@10**: number of different games ever recommended across selected users / 1,000 candidate games. Measures breadth, **not relevance**.
- **Inter-user Personalization@10**: `1 - average share of overlap` between user pairs' Top-10 lists; higher indicates lists differ. **Different lists alone do not guarantee quality**.
- **Intra-list Diversity@10**: mean `1 - cosine similarity` among pairs of games within a Top-10 list, based on mechanics and themes. It need not improve as content weight grows.
- **Novelty**: average `-log2((train_game_count+1)/(train_ratings+catalog_size))`; higher means less frequently rated in Train.
- **Tail Hit Rate@10**: among users with at least one held-out liked tail game, fraction whose Top-10 includes one. The **tail** is games outside the top 20% of the already-top-1000 catalog by Train rating counts; the split is Train-only. Evaluated on 684 of the 1,000 sampled users; **not directly comparable** to ordinary Hit Rate denominators.
- **Tail Recall@10**: fraction of held-out liked tail games recovered among those same 684 eligible users.

## Actual validation comparison

| Content : Popularity | Observed NDCG@10 | Observed Recall@10 | Coverage | Personalization | Tail Hit@10 | Tail exposure share |
|---|---:|---:|---:|---:|---:|---:|
| 0 : 100 | 0.0919 | 13.46% | 5.4% | 0.473 | 0.00% | 0.00% |
| 10 : 90 | 0.0824 | 12.38% | 9.8% | 0.745 | 0.00% | 0.00% |
| 30 : 70 | 0.0663 | 9.53% | 15.1% | 0.815 | 0.00% | 0.00% |
| 50 : 50 | 0.0551 | 7.60% | 19.6% | 0.843 | 0.29% | 0.24% |
| 70 : 30 | 0.0442 | 5.87% | 26.0% | 0.865 | 1.17% | 7.35% |
| 90 : 10 | 0.0322 | 4.19% | 37.8% | 0.897 | **4.09%** | 39.44% |
| 100 : 0 | 0.0108 | 1.71% | **50.4%** | **0.930** | 3.95% | 85.34% |

Pure popularity maximizes observed relevance, yet has essentially no tail exposure. Pure content greatly widens coverage and user-to-user variation, yet is poor at recovering observed held-out positives. Higher content weight does **not** consistently raise intra-list diversity. Content 70% / popularity 30% is one interpretable **exploratory compromise**, not a proven global optimum: observed NDCG@10=0.0442, coverage=26%, personalization=0.865, Tail Hit=1.17%. Content 90% / popularity 10% is more tail-oriented but sacrifices further observed relevance.

## Recommendation for the team report

1. Report **RMSE and MAE** separately for rating prediction, as planned. Neither measures quality of a Top-10 ranking directly.
2. Report **Observed NDCG@10 and Recall@10** plus **Catalog Coverage, Personalization, and Tail Hit@10** for rankings; state denominators, candidate set, and missing-label limitations.
3. Keep **Popularity** as a baseline, **Content-only** as a personalized reference, and optionally compare **70:30** and **90:10** exploratory rank blends. Do not call a popularity-heavy blend the final personalized hybrid.
4. Once the teammate adds **Collaborative Filtering** and SVD, compare all algorithms under **identical candidate sets and same validation users**. Choose weights by a predeclared project objective, without repeatedly checking Test.
5. For confirmatory test claims, create a *fresh untouched split* or use a new temporal evaluation if timestamps exist; the previously examined Test set cannot be called pristine anymore.

**Reproduction** (with authorized user-provided processed BGG data):

```bash
python -m pytest -q
python -c "from boardrec.multimetric import run_multimetric; print(run_multimetric('data/processed', 'outputs/bgg_multimetric_1000', max_users=1000).to_string(index=False))"
```

Aggregated example outputs: `examples/real_bgg_multimetric_1000/`. Raw usernames, ratings and train/val/test rows **must not** be published.

**Data license reminder:** authorization for ML model training/use must be checked independently of public Kaggle download availability.