# Evaluation protocol and final Test results

**Fixed dataset:** 4,997 users, 1,000 games, 362,787 observed ratings; user-wise Train 290,572 / Validation 36,212 / Test 36,003. Dataset selection uses the 1,000 most-rated games before splitting; that choice may introduce selection bias.

**Model selection:** all candidate feature settings, neighborhood sizes, SVD ranks and Hybrid α values were selected using Validation RMSE. No model was reselected using Test Recall.

## Five metrics

| Metric | Formula / definition | Cohort |
|---|---|---|
| RMSE | sqrt(mean((predicted − actual)²)) | All 36,003 observed Test ratings |
| MAE | mean(abs(predicted − actual)) | All 36,003 observed Test ratings |
| Hit Rate@10 | mean(1 if Top-10 intersects user Test positives else 0) | 250 eligible Test users |
| **Macro Recall@10** | mean_user(\|Top-10 ∩ observed relevant Test games\| / \|observed relevant Test games\|) | The same 250 users |
| Catalog Coverage@10 | distinct games across all Top-10 lists / 1,000 candidate games | The same 250 users |

**Eligible user:** at least one observed Test rating ≥8. The 250 users are sampled without replacement from the 4,460 eligible users with `numpy.random.default_rng(42).choice(...)`; all models use the identical set. Candidate games exclude the user's Training-rated games. No unobserved item is treated as a known negative.

**Macro versus Micro Recall:** each eligible user has equal weight in Macro Recall. For the SVD model, 30 observed relevant Test games were retrieved out of 834 across the sampled users (Micro Recall = 30/834 = 3.60%), while the actual Macro Recall@10 is **4.13%**.

## Final Test (five model configurations)

| Model | RMSE ↓ | MAE ↓ | Hit Rate@10 ↑ | Macro Recall@10 ↑ | Coverage@10 ↑ |
|---|---:|---:|---:|---:|---:|
| Content | 1.2457 | 0.9399 | 3.2% | 0.97% | 31.5% |
| SVD | 1.1157 | 0.8447 | 10.8% | 4.13% | 4.2% |
| Item Memory CF | 1.0989 | 0.8296 | 6.8% | 3.14% | 8.1% |
| Hybrid + SVD | 1.1161 | 0.8449 | 10.8% | 4.11% | 4.6% |
| Hybrid + Memory CF | 1.1019 | 0.8317 | 6.8% | 3.09% | 8.7% |

Exact decimals, settings and aggregate counts: [final_test_metrics.csv](../examples/real_bgg_memory_cf_hybrid/final_test_metrics.csv). The same recommendations reproduced the previously recorded Hit Rate, coverage and distinct recommended-item counts after the Recall addition.

## Reproducibility and limitations

The `boardrec.evaluation.evaluate` function returns all five metrics, including `recall@10`. The final Test script writes these to CSV and JSON. Validation experiments can report Recall with the same metric implementation; historical Validation CSV files were produced before Recall was added and should not be mistaken for new five-metric runs.

Ratings below 8 are not counted as relevant Test positives, but unrated games are **unknown**, not disliked. Top-10 metrics on 250 users have sample variability. The original candidate catalog was selected using full-data popularity counts before splitting. Earlier exploratory work inspected some Test information, so these are not pristine confirmatory measurements; an independent holdout or temporal split would be preferable.
