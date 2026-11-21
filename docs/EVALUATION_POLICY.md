# Final evaluation policy — pure Content-based Filtering

Original decision date: 2026-10-27. This file records the original four-metric evaluation policy; on 2026-11-22 the final Test comparison was extended with Macro Recall@10, without changing the selected models or their settings.

## Recommendation algorithm

Use `ContentBasedRecommender` with binary `mechanics` / `themes` game metadata, cosine similarity and each user's Train-set ratings. **No popularity score or popularity-blend weight is applied to the Top-10 recommendations.**

The earlier `ranking.py` and `ranking_experiments.py` modules are **historical exploratory comparisons only**; `boardrec.cli run` and `scripts/run_bgg.py` do not invoke them. When the teammate implements CF, the *planned* final Hybrid combines Content-based with Collaborative Filtering, not popularity.

Caveat: the dataset itself is limited to the 1,000 games with most rating observations as a practical preprocessing step; this produces an already-popularity-skewed candidate catalog. That is separate from recommendation scoring and must be disclosed.

## Final comparison metrics — five (originally four)

| Metric | Role | Interpretation |
|---|---|---|
| **RMSE** | Mandatory rating accuracy | Root mean squared error of held-out 1–10 star ratings; lower is better |
| **MAE** | Mandatory rating accuracy | Mean absolute error of held-out 1–10 ratings; lower is better |
| **Hit Rate@10** | Mandatory Top-10 relevance | Among eligible users with ≥1 held-out rating ≥8, fraction whose Top-10 contains ≥1 such game; higher is better |
| **Recall@10 (Macro)** | Additional Top-10 relevance | Mean across the same eligible users of relevant held-out games found in Top-10 / all their observed relevant held-out games; higher is better |
| **Catalog Coverage@10** | One supplementary Top-10 breadth metric | Distinct games in all evaluated Top-10 lists / size of the *shared* candidate catalog; higher indicates more catalog breadth, not necessarily accuracy |

RMSE and MAE should be presented together because they capture different error sensitivities. Hit Rate measures a binary retrieval success. Coverage captures an orthogonal property: whether the system recommends only a narrow set of items.

The initial Validation report used four metrics; the 2026-11-22 final Test comparison adds Recall@10 while preserving the original Hit Rate/Coverage cohort. NDCG@10, Personalization, and Tail Hit Rate remain optional future evaluations. No model settings were tuned using Test Recall.

## Actual BGG validation result (no popularity ranking)

The pure `mechanics + themes` Content-based model was evaluated on the real user-provided 4,997-user processed BGG dataset. **RMSE and MAE use all 36,212 Validation ratings; Hit Rate and Coverage use the same 1,000 seeded, eligible Validation users**.

| RMSE ↓ | MAE ↓ | Hit Rate@10 ↑ | Catalog Coverage@10 ↑ |
|---:|---:|---:|---:|
| **1.2384** | **0.9390** | **6.1% (61/1,000 users)** | **50.4% (504/1,000 games)** |

This is **Validation**, not a fresh Test result. The originally reported 3.2% Test Hit Rate used a different 250-user sample; do not mix these into one comparison row. Full aggregate JSON: [validation_metrics.json](../examples/real_bgg_official_1000/validation_metrics.json).

## Protocol and cautions

1. Compare `mechanics`, `themes`, `both` Content-based feature sets on **Validation** (not Test). For continuity with the original assignment, choose the feature set by minimum Validation RMSE; retain historical four-metric Validation outputs and add Recall to the final Test comparison.
2. Keep the per-user Train/Validation/Test split (80/10/10), seed 42, candidate catalog and `Top-10` rule identical across models.
3. Use the **same deterministic eligible-user sample** for Hit Rate, Recall and Coverage. Present the number of evaluated users and catalog size beside each result. Coverage depends on how many people were evaluated.
4. Unknown/unrated games are **unlabeled**, not disliked. Therefore Hit Rate is an observed-positive offline proxy, not actual real-world recommendation success.
5. Earlier exploratory work already looked at the Test split. Do **not** use it repeatedly to tune the model or describe it as a pristine untouched test; a genuinely confirmatory result requires a fresh holdout.
6. Only aggregated metrics may be shared publicly. Original BGG ratings, usernames and histories should not be committed. Independently confirm dataset training-use rights.

## Current code behavior

`boardrec.evaluation.evaluate` reports `rmse`, `mae`, `hit_rate@10`, `recall@10`, `catalog_coverage@10` in newly generated outputs along with sample counts. `Catalog Coverage` is computed over the **same eligible users** and recommendations already used for Hit Rate, so this extra metric adds no second recommendation pass.

Run (once authorized input data is prepared):

```bash
python -m pytest -q
python -m boardrec.cli run --data-dir data/processed --out outputs/bgg_content --max-hit-users 500
```

**Historical results:** the 2026-10-26 test run of the pure Content model found RMSE ≈1.2457, MAE ≈0.9399 and Hit Rate@10 =3.2% for 250 eligible Test users. Coverage was not included in that original Test report, so it cannot be claimed for the same run without rerunning the evaluator. A separate Validation experiment with 1,000 eligible users reported pure Content Catalog Coverage@10 of 50.4%; these two figures are **not the same evaluation cohort or split**.

## Added final Test Recall@10

The final five-model Test results use the original 250 eligible users selected with seed 42. Ratings ≥8 in Test define observed relevant games. Recall@10 is the simple average of per-user recall values (Macro Recall), not the pooled ratio of recovered relevant games to all positive records. The SVD model retrieved 30 of 834 observed relevant Test games across these users: its pooled Micro Recall would be 3.60%, but its actual Macro Recall@10 was **4.13%**. All five models' original Hit Rate and Catalog Coverage values were reproduced.

See [final_test_metrics.csv](../examples/real_bgg_memory_cf_hybrid/final_test_metrics.csv) for results. This is a post-hoc metric computation on the existing held-out sample; Test had already been inspected earlier, so it is not a pristine independent confirmation.
