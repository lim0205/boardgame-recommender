# Final evaluation policy — pure Content-based Filtering

Decision date: 2026-10-27. This file states the **official project evaluation policy**, not a new experimental score.

## Recommendation algorithm

Use `ContentBasedRecommender` with binary `mechanics` / `themes` game metadata, cosine similarity and each user's Train-set ratings. **No popularity score or popularity-blend weight is applied to the Top-10 recommendations.**

The earlier `ranking.py` and `ranking_experiments.py` modules are **historical exploratory comparisons only**; `boardrec.cli run` and `scripts/run_bgg.py` do not invoke them. When the teammate implements CF, the *planned* final Hybrid combines Content-based with Collaborative Filtering, not popularity.

Caveat: the dataset itself is limited to the 1,000 games with most rating observations as a practical preprocessing step; this produces an already-popularity-skewed candidate catalog. That is separate from recommendation scoring and must be disclosed.

## Official metrics — just four

| Metric | Role | Interpretation |
|---|---|---|
| **RMSE** | Mandatory rating accuracy | Root mean squared error of held-out 1–10 star ratings; lower is better |
| **MAE** | Mandatory rating accuracy | Mean absolute error of held-out 1–10 ratings; lower is better |
| **Hit Rate@10** | Mandatory Top-10 relevance | Among eligible users with ≥1 held-out rating ≥8, fraction whose Top-10 contains ≥1 such game; higher is better |
| **Catalog Coverage@10** | One supplementary Top-10 breadth metric | Distinct games in all evaluated Top-10 lists / size of the *shared* candidate catalog; higher indicates more catalog breadth, not necessarily accuracy |

RMSE and MAE should be presented together because they capture different error sensitivities. Hit Rate measures a binary retrieval success. Coverage captures an orthogonal property: whether the system recommends only a narrow set of items.

Other measures (NDCG@10, Recall@10, Personalization, Tail Hit Rate) are **optional prior exploratory analyses**, not required for the final comparison table. There is no need to tune a model on all metrics.

## Protocol and cautions

1. Compare `mechanics`, `themes`, `both` Content-based feature sets on **Validation** (not Test). For continuity with the original assignment, choose the feature set by minimum Validation RMSE; report the four metrics for that feature set.
2. Keep the per-user Train/Validation/Test split (80/10/10), seed 42, candidate catalog and `Top-10` rule identical across models.
3. Use the **same deterministic eligible-user sample** for Hit Rate and Coverage. Present the number of evaluated users and catalog size beside each result. Coverage depends on how many people were evaluated.
4. Unknown/unrated games are **unlabeled**, not disliked. Therefore Hit Rate is an observed-positive offline proxy, not actual real-world recommendation success.
5. Earlier exploratory work already looked at the Test split. Do **not** use it repeatedly to tune the model or describe it as a pristine untouched test; a genuinely confirmatory result requires a fresh holdout.
6. Only aggregated metrics may be shared publicly. Original BGG ratings, usernames and histories should not be committed. Independently confirm dataset training-use rights.

## Current code behavior

`boardrec.evaluation.evaluate` reports `rmse`, `mae`, `hit_rate@10`, `catalog_coverage@10` in its output along with sample counts. `Catalog Coverage` is computed over the **same eligible users** and recommendations already used for Hit Rate, so this extra metric adds no second recommendation pass.

Run (once authorized input data is prepared):

```bash
python -m pytest -q
python -m boardrec.cli run --data-dir data/processed --out outputs/bgg_content --max-hit-users 500
```

**Historical results:** the 2026-10-26 test run of the pure Content model found RMSE ≈1.2457, MAE ≈0.9399 and Hit Rate@10 =3.2% for 250 eligible Test users. Coverage was not included in that original Test report, so it cannot be claimed for the same run without rerunning the evaluator. A separate Validation experiment with 1,000 eligible users reported pure Content Catalog Coverage@10 of 50.4%; these two figures are **not the same evaluation cohort or split**.
