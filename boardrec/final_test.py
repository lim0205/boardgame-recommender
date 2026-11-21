"""One-time TEST evaluation of the settings already selected on Validation.

Nothing here selects hyperparameters: every setting is read from the
Validation outputs (or the documented SVD choice). Run this once, after all
Validation experiments are finished, and do not tune again on its results.

Models evaluated (all fitted on TRAIN only, same 250-user Top-10 sample for Hit Rate, Recall and Coverage):
  content        Content-based (feature set chosen on Validation)
  svd            Truncated SVD (n_factors / residual_scale chosen on Validation)
  memory_cf      Memory-based CF (setting chosen by run_memory_cf.py)
  hybrid_svd     Content + SVD, alpha chosen on Validation with 0 < alpha < 1
  hybrid_memory  Content + Memory CF, alpha chosen on Validation with 0 < alpha < 1

The hybrids exclude alpha = 0 and 1 because those endpoints are the pure CF and
pure Content rows, which are already reported separately.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .content import ContentBasedRecommender
from .evaluation import evaluate
from .hybrid import CachedPredictions, HybridRecommender
from .memory_cf import MemoryCFRecommender
from .preprocessing import load_prepared
from .svd import SVDRecommender

METRICS = ("rmse", "mae", "hit_rate@10", "recall@10", "catalog_coverage@10")


def select_hybrid_alphas(hybrid_validation_csv: str | Path) -> dict[str, float]:
    """Best Validation-RMSE alpha strictly inside (0, 1) for each CF model."""
    table = pd.read_csv(hybrid_validation_csv)
    inner = table.loc[(table.alpha > 0) & (table.alpha < 1)]
    if inner.empty:
        raise ValueError("Hybrid validation table has no 0 < alpha < 1 rows")
    best = inner.sort_values(["rmse", "alpha"]).groupby("cf_model", sort=False).head(1)
    return {row.cf_model: float(row.alpha) for row in best.itertuples()}


def run_final_test(data_dir: str | Path, output_dir: str | Path, *,
                   memory_cf_results: str | Path, hybrid_validation_csv: str | Path,
                   content_feature_set: str = "both", svd_params: dict | None = None,
                   max_hit_users: int = 250) -> dict:
    train, _, test, games, mechanics, themes = load_prepared(data_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Reuse Validation-selected parameters; Test must not tune models.
    mcf = json.loads(Path(memory_cf_results).read_text(encoding="utf-8"))["best_hyperparameters"]
    mcf_params = {k: mcf[k] for k in ("kind", "similarity", "k", "min_support", "shrinkage", "item_reg")}
    svd_params = {"n_factors": 10, "residual_scale": 1.0, "item_reg": 10.0, "seed": 42,
                  **(svd_params or {})}
    alphas = select_hybrid_alphas(hybrid_validation_csv)

    content = CachedPredictions(ContentBasedRecommender(feature_set=content_feature_set)
                                .fit(train, games, mechanics, themes))
    svd = CachedPredictions(SVDRecommender(**svd_params).fit(train, games))
    memory = CachedPredictions(MemoryCFRecommender(**mcf_params).fit(train, games))
    # Alpha endpoints reconstruct the standalone models for a fair comparison.
    models = {
        "content": (HybridRecommender(content, svd, 1.0, train), {"feature_set": content_feature_set}),
        "svd": (HybridRecommender(content, svd, 0.0, train), svd_params),
        "memory_cf": (HybridRecommender(content, memory, 0.0, train), mcf_params),
    }
    if "svd" in alphas:
        models["hybrid_svd"] = (HybridRecommender(content, svd, alphas["svd"], train),
                                {"alpha": alphas["svd"]})
    if "memory_cf" in alphas:
        models["hybrid_memory_cf"] = (HybridRecommender(content, memory, alphas["memory_cf"], train),
                                      {"alpha": alphas["memory_cf"]})

    records = []
    for name, (model, params) in models.items():
        # Rating error uses all Test ratings; Top-10 uses the same 250-user sample.
        measured = evaluate(model, test, max_hit_users=max_hit_users)
        records.append({"model": name, "split": "test",
                        "settings": json.dumps(params, sort_keys=True), **measured})
    table = pd.DataFrame(records)
    table.to_csv(output / "test_metrics.csv", index=False)
    report = {
        "split": "test",
        "note": ("Settings fixed on Validation beforehand; TEST evaluated once. "
                 "Earlier exploratory work already viewed the Test split, so it is not a pristine holdout. "
                 "No popularity term. Unrated items are unlabeled."),
        "evaluated_users": int(table.evaluated_users.iloc[0]),
        "candidate_catalog_size": int(table.candidate_catalog_size.iloc[0]),
        "results": {r["model"]: {"settings": json.loads(r["settings"]),
                                 **{m: (None if pd.isna(r[m]) else float(r[m])) for m in METRICS}}
                    for r in records},
    }
    (output / "results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
