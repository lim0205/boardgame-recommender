"""Validation-only Hybrid (Content + CF) comparison. Test is never read.

For every CF candidate (SVD and Memory-based CF) the alpha grid is evaluated
with the shared evaluator on the same Validation users. alpha=1.0 equals pure
Content-based and alpha=0.0 equals the pure CF model, so the standalone
models appear in the same table. Selection: lowest Validation RMSE.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .content import ContentBasedRecommender
from .evaluation import evaluate
from .hybrid import CachedPredictions, HybridRecommender
from .memory_cf import MemoryCFRecommender
from .preprocessing import load_prepared
from .svd import SVDRecommender

METRICS = ("rmse", "mae", "hit_rate@10", "recall@10", "catalog_coverage@10")
DEFAULT_ALPHAS = tuple(round(a, 1) for a in np.linspace(0.0, 1.0, 11))


def run_hybrid_experiments(data_dir: str | Path, output_dir: str | Path, *,
                           alphas=DEFAULT_ALPHAS, content_feature_set: str = "both",
                           svd_params: dict | None = None,
                           memory_cf_params: dict | None = None,
                           max_hit_users: int = 250) -> dict:
    train, val, _, games, mechanics, themes = load_prepared(data_dir)
    alphas = [float(a) for a in alphas]
    if not alphas or any(not 0.0 <= a <= 1.0 for a in alphas):
        raise ValueError("alphas must be nonempty and within [0, 1]")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    svd_params = {"n_factors": 10, "residual_scale": 1.0, "item_reg": 10.0, "seed": 42,
                  **(svd_params or {})}
    memory_cf_params = {"kind": "item", "similarity": "pearson", "k": 40,
                        **(memory_cf_params or {})}

    content = CachedPredictions(ContentBasedRecommender(feature_set=content_feature_set)
                                .fit(train, games, mechanics, themes))
    # Fit CF components once; reuse their predictions across alpha values.
    cf_models = {
        "svd": CachedPredictions(SVDRecommender(**svd_params).fit(train, games)),
        "memory_cf": CachedPredictions(MemoryCFRecommender(**memory_cf_params).fit(train, games)),
    }

    records = []
    for cf_name, cf in cf_models.items():
        # alpha=0/1 are pure models; interior values produce weighted hybrids.
        for alpha in alphas:
            model = HybridRecommender(content, cf, alpha=alpha, train=train)
            measured = evaluate(model, val, max_hit_users=max_hit_users)
            records.append({"cf_model": cf_name, "alpha": alpha, "split": "val", **measured})
    table = pd.DataFrame(records)
    table.sort_values(["cf_model", "alpha"]).to_csv(output / "validation_metrics.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for cf_name, subset in table.groupby("cf_model"):
        subset = subset.sort_values("alpha")
        for ax, metric in zip(axes, ("rmse", "hit_rate@10", "catalog_coverage@10")):
            ax.plot(subset.alpha, subset[metric], marker="o", label=cf_name)
    for ax, label in zip(axes, ("RMSE (lower is better)", "Hit Rate@10", "Catalog Coverage@10")):
        ax.set_xlabel("alpha (content weight; 0 = pure CF, 1 = pure Content)")
        ax.set_ylabel(f"Validation {label}")
        ax.legend()
    fig.suptitle("Hybrid: alpha * Content + (1 - alpha) * CF")
    fig.tight_layout()
    fig.savefig(output / "validation_alpha.svg")
    plt.close(fig)

    # Validation chooses settings; final_test uses the best INTERIOR alpha for each CF.
    ranked = table.sort_values(["rmse", "cf_model", "alpha"]).reset_index(drop=True)
    best = ranked.iloc[0]

    def _metrics(row):
        return {m: (None if pd.isna(row[m]) else float(row[m])) for m in METRICS}

    per_cf = {name: {"alpha": float(sub.iloc[0].alpha), **_metrics(sub.iloc[0])}
              for name, sub in ranked.groupby("cf_model", sort=False)}
    report = {
        "model": "alpha * Content-based + (1 - alpha) * CF (same 1-10 scale)",
        "selected_by": "Lowest Validation RMSE over (cf_model, alpha); no Test used",
        "content_feature_set": content_feature_set,
        "svd_params": svd_params,
        "memory_cf_params": memory_cf_params,
        "best": {"cf_model": best.cf_model, "alpha": float(best.alpha), **_metrics(best)},
        "best_alpha_per_cf_model": per_cf,
        "evaluated_users": int(best.evaluated_users),
        "candidate_catalog_size": int(best.candidate_catalog_size),
        "note": "alpha=1 is pure Content, alpha=0 is pure CF. No popularity term. Unrated items are unlabeled.",
    }
    (output / "results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
