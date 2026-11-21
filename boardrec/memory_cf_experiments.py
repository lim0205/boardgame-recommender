"""Validation-only comparison of Memory-based CF settings. Test is never read."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .evaluation import evaluate
from .memory_cf import MemoryCFRecommender
from .preprocessing import load_prepared

METRICS = ("rmse", "mae", "hit_rate@10", "recall@10", "catalog_coverage@10")


def run_memory_cf_experiments(data_dir: str | Path, output_dir: str | Path, *,
                              kinds=("user", "item"), similarities=("cosine", "pearson"),
                              ks=(10, 20, 40, 80), min_support: int = 3,
                              shrinkage: float = 10.0, item_reg: float = 10.0,
                              max_hit_users: int = 250) -> dict:
    """Fit on TRAIN, evaluate each setting on VAL, select by lowest VAL RMSE."""
    train, val, _, games, _, _ = load_prepared(data_dir)
    if not kinds or not similarities or not ks:
        raise ValueError("kinds, similarities and ks must be nonempty")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    # Tune user/item CF, cosine/Pearson similarity and neighbor count on VAL only.
    for kind in kinds:
        for sim in similarities:
            # Similarities do not depend on k: fit once, then vary k.
            model = MemoryCFRecommender(kind=kind, similarity=sim, k=int(ks[0]),
                                        min_support=min_support, shrinkage=shrinkage,
                                        item_reg=item_reg).fit(train, games)
            for k in ks:
                # Reuse each similarity matrix; changing k affects neighbor selection only.
                model.k = int(k)
                measured = evaluate(model, val, max_hit_users=max_hit_users)
                records.append({"kind": kind, "similarity": sim, "k": int(k),
                                "min_support": min_support, "shrinkage": shrinkage,
                                "split": "val", **measured})
            del model
    table = pd.DataFrame(records).sort_values(["rmse", "kind", "similarity", "k"]).reset_index(drop=True)
    table.to_csv(output / "validation_metrics.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for (kind, sim), subset in table.groupby(["kind", "similarity"]):
        subset = subset.sort_values("k")
        axes[0].plot(subset.k, subset.rmse, marker="o", label=f"{kind}/{sim}")
        axes[1].plot(subset.k, subset["hit_rate@10"], marker="o", label=f"{kind}/{sim}")
    axes[0].set_ylabel("Validation RMSE (lower is better)")
    axes[1].set_ylabel("Validation Hit Rate@10 (higher is better)")
    for ax in axes:
        ax.set_xlabel("Neighbours (k)")
        ax.set_xscale("log", base=2)
        ax.legend()
    fig.suptitle("Memory-based CF validation")
    fig.tight_layout()
    fig.savefig(output / "validation_metrics.svg")
    plt.close(fig)

    # Pick the lowest Validation RMSE; Test is not used for model selection.
    best = table.iloc[0]
    report = {
        "model": "kNN CF on train-only baseline residuals (user mean + shrunken item bias)",
        "selected_by": "Lowest Validation RMSE; no Test used",
        "best_hyperparameters": {"kind": best.kind, "similarity": best.similarity, "k": int(best.k),
                                 "min_support": int(min_support), "shrinkage": float(shrinkage),
                                 "item_reg": float(item_reg)},
        "validation_metrics": {m: (None if pd.isna(best[m]) else float(best[m])) for m in METRICS},
        "evaluated_users": int(best.evaluated_users),
        "candidate_catalog_size": int(best.candidate_catalog_size),
        "note": "Train-only fit; Validation-only selection. Unrated items are unlabeled. No popularity term.",
    }
    (output / "results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
