"""Select Content-based feature variant on VAL; evaluate winner on TEST once."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .content import ContentBasedRecommender, FEATURE_SETS
from .evaluation import evaluate
from .preprocessing import load_prepared


def _jsonable(record: dict) -> dict:
    return {k: (None if isinstance(v, (float, np.floating)) and not np.isfinite(v)
                else v) for k, v in record.items()}


def run_experiments(data_dir: str | Path, output_dir: str | Path,
                    max_hit_users: int = 500) -> dict:
    train, val, test, games, mechanics, themes = load_prepared(data_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for variant in FEATURE_SETS:
        model = ContentBasedRecommender(feature_set=variant).fit(
            train, games, mechanics, themes)
        scores = evaluate(model, val, max_hit_users=max_hit_users)
        records.append({"feature_set": variant, "split": "val", **scores})
    validation = pd.DataFrame(records)
    if validation.rmse.notna().sum() == 0:
        raise ValueError("Validation RMSE is undefined; check data")
    best_variant = validation.sort_values(["rmse", "feature_set"]).iloc[0].feature_set
    # Test must never be used for variant selection.
    best_model = ContentBasedRecommender(feature_set=best_variant).fit(
        train, games, mechanics, themes)
    test_metrics = evaluate(best_model, test, max_hit_users=max_hit_users)
    results = pd.concat([validation, pd.DataFrame([{"feature_set": best_variant,
                                                   "split": "test", **test_metrics}])],
                        ignore_index=True)
    results.to_csv(output / "metrics.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(validation.feature_set, validation.rmse)
    ax.set_ylabel("Validation RMSE (lower is better)")
    ax.set_xlabel("Content features")
    ax.set_title("Content-based feature comparison")
    fig.tight_layout()
    fig.savefig(output / "validation_rmse.png", dpi=160)
    plt.close(fig)
    example_user = str(train.user_id.iloc[0])
    example = best_model.recommend(example_user, n=10)
    if "Name" in best_model.games_.columns:
        example["name"] = best_model.games_.reindex(example.game_id)["Name"].to_numpy()
    example.to_csv(output / "example_top10.csv", index=False)
    report = {"selected_by": "lowest validation RMSE", "best_feature_set": best_variant,
              "validation": [_jsonable(r) for r in records],
              "test": _jsonable(test_metrics), "example_user": "redacted",
              "warning": "Offline held-out positives are sparsely labeled; do not interpret Hit Rate as online satisfaction."}
    (output / "results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
