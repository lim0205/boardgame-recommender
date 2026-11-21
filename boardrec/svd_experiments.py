"""Validation-only comparison for classical truncated-SVD recommender."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .evaluation import evaluate
from .preprocessing import load_prepared
from .svd import SVDRecommender


def run_svd_experiments(data_dir: str | Path, output_dir: str | Path, *,
                        factors=(10, 20, 40), scales=(0.5, 1.0),
                        item_reg=10.0, seed=42, max_hit_users=250) -> dict:
    """Train models on TRAIN, select by lowest Validation RMSE. TEST unused."""
    train, val, _, games, _, _ = load_prepared(data_dir)
    if not factors or not scales:
        raise ValueError('factors and scales must be nonempty')
    if len(set(factors)) != len(factors) or len(set(scales)) != len(scales):
        raise ValueError('Hyperparameter grids must contain unique values')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    # Changing k changes the truncated decomposition, so retrain for each factor count.
    for k in factors:
        model = SVDRecommender(n_factors=int(k), item_reg=float(item_reg),
                               residual_scale=1.0, seed=seed).fit(train, games)
        for scale in scales:
            # The same learned factors can be evaluated at multiple residual scales.
            model.residual_scale = float(scale)
            measured = evaluate(model, val, max_hit_users=max_hit_users)
            records.append({"n_factors": int(k), "residual_scale": float(scale),
                            "item_reg": float(item_reg), "split": "val", **measured})
    table = pd.DataFrame(records).sort_values(['rmse','n_factors','residual_scale'])
    # Select hyperparameters by Validation RMSE; leave Test untouched.
    best = table.iloc[0]
    table.to_csv(output / 'validation_metrics.csv', index=False)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for scale, subset in table.groupby('residual_scale'):
        subset = subset.sort_values('n_factors')
        ax.plot(subset.n_factors, subset.rmse, marker='o', label=f'residual_scale={scale:g}')
    ax.set_xlabel('Number of latent factors (k)')
    ax.set_ylabel('Validation RMSE (lower is better)')
    ax.set_title('BGG: truncated SVD validation')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'validation_rmse.svg')
    plt.close(fig)
    best_run = {"n_factors": int(best.n_factors),
                "residual_scale": float(best.residual_scale),
                "item_reg": float(best.item_reg)}
    metrics = {key: (None if pd.isna(best[key]) else float(best[key]))
               for key in ('rmse','mae','hit_rate@10','recall@10','catalog_coverage@10')}
    report = {"model": "Sparse truncated SVD with user mean + item bias",
              "selected_by": "Lowest Validation RMSE; no Test used",
              "best_hyperparameters": best_run,
              "validation_metrics": metrics,
              "evaluated_users": int(best.evaluated_users),
              "candidate_catalog_size": int(best.candidate_catalog_size),
              "note": "Train-only fit; Validation-only tuning. Unrated items are unlabeled."
              " Previous work already inspected the Test split, therefore the old Test is not pristine."}
    (output / 'results.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    return report
