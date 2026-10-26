"""Validation-only selection of a ranking model with reproducible user sampling.

Only one selected variant is evaluated on TEST. A popularity and pure-content
baseline on VAL help interpret results, not cherry-pick TEST.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .content import ContentBasedRecommender, FEATURE_SETS
from .preprocessing import load_prepared
from .ranking import PopularityRanker, BlendedRanker, _percentile_ranks


def eligible_sample(holdout: pd.DataFrame, threshold=8.0,
                    max_users: int | None = 1000, seed: int = 42):
    positive = holdout.loc[holdout.rating >= threshold]
    users = sorted(positive.user_id.astype(str).unique().tolist())
    if max_users is not None and len(users) > max_users:
        users = sorted(np.random.default_rng(seed).choice(users, max_users, replace=False))
    relevant = {str(uid): set(g.game_id.astype(str)) for uid, g in positive.groupby("user_id")}
    return users, relevant, positive.user_id.nunique()


def validate_ranking(train, val, games, mechanics, themes, *,
                     alpha_grid=None, k=10, max_users=1000, seed=42):
    """Tune feature set and blend alpha on validation, never on test."""
    if alpha_grid is None:
        alpha_grid = np.linspace(0, 1, 11)
    if k < 1:
        raise ValueError("k must be positive")
    alphas = [float(np.round(x, 10)) for x in alpha_grid]
    if len(alphas) != len(set(alphas)) or not alphas or any(not 0 <= x <= 1 for x in alphas):
        raise ValueError("alpha_grid must contain unique weights in [0,1]")
    users, relevant, eligible_n = eligible_sample(val, max_users=max_users, seed=seed)
    if not users:
        raise ValueError("Validation holdout has no users with positive ratings")
    popularity = PopularityRanker().fit(train, games.game_id.astype(str))
    records = []
    for feature in FEATURE_SETS:
        content = ContentBasedRecommender(feature_set=feature).fit(train, games, mechanics, themes)
        hits = np.zeros(len(alphas), dtype=int)
        recommended_games = [set() for _ in alphas]
        for user in users:
            candidates, pop_rank = popularity.rank_for_user(user)
            content_prediction = content.predict_many(user, candidates)
            content_rank = _percentile_ranks(content_prediction)
            # candidates are alphabetically sorted, so ties are reproducible
            for j, alpha in enumerate(alphas):
                score = alpha * content_rank + (1 - alpha) * pop_rank
                indices = np.argsort(-score, kind="stable")[:k]
                recommended = {candidates[i] for i in indices}
                hits[j] += bool(recommended & relevant[user])
                recommended_games[j].update(recommended)
        for j, alpha in enumerate(alphas):
            records.append({"feature_set": feature, "content_weight": alpha,
                            "validation_hit_rate_at_10": hits[j]/len(users),
                            "hits": int(hits[j]), "evaluated_users": len(users),
                            "eligible_users": int(eligible_n),
                            "distinct_recommended_games": len(recommended_games[j])})
    validation = pd.DataFrame(records)
    # Prefer personalization when hit rates tie; use fewer candidate features as next tie break.
    winner = validation.sort_values(["validation_hit_rate_at_10", "content_weight", "feature_set"],
                                    ascending=[False, False, True], kind="stable").iloc[0]
    return validation, {"feature_set": str(winner.feature_set),
                        "content_weight": float(winner.content_weight)}, popularity


def test_selected(train, test, games, mechanics, themes, selection, *,
                  k=10, max_users=1000, seed=42):
    """Once-per-study evaluation of validation-selected ranking model on test."""
    content = ContentBasedRecommender(feature_set=selection["feature_set"]).fit(
        train, games, mechanics, themes)
    popularity = PopularityRanker().fit(train, games.game_id.astype(str))
    blended = BlendedRanker(content, popularity, selection["content_weight"])
    users, relevant, eligible_n = eligible_sample(test, max_users=max_users, seed=seed)
    hits = 0
    distinct = set()
    for user in users:
        result = blended.recommend(user, n=k)
        items = set(result.game_id)
        hits += bool(items & relevant[user])
        distinct.update(items)
    return {"hit_rate_at_10": hits / len(users) if users else None,
            "hits": hits, "evaluated_users": len(users),
            "eligible_users": int(eligible_n),
            "distinct_recommended_games": len(distinct)}


def select_balanced_candidate(validation: pd.DataFrame, best: dict,
                              max_absolute_hit_loss: float = 0.02,
                              min_coverage_multiplier: float = 1.5) -> dict | None:
    """Validation-only tradeoff: retain accuracy within 2 pp, >=1.5x coverage.

    Requires a nonzero content weight; independent of test labels.
    """
    top = validation.loc[(validation.feature_set == best["feature_set"]) &
                         (validation.content_weight == best["content_weight"])].iloc[0]
    eligible = validation.loc[
        (validation.content_weight > 0) &
        (validation.validation_hit_rate_at_10 >= top.validation_hit_rate_at_10 - max_absolute_hit_loss) &
        (validation.distinct_recommended_games >= top.distinct_recommended_games * min_coverage_multiplier)
    ]
    if eligible.empty:
        return None
    pick = eligible.sort_values(["content_weight", "validation_hit_rate_at_10", "feature_set"],
                                ascending=[False, False, True], kind="stable").iloc[0]
    return {"feature_set": str(pick.feature_set), "content_weight": float(pick.content_weight)}


def run_ranking_experiment(data_dir, output_dir, *, max_users=1000, k=10, seed=42):
    train, val, test, games, mechanics, themes = load_prepared(data_dir)
    validation, winner, _ = validate_ranking(
        train, val, games, mechanics, themes, k=k, max_users=max_users, seed=seed)
    balanced = select_balanced_candidate(validation, winner)
    # Both predefined strategies come from validation only. In this project the
    # same test split had already been viewed before this experiment; this is
    # explicitly an exploratory, rather than pristine, test comparison.
    test_result = test_selected(train, test, games, mechanics, themes, winner,
                                k=k, max_users=max_users, seed=seed)
    balanced_test = (test_selected(train, test, games, mechanics, themes, balanced,
                                   k=k, max_users=max_users, seed=seed)
                     if balanced is not None else None)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    validation.to_csv(destination / "validation_ranking_grid.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.3, 4.3))
    for feature, table in validation.groupby("feature_set", sort=True):
        ax.plot(table.content_weight, 100*table.validation_hit_rate_at_10, marker="o", label=feature)
    ax.set_xlabel("Content weight alpha (0 = popularity, 1 = content)")
    ax.set_ylabel("Validation Hit Rate@10 (%)")
    ax.set_title("Content + popularity ranking: validation only")
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination / "validation_ranking.png", dpi=160)
    fig.savefig(destination / "validation_ranking.svg")
    plt.close(fig)
    report = {
        "ranking_selection": "Maximum validation Hit Rate@10; tie-break favors higher content weight",
        "selection": winner,
        "best_validation": validation.loc[(validation.feature_set == winner["feature_set"]) &
                    (validation.content_weight == winner["content_weight"])].iloc[0].to_dict(),
        "test": test_result,
        "balanced_tradeoff": balanced,
        "balanced_validation": (validation.loc[(validation.feature_set == balanced["feature_set"]) &
                                (validation.content_weight == balanced["content_weight"])].iloc[0].to_dict()
                                if balanced is not None else None),
        "balanced_test": balanced_test,
        "balanced_selection_criterion": "Validation-only: nonzero content weight, <=2 pp hit-rate loss, >=1.5x distinct recommendation coverage",
        "validation_popularity_baseline": validation.loc[validation.content_weight == 0].iloc[0].to_dict(),
        "validation_pure_content": validation.loc[validation.content_weight == 1].to_dict("records"),
        "data_note": "User-provided BGG, 4,997-user sampled prepared data. No user identities stored.",
        "evaluation_caveat": "Offline positive-only labels; not live CTR or actual user satisfaction.",
        "test_caveat": "Test split had been inspected before this improvement project; post-hoc exploratory comparison, not pristine holdout."
    }
    def conversion(obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        raise TypeError(f"Cannot JSON-serialize {type(obj)}")
    (destination / "ranking_results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=conversion), encoding="utf-8")
    return report
