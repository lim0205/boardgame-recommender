"""Runnable example demonstrating API integration, NOT a CF model.

Run after `python -m boardrec.cli demo`:
    python examples/teammate_integration.py
Replace baseline_predict with the fitted teammate CF/SVD model's predict function.
"""
from pathlib import Path

from boardrec.adapters import PredictAdapter
from boardrec.evaluation import evaluate
from boardrec.preprocessing import load_prepared


def main():
    train, val, _test, games, mechanics, themes = load_prepared("data/demo/processed")
    user_means = train.groupby("user_id").rating.mean().to_dict()
    global_mean = float(train.rating.mean())

    def baseline_predict(user_id: str, game_id: str) -> float:
        # Placeholder for demonstration; NOT memory CF or SVD.
        return float(user_means.get(str(user_id), global_mean))

    adapter = PredictAdapter(baseline_predict, train=train, catalog_game_ids=games.game_id)
    metrics = evaluate(adapter, val, max_hit_users=100)
    print("Baseline (train user mean), validation only:", metrics)


if __name__ == "__main__":
    main()
