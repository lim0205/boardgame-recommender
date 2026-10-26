"""Reproduce BGG rank-blending selection on already-prepared, authorized data.

Usage:
python scripts/run_ranking.py --prepared-dir data/processed --rights-confirmed
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from boardrec.ranking_experiments import run_ranking_experiment


def main():
    parser = argparse.ArgumentParser(description="Validation-based content and popularity rank tuning")
    parser.add_argument("--prepared-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out", type=Path, default=Path("outputs/bgg_ranking"))
    parser.add_argument("--max-users", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rights-confirmed", action="store_true")
    args = parser.parse_args()
    if not args.rights_confirmed:
        parser.error("Confirm that your dataset use is permitted before running this experiment")
    result = run_ranking_experiment(args.prepared_dir, args.out,
                                    max_users=args.max_users, seed=args.seed)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
