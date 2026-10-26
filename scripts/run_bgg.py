"""Run the existing recommender pipeline on a user-supplied, authorized BGG ZIP.

Usage:
  python scripts/run_bgg.py --archive data/raw/board-games.zip --rights-confirmed
  python scripts/run_bgg.py --csv-dir data/raw --rights-confirmed
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from boardrec.bgg_io import find_csvs, extract_required_csvs, check_bgg_schema
from boardrec.preprocessing import PrepConfig, prepare
from boardrec.experiments import run_experiments


def main(argv=None):
    parser = argparse.ArgumentParser(description="BGG CSV experiment (authorization required)")
    inp = parser.add_mutually_exclusive_group(required=True)
    inp.add_argument("--archive", type=Path, help="Kaggle ZIP already downloaded")
    inp.add_argument("--csv-dir", type=Path, help="Extracted CSV folder")
    parser.add_argument("--rights-confirmed", action="store_true",
                        help="Confirm permission for model training with this data")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out", type=Path, default=Path("outputs/bgg"))
    parser.add_argument("--top-games", type=int, default=1000)
    parser.add_argument("--min-user-ratings", type=int, default=20)
    parser.add_argument("--max-users", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-hit-users", type=int, default=500)
    args = parser.parse_args(argv)
    if not args.rights_confirmed:
        parser.error("Verify BGG data usage permission before training. --rights-confirmed is required.")
    paths = extract_required_csvs(args.archive, args.raw_dir) if args.archive else find_csvs(args.csv_dir)
    headers = check_bgg_schema(paths)
    cfg = PrepConfig(top_games=args.top_games, min_user_ratings=args.min_user_ratings,
                     max_users=args.max_users, seed=args.seed)
    manifest = prepare(paths["user_ratings.csv"], paths["games.csv"],
                       paths["mechanics.csv"], paths["themes.csv"],
                       args.processed_dir, cfg)
    report = run_experiments(args.processed_dir, args.out, max_hit_users=args.max_hit_users)
    print(json.dumps({"dataset": "user-provided Kaggle CSV", "headers": headers,
                      "preprocessing": manifest, "experiment": report},
                     ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
