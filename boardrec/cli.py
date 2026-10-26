"""Command-line entrypoint: prepare / run / demo."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .preprocessing import PrepConfig, prepare
from .experiments import run_experiments
from .demo_data import generate_demo


def _config(args):
    return PrepConfig(top_games=args.top_games, min_user_ratings=args.min_user_ratings,
                      max_users=args.max_users, seed=args.seed, chunksize=args.chunksize)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Board-game content-based recommendation project")
    sub = parser.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--top-games", type=int, default=1000)
    common.add_argument("--min-user-ratings", type=int, default=20)
    common.add_argument("--max-users", type=int, default=20000)
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--chunksize", type=int, default=200000)

    p = sub.add_parser("prepare", parents=[common], help="Read authorized input CSVs and split ratings")
    for file in ("ratings", "games", "mechanics", "themes"):
        p.add_argument(f"--{file}", required=True, type=Path)
    p.add_argument("--out", default="data/processed", type=Path)

    p = sub.add_parser("run", help="Evaluate variants using prepared data")
    p.add_argument("--data-dir", default="data/processed", type=Path)
    p.add_argument("--out", default="outputs", type=Path)
    p.add_argument("--max-hit-users", type=int, default=500)

    p = sub.add_parser("demo", parents=[common], help="Generate synthetic data, prepare, evaluate")
    p.add_argument("--work-dir", default="data/demo", type=Path)
    p.add_argument("--out", default="outputs/demo", type=Path)

    args = parser.parse_args(argv)
    if args.cmd == "prepare":
        result = prepare(args.ratings, args.games, args.mechanics, args.themes,
                         args.out, _config(args))
    elif args.cmd == "run":
        result = run_experiments(args.data_dir, args.out, max_hit_users=args.max_hit_users)
    else:
        raw = generate_demo(args.work_dir / "raw")
        prepared = args.work_dir / "processed"
        result = {"preparation": prepare(raw / "user_ratings.csv", raw / "games.csv",
                                         raw / "mechanics.csv", raw / "themes.csv",
                                         prepared, _config(args)),
                  "experiment": run_experiments(prepared, args.out)}
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
