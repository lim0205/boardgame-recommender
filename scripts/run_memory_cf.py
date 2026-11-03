"""Memory-based CF model selection on preprocessed BGG data. Never tunes using Test.

Run: python scripts/run_memory_cf.py --data-dir data/processed --out outputs/bgg_memory_cf
"""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from boardrec.memory_cf_experiments import run_memory_cf_experiments


def main(argv=None):
    parser = argparse.ArgumentParser(description="BGG Memory-based CF validation experiment")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out", type=Path, default=Path("outputs/bgg_memory_cf"))
    parser.add_argument("--kinds", nargs="+", default=["user", "item"], choices=["user", "item"])
    parser.add_argument("--similarities", nargs="+", default=["cosine", "pearson"],
                        choices=["cosine", "pearson"])
    parser.add_argument("--ks", type=int, nargs="+", default=[10, 20, 40, 80])
    parser.add_argument("--min-support", type=int, default=3)
    parser.add_argument("--shrinkage", type=float, default=10.0)
    parser.add_argument("--item-reg", type=float, default=10.0)
    parser.add_argument("--max-hit-users", type=int, default=250)
    args = parser.parse_args(argv)
    report = run_memory_cf_experiments(args.data_dir, args.out, kinds=args.kinds,
                                       similarities=args.similarities, ks=args.ks,
                                       min_support=args.min_support, shrinkage=args.shrinkage,
                                       item_reg=args.item_reg, max_hit_users=args.max_hit_users)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
