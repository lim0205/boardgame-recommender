"""SVD model selection on preprocessed BGG data. Never tunes using Test.

Run: python scripts/run_svd.py --data-dir data/processed --out outputs/bgg_svd
"""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from boardrec.svd_experiments import run_svd_experiments


def main(argv=None):
    parser = argparse.ArgumentParser(description='BGG Truncated-SVD validation experiment')
    parser.add_argument('--data-dir', type=Path, default=Path('data/processed'))
    parser.add_argument('--out', type=Path, default=Path('outputs/bgg_svd'))
    parser.add_argument('--factors', type=int, nargs='+', default=[10,20,40])
    parser.add_argument('--scales', type=float, nargs='+', default=[0.5,1.0])
    parser.add_argument('--item-reg', type=float, default=10.0)
    parser.add_argument('--max-hit-users', type=int, default=250)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    report = run_svd_experiments(args.data_dir, args.out, factors=args.factors,
                                 scales=args.scales, item_reg=args.item_reg,
                                 seed=args.seed, max_hit_users=args.max_hit_users)
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    main()
