"""Evaluate the Validation-selected models on TEST exactly once.

Run only after run_memory_cf.py and run_hybrid.py have finished:
  python scripts/run_final_test.py --data-dir data/processed --out outputs/final_test
"""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from boardrec.final_test import run_final_test


def main(argv=None):
    parser = argparse.ArgumentParser(description="One-time Test evaluation of Validation-selected models")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out", type=Path, default=Path("outputs/final_test"))
    parser.add_argument("--memory-cf-results", type=Path,
                        default=Path("outputs/bgg_memory_cf/results.json"))
    parser.add_argument("--hybrid-validation", type=Path,
                        default=Path("outputs/bgg_hybrid/validation_metrics.csv"))
    parser.add_argument("--content-features", default="both", choices=["mechanics", "themes", "both"])
    parser.add_argument("--svd-factors", type=int, default=10)
    parser.add_argument("--svd-scale", type=float, default=1.0)
    parser.add_argument("--max-hit-users", type=int, default=250)
    args = parser.parse_args(argv)
    report = run_final_test(args.data_dir, args.out, memory_cf_results=args.memory_cf_results,
                            hybrid_validation_csv=args.hybrid_validation,
                            content_feature_set=args.content_features,
                            svd_params={"n_factors": args.svd_factors, "residual_scale": args.svd_scale},
                            max_hit_users=args.max_hit_users)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
