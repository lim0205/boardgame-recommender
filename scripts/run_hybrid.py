"""Hybrid (Content + CF) alpha selection on Validation. Never tunes using Test.

Run after run_svd.py / run_memory_cf.py, passing their selected settings:
  python scripts/run_hybrid.py --data-dir data/processed --out outputs/bgg_hybrid \
      --svd-factors 10 --svd-scale 1.0 \
      --memory-cf-results outputs/bgg_memory_cf/results.json
"""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from boardrec.hybrid_experiments import run_hybrid_experiments, DEFAULT_ALPHAS


def main(argv=None):
    parser = argparse.ArgumentParser(description="BGG Hybrid (Content + CF) validation experiment")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out", type=Path, default=Path("outputs/bgg_hybrid"))
    parser.add_argument("--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS))
    parser.add_argument("--content-features", default="both", choices=["mechanics", "themes", "both"])
    parser.add_argument("--svd-factors", type=int, default=10)
    parser.add_argument("--svd-scale", type=float, default=1.0)
    parser.add_argument("--cf-kind", default="item", choices=["user", "item"])
    parser.add_argument("--cf-similarity", default="pearson", choices=["cosine", "pearson"])
    parser.add_argument("--cf-k", type=int, default=40)
    parser.add_argument("--cf-min-support", type=int, default=3)
    parser.add_argument("--cf-shrinkage", type=float, default=10.0)
    parser.add_argument("--memory-cf-results", type=Path, default=None,
                        help="results.json from run_memory_cf.py; overrides the --cf-* options")
    parser.add_argument("--max-hit-users", type=int, default=250)
    args = parser.parse_args(argv)
    if args.memory_cf_results is not None:
        best = json.loads(args.memory_cf_results.read_text(encoding="utf-8"))["best_hyperparameters"]
        args.cf_kind, args.cf_similarity, args.cf_k = best["kind"], best["similarity"], int(best["k"])
        args.cf_min_support, args.cf_shrinkage = int(best["min_support"]), float(best["shrinkage"])
    report = run_hybrid_experiments(
        args.data_dir, args.out, alphas=args.alphas, content_feature_set=args.content_features,
        svd_params={"n_factors": args.svd_factors, "residual_scale": args.svd_scale},
        memory_cf_params={"kind": args.cf_kind, "similarity": args.cf_similarity, "k": args.cf_k,
                          "min_support": args.cf_min_support, "shrinkage": args.cf_shrinkage},
        max_hit_users=args.max_hit_users)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
