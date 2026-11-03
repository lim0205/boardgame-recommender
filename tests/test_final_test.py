import pandas as pd

from boardrec.demo_data import generate_demo
from boardrec.final_test import run_final_test, select_hybrid_alphas
from boardrec.hybrid_experiments import run_hybrid_experiments
from boardrec.memory_cf_experiments import run_memory_cf_experiments
from boardrec.preprocessing import PrepConfig, prepare


def test_select_hybrid_alphas_excludes_endpoints(tmp_path):
    path = tmp_path / "v.csv"
    pd.DataFrame({"cf_model": ["svd"] * 3 + ["memory_cf"] * 3,
                  "alpha": [0.0, 0.1, 1.0] * 2,
                  "rmse": [1.0, 1.1, 1.2, 0.9, 1.3, 1.25]}).to_csv(path, index=False)
    assert select_hybrid_alphas(path) == {"svd": 0.1, "memory_cf": 0.1}


def test_final_test_end_to_end(tmp_path):
    raw = generate_demo(tmp_path / "raw")
    prepared = tmp_path / "prepared"
    prepare(raw / "user_ratings.csv", raw / "games.csv", raw / "mechanics.csv",
            raw / "themes.csv", prepared, PrepConfig(top_games=60, min_user_ratings=8, max_users=20))
    run_memory_cf_experiments(prepared, tmp_path / "mcf", ks=(5,), max_hit_users=5)
    run_hybrid_experiments(prepared, tmp_path / "hyb", alphas=(0.0, 0.5, 1.0),
                           svd_params={"n_factors": 3}, memory_cf_params={"k": 5}, max_hit_users=5)
    report = run_final_test(prepared, tmp_path / "final",
                            memory_cf_results=tmp_path / "mcf" / "results.json",
                            hybrid_validation_csv=tmp_path / "hyb" / "validation_metrics.csv",
                            svd_params={"n_factors": 3}, max_hit_users=5)
    assert set(report["results"]) == {"content", "svd", "memory_cf", "hybrid_svd", "hybrid_memory_cf"}
    assert report["results"]["hybrid_svd"]["settings"] == {"alpha": 0.5}
    assert (tmp_path / "final" / "test_metrics.csv").stat().st_size > 0
