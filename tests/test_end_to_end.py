import json
from pathlib import Path

from boardrec.cli import main


def test_demo_cli_produces_reproducible_artifacts(tmp_path):
    work = tmp_path / "demo"
    output = tmp_path / "outputs"
    main(["demo", "--work-dir", str(work), "--out", str(output),
          "--top-games", "35", "--min-user-ratings", "8", "--max-users", "25",
          "--chunksize", "73"])
    for name in ["results.json", "metrics.csv", "validation_rmse.png", "example_top10.csv"]:
        assert (output / name).stat().st_size > 0
    result = json.loads((output / "results.json").read_text(encoding="utf-8"))
    assert result["best_feature_set"] in {"mechanics", "themes", "both"}
    assert result["test"]["n_ratings"] > 0
