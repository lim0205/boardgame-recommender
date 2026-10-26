"""Schema and ZIP tests use imaginary records, never actual BGG data."""
from zipfile import ZipFile
import pandas as pd
import pytest
from boardrec.bgg_io import find_csvs, extract_required_csvs, check_bgg_schema


def _sample_zip(path):
    samples = {
        "games.csv": "BGGId,Name\n100,Imaginary One\n101,Imaginary Two\n",
        "user_ratings.csv": "BGGId,Username,Rating\n100,test_user_1,9\n101,test_user_1,7\n",
        "mechanics.csv": "BGGId,Dice Rolling,Deck Building\n100,1,0\n101,0,1\n",
        "themes.csv": "BGGId,Fantasy,Science Fiction\n100,1,0\n101,0,1\n",
    }
    with ZipFile(path, "w") as zf:
        for name, body in samples.items():
            zf.writestr("nested/" + name, body)


def test_import_bgg_csv_header_pattern(tmp_path):
    archive = tmp_path / "dataset.zip"
    _sample_zip(archive)
    paths = extract_required_csvs(archive, tmp_path / "raw")
    assert set(paths) == {"user_ratings.csv", "games.csv", "mechanics.csv", "themes.csv"}
    assert check_bgg_schema(paths)["user_ratings.csv"] == ["BGGId", "Username", "Rating"]
    assert paths == find_csvs(tmp_path / "raw")
    assert pd.read_csv(paths["user_ratings.csv"]).shape == (2, 3)


def test_missing_file_rejected(tmp_path):
    archive = tmp_path / "bad.zip"
    with ZipFile(archive, "w") as zf:
        zf.writestr("user_ratings.csv", "BGGId,Username,Rating\n100,a,8\n")
    with pytest.raises(ValueError, match="missing="):
        extract_required_csvs(archive, tmp_path / "raw")


def test_duplicate_file_rejected(tmp_path):
    archive = tmp_path / "duplicates.zip"
    _sample_zip(archive)
    with ZipFile(archive, "a") as zf:
        zf.writestr("second/games.csv", "BGGId,Name\n1,Other\n")
    with pytest.raises(ValueError, match="duplicate="):
        extract_required_csvs(archive, tmp_path / "raw")
