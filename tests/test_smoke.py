from pathlib import Path


def test_sample_data_exists():
    assert Path("data/sample/sample_hotels.csv").exists()
