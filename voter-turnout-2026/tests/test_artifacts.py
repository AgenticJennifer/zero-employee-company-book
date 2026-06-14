"""Tests for src/artifacts.py."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

SYNTHETIC_CSV = Path("voter-turnout-2026/data/synthetic/voters.csv")


def test_diff_datasets_returns_nonempty_string():
    from src.artifacts import diff_datasets

    a1 = MagicMock()
    a1.metadata = {
        "row_count": 50_000,
        "positive_rate": 0.298,
        "ingest_date": "2026-06-01T00:00:00+00:00",
    }
    a2 = MagicMock()
    a2.metadata = {
        "row_count": 51_243,
        "positive_rate": 0.312,
        "ingest_date": "2026-06-13T00:00:00+00:00",
    }

    result = diff_datasets(a1, a2)
    assert isinstance(result, str)
    assert len(result) > 0
    assert "1,243" in result
    assert "+2.5" in result


def test_diff_datasets_handles_removal():
    from src.artifacts import diff_datasets

    a1 = MagicMock()
    a1.metadata = {"row_count": 50_000, "positive_rate": 0.30, "ingest_date": "2026-05-01T00:00:00+00:00"}
    a2 = MagicMock()
    a2.metadata = {"row_count": 48_000, "positive_rate": 0.29, "ingest_date": "2026-06-01T00:00:00+00:00"}

    result = diff_datasets(a1, a2)
    assert "removed" in result.lower()


def test_log_dataset_artifact_does_not_throw():
    if not SYNTHETIC_CSV.exists():
        pytest.skip("Run `make generate-data` first")

    from src.artifacts import log_dataset_artifact
    from src.ingest import load_voter_file

    _, metadata = load_voter_file(SYNTHETIC_CSV)
    mock_run = MagicMock()
    mock_artifact = MagicMock()
    mock_artifact.metadata = {}

    with patch("src.artifacts.wandb.Artifact", return_value=mock_artifact):
        log_dataset_artifact(mock_run, SYNTHETIC_CSV, county=None, metadata=metadata)

    mock_run.log_artifact.assert_called_once()
