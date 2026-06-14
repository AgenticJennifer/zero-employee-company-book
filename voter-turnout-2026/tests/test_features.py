"""Tests for src/features.py."""

from pathlib import Path

import pytest

SYNTHETIC_CSV = Path("voter-turnout-2026/data/synthetic/voters.csv")


@pytest.fixture(scope="session")
def voter_df():
    if not SYNTHETIC_CSV.exists():
        pytest.skip("Run `make generate-data` first")
    from src.ingest import load_voter_file
    df, _ = load_voter_file(SYNTHETIC_CSV)
    return df


def test_feature_matrix_shape(voter_df):
    from src.features import FEATURE_COLUMNS, engineer_features
    X = engineer_features(voter_df)[FEATURE_COLUMNS]
    assert X.shape == (len(voter_df), len(FEATURE_COLUMNS))


def test_no_nan_in_output_features(voter_df):
    from src.features import FEATURE_COLUMNS, engineer_features
    X = engineer_features(voter_df)[FEATURE_COLUMNS]
    assert not X.isnull().any().any()


def test_class_imbalance_within_expected_range(voter_df):
    from src.features import split_features
    _, _, _, y_train, _, _ = split_features(voter_df)
    rate = y_train.mean()
    assert 0.20 <= rate <= 0.45, f"Positive rate {rate:.3f} outside [0.20, 0.45]"


def test_turnout_consistency_ratio_bounded(voter_df):
    from src.features import engineer_features
    out = engineer_features(voter_df)
    assert out["turnout_consistency_ratio"].between(0.0, 1.0).all()


def test_registration_age_ratio_bounded(voter_df):
    from src.features import engineer_features
    out = engineer_features(voter_df)
    assert out["registration_age_ratio"].between(0.0, 1.0).all()
