"""Tests for src/ingest.py."""

from pathlib import Path

import pandas as pd
import pytest

SYNTHETIC_CSV = Path("voter-turnout-2026/data/synthetic/voters.csv")


@pytest.fixture(scope="session")
def voter_df():
    """Load synthetic voter file; skip if not yet generated."""
    if not SYNTHETIC_CSV.exists():
        pytest.skip("Run `make generate-data` first")
    from src.ingest import load_voter_file
    df, _ = load_voter_file(SYNTHETIC_CSV)
    return df


def test_schema_has_all_required_columns(voter_df):
    required = {
        "voter_id", "county", "age", "party_registration", "years_registered",
        "midterm_turnout_count", "presidential_turnout_count", "address_changes_4yr",
        "mail_ballot_history", "median_income_tract", "target_vote_propensity", "target_support",
    }
    missing = required - set(voter_df.columns)
    assert not missing, f"Missing columns: {missing}"


def test_no_null_voter_ids(voter_df):
    assert not voter_df["voter_id"].isnull().any()


def test_propensity_in_unit_interval(voter_df):
    assert voter_df["target_vote_propensity"].between(0.0, 1.0).all()


def test_target_support_is_binary(voter_df):
    vals = set(voter_df["target_support"].unique())
    assert vals.issubset({0, 1}), f"Unexpected values: {vals}"


def test_sha256_is_consistent():
    if not SYNTHETIC_CSV.exists():
        pytest.skip("Run `make generate-data` first")
    from src.ingest import sha256_of_file
    assert sha256_of_file(SYNTHETIC_CSV) == sha256_of_file(SYNTHETIC_CSV)
    assert len(sha256_of_file(SYNTHETIC_CSV)) == 64
