"""Load and validate voter file CSVs."""

import hashlib
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {
    "voter_id",
    "county",
    "age",
    "party_registration",
    "years_registered",
    "midterm_turnout_count",
    "presidential_turnout_count",
    "address_changes_4yr",
    "mail_ballot_history",
    "median_income_tract",
    "target_vote_propensity",
    "target_support",
}

VALID_PARTIES = {"DEM", "REP", "IND", "NPA"}


def sha256_of_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file.

    Args:
        path: Path to the file.

    Returns:
        64-character hex digest string.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_voter_file(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load and validate a voter file CSV.

    Args:
        path: Path to the CSV.

    Returns:
        Tuple of (DataFrame, metadata dict with row_count, positive_rate,
        sha256_hash, ingest_date, county_scope).

    Raises:
        ValueError: If schema validation fails.
    """
    df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df["voter_id"].isnull().any():
        raise ValueError("voter_id column contains nulls")

    if df["voter_id"].duplicated().any():
        logger.warning("Duplicate voter_id values detected")

    bad_propensity = ~df["target_vote_propensity"].between(0.0, 1.0, inclusive="both")
    if bad_propensity.any():
        raise ValueError(f"{bad_propensity.sum()} rows have target_vote_propensity outside [0, 1]")

    invalid_party = ~df["party_registration"].isin(VALID_PARTIES)
    if invalid_party.any():
        logger.warning("%d rows have unexpected party_registration values", invalid_party.sum())

    metadata: dict[str, Any] = {
        "row_count": len(df),
        "positive_rate": float(df["target_support"].mean()),
        "sha256_hash": sha256_of_file(path),
        "ingest_date": pd.Timestamp.now(tz="UTC").isoformat(),
        "county_scope": sorted(df["county"].unique().tolist()),
    }
    logger.info(
        "Loaded %d voters from %s (positive rate: %.1f%%)",
        len(df),
        path,
        metadata["positive_rate"] * 100,
    )
    return df, metadata
