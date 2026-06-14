"""Feature engineering for the voter propensity model."""

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "age",
    "party_registration_encoded",
    "years_registered",
    "midterm_turnout_count",
    "presidential_turnout_count",
    "address_changes_4yr",
    "mail_ballot_history",
    "median_income_tract",
    "turnout_consistency_ratio",
    "registration_age_ratio",
]

# Fixed encoding order ensures consistent label encoding across train/inference
PARTY_ORDER = ["DEM", "REP", "IND", "NPA"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create feature matrix from raw voter file.

    Args:
        df: Raw voter DataFrame from load_voter_file.

    Returns:
        Copy of df with additional engineered feature columns.
    """
    out = df.copy()

    le = LabelEncoder()
    le.fit(PARTY_ORDER)
    out["party_registration_encoded"] = le.transform(out["party_registration"])

    total_elections = out["midterm_turnout_count"] + out["presidential_turnout_count"]
    max_possible = 5 + 6  # 5 midterms + 6 presidential cycles tracked
    out["turnout_consistency_ratio"] = total_elections / max_possible

    eligible_years = np.maximum(out["age"] - 18, 1)
    out["registration_age_ratio"] = np.clip(out["years_registered"] / eligible_years, 0.0, 1.0)

    return out


def split_features(
    df: pd.DataFrame,
    target: str = "target_support",
    train_split: float = 0.8,
    val_split: float = 0.1,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split data into stratified train / val / test sets.

    Args:
        df: Raw voter DataFrame.
        target: Column name for the label.
        train_split: Fraction of data for training.
        val_split: Fraction for validation (remainder goes to test).
        random_seed: Controls train_test_split randomness.

    Returns:
        Tuple (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    engineered = engineer_features(df)
    X = engineered[FEATURE_COLUMNS]
    y = engineered[target]

    test_split = 1.0 - train_split - val_split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(val_split + test_split), random_state=random_seed, stratify=y
    )
    relative_test = test_split / (val_split + test_split)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test, random_state=random_seed, stratify=y_temp
    )

    logger.info(
        "Split → train=%d  val=%d  test=%d  (positive rate train=%.1f%%)",
        len(X_train),
        len(X_val),
        len(X_test),
        y_train.mean() * 100,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
