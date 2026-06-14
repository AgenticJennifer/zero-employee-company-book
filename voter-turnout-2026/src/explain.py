"""SHAP-based voter profile tables and plain-English explanations."""

import logging
from typing import Any

import numpy as np
import pandas as pd
import shap
import wandb
import xgboost as xgb

logger = logging.getLogger(__name__)

_HIGH_THRESHOLD = 0.65
_MEDIUM_THRESHOLD = 0.35

VOTER_TABLE_COLUMNS = [
    "voter_id",
    "county",
    "age_band",
    "party",
    "predicted_propensity",
    "predicted_support_prob",
    "top_3_features",
    "plain_english_reason",
    "risk_tier",
]

COUNTY_SUMMARY_COLUMNS = [
    "county",
    "total_voters",
    "high_tier",
    "medium_tier",
    "low_tier",
    "high_tier_pct",
    "avg_propensity",
]


def _risk_tier(score: float) -> str:
    if score >= _HIGH_THRESHOLD:
        return "High"
    if score >= _MEDIUM_THRESHOLD:
        return "Medium"
    return "Low"


def _age_band(age: int) -> str:
    if age < 30:
        return "18-29"
    if age < 45:
        return "30-44"
    if age < 60:
        return "45-59"
    if age < 75:
        return "60-74"
    return "75+"


def generate_reason(
    shap_values: np.ndarray,
    feature_names: list[str],
    row: pd.Series,
) -> str:
    """Generate a plain-English explanation from SHAP values for one voter.

    Args:
        shap_values: 1-D array of SHAP values, aligned with feature_names.
        feature_names: Feature names in the same order as shap_values.
        row: Feature values for this voter (pre-engineering originals preferred).

    Returns:
        Readable string, e.g. 'Consistent midterm voter (5/5). Long-term
        registrant (18 yrs). Mail ballot history suggests high accessibility.'
    """
    impact = dict(zip(feature_names, np.abs(shap_values)))
    top3 = sorted(impact, key=impact.get, reverse=True)[:3]  # type: ignore[arg-type]

    clauses: list[str] = []
    for feat in top3:
        val_raw = row.get(feat, None)
        if feat == "midterm_turnout_count" and val_raw is not None:
            v = int(val_raw)
            clauses.append(f"Consistent midterm voter ({v}/5)" if v >= 4 else f"Limited midterm history ({v}/5)")
        elif feat == "presidential_turnout_count" and val_raw is not None:
            v = int(val_raw)
            clauses.append(f"Strong presidential turnout ({v}/6)" if v >= 5 else f"Inconsistent presidential voting ({v}/6)")
        elif feat == "years_registered" and val_raw is not None:
            v = int(val_raw)
            clauses.append(f"Long-term registrant ({v} yrs)" if v >= 10 else f"Newly registered ({v} yrs)")
        elif feat == "mail_ballot_history" and val_raw is not None:
            v = int(val_raw)
            clauses.append("Mail ballot history suggests high accessibility" if v >= 3 else "No mail ballot history")
        elif feat == "age" and val_raw is not None:
            v = int(val_raw)
            if v < 30:
                clauses.append("Young voter segment — high uncertainty")
            elif v >= 65:
                clauses.append("Senior voter — typically reliable turnout")
            else:
                clauses.append(f"Mid-age voter ({v})")
        elif feat == "address_changes_4yr" and val_raw is not None:
            v = int(val_raw)
            clauses.append(
                f"Residential instability ({v} address changes)" if v >= 2 else "Stable residential address"
            )
        elif feat == "median_income_tract" and val_raw is not None:
            v = float(val_raw)
            clauses.append(
                "Higher-income tract (strong resource access)" if v >= 80_000 else "Lower-income tract"
            )
        else:
            clauses.append(feat.replace("_", " ").capitalize())

    return (" ".join(f"{c}." if not c.endswith(".") else c for c in clauses)).strip() if clauses else "Insufficient feature data."


def _build_rows(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    voter_metadata: pd.DataFrame,
    propensity_scores: np.ndarray,
) -> list[list[Any]]:
    """Assemble raw row data for voter profile tables.

    Shared by build_voter_table and build_county_tables so SHAP is computed once.

    Args:
        shap_values: (n, features) SHAP value matrix from TreeExplainer.
        X: Engineered feature DataFrame aligned with shap_values.
        voter_metadata: Original voter columns (voter_id, county, age, party_registration).
        propensity_scores: 1-D predicted probabilities, aligned with X.

    Returns:
        List of rows matching VOTER_TABLE_COLUMNS ordering.
    """
    feat_names = list(X.columns)
    rows: list[list[Any]] = []

    for i in range(len(X)):
        sv = shap_values[i]
        feat_row = X.iloc[i]
        meta_row = voter_metadata.iloc[i]

        top_idx = np.argsort(np.abs(sv))[::-1][:3]
        top3_str = "; ".join(
            f"{feat_names[j]}={feat_row.iloc[j]:.2f} (SHAP={sv[j]:+.3f})" for j in top_idx
        )
        reason = generate_reason(sv, feat_names, feat_row)

        rows.append(
            [
                str(meta_row.get("voter_id", "")),
                str(meta_row.get("county", "")),
                _age_band(int(meta_row.get("age", 25))),
                str(meta_row.get("party_registration", "")),
                round(float(propensity_scores[i]), 4),
                round(float(propensity_scores[i]), 4),
                top3_str,
                reason,
                _risk_tier(float(propensity_scores[i])),
            ]
        )
    return rows


def build_voter_table(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    voter_metadata: pd.DataFrame,
    run: wandb.run.Run,
    max_rows: int = 5000,
) -> wandb.Table:
    """Build and log a combined voter profile table with SHAP explanations.

    Args:
        model: Trained XGBClassifier.
        X_test: Engineered feature matrix for the test split.
        voter_metadata: Original voter DataFrame (voter_id, county, age, party_registration).
        run: Active W&B run to log the table to.
        max_rows: Row cap — large tables slow W&B upload.

    Returns:
        The logged wandb.Table (key: 'voter_profiles').
    """
    if len(X_test) > max_rows:
        X_test = X_test.iloc[:max_rows].reset_index(drop=True)
        voter_metadata = voter_metadata.iloc[:max_rows].reset_index(drop=True)
    else:
        X_test = X_test.reset_index(drop=True)
        voter_metadata = voter_metadata.reset_index(drop=True)

    scores = model.predict_proba(X_test.values)[:, 1]
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_test.values)

    rows = _build_rows(sv, X_test, voter_metadata, scores)
    table = wandb.Table(data=rows, columns=VOTER_TABLE_COLUMNS)
    run.log({"voter_profiles": table})
    logger.info("Logged voter profile table (%d rows)", len(rows))
    return table


def build_county_tables(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    voter_metadata: pd.DataFrame,
    run: wandb.run.Run,
    max_rows_per_county: int = 1000,
) -> dict[str, wandb.Table]:
    """Build per-county voter profile tables and an aggregated county summary table.

    Computes SHAP values once across all test rows, then partitions by county.
    Logs:
      - ``voter_profiles/{county}`` for each county (capped at max_rows_per_county)
      - ``county_summary``: High/Medium/Low tier counts + avg propensity per county
    Also stores county stats dict in ``run.summary['county_stats']`` for report assembly.

    Args:
        model: Trained XGBClassifier.
        X_test: Engineered feature matrix for the test split.
        voter_metadata: Original voter DataFrame with a 'county' column.
        run: Active W&B run.
        max_rows_per_county: Row cap per county table to keep W&B upload fast.

    Returns:
        Dict mapping county name to its logged wandb.Table.
    """
    X_reset = X_test.reset_index(drop=True)
    meta_reset = voter_metadata.reset_index(drop=True)

    scores = model.predict_proba(X_reset.values)[:, 1]
    explainer = shap.TreeExplainer(model)
    shap_matrix = explainer.shap_values(X_reset.values)

    all_rows = _build_rows(shap_matrix, X_reset, meta_reset, scores)

    counties = sorted(meta_reset["county"].unique().tolist())
    log_batch: dict[str, wandb.Table] = {}
    county_tables: dict[str, wandb.Table] = {}
    summary_rows: list[list[Any]] = []
    county_stats: dict[str, dict[str, float]] = {}

    for county in counties:
        county_rows = [r for r in all_rows if r[1] == county]
        if not county_rows:
            logger.warning("No test rows for county %s — skipping", county)
            continue

        tiers = [r[8] for r in county_rows]
        prop_scores = [r[4] for r in county_rows]
        high = tiers.count("High")
        medium = tiers.count("Medium")
        low = tiers.count("Low")
        total = len(county_rows)
        avg_prop = round(sum(prop_scores) / total, 4)
        high_pct = round(high / total * 100, 1)

        summary_rows.append([county, total, high, medium, low, high_pct, avg_prop])
        county_stats[county] = {
            "total": total,
            "high": high,
            "medium": medium,
            "low": low,
            "high_pct": high_pct,
            "avg_propensity": avg_prop,
        }

        capped = county_rows[:max_rows_per_county]
        county_table = wandb.Table(data=capped, columns=VOTER_TABLE_COLUMNS)
        log_batch[f"voter_profiles/{county}"] = county_table
        county_tables[county] = county_table
        logger.info("Built %s table: %d rows (%d High / %d Med / %d Low)", county, total, high, medium, low)

    summary_table = wandb.Table(data=summary_rows, columns=COUNTY_SUMMARY_COLUMNS)
    log_batch["county_summary"] = summary_table
    run.log(log_batch)

    # Store aggregated stats in run summary so report.py can fetch without re-downloading tables
    run.summary.update({"county_stats": county_stats})
    logger.info("Logged county_summary table and voter_profiles/{county} for %d counties", len(county_tables))
    return county_tables
