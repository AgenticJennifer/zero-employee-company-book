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


def build_voter_table(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    voter_metadata: pd.DataFrame,
    run: wandb.run.Run,
    max_rows: int = 5000,
) -> wandb.Table:
    """Build and log a W&B voter profile table with SHAP explanations.

    Args:
        model: Trained XGBClassifier.
        X_test: Engineered feature matrix for the test split.
        voter_metadata: Original voter DataFrame (voter_id, county, age, party_registration).
        run: Active W&B run to log the table to.
        max_rows: Row cap — large tables slow W&B upload.

    Returns:
        The logged wandb.Table.
    """
    if len(X_test) > max_rows:
        X_test = X_test.iloc[:max_rows]
        voter_metadata = voter_metadata.iloc[:max_rows]

    propensity_scores = model.predict_proba(X_test.values)[:, 1]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test.values)

    feat_names = list(X_test.columns)
    rows: list[list[Any]] = []

    for i in range(len(X_test)):
        sv = shap_values[i]
        feat_row = X_test.iloc[i]

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
                round(float(propensity_scores[i]), 4),  # same model; separate support model would differ
                top3_str,
                reason,
                _risk_tier(float(propensity_scores[i])),
            ]
        )

    columns = [
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
    voter_table = wandb.Table(data=rows, columns=columns)
    run.log({"voter_profiles": voter_table})
    logger.info("Logged voter profile table (%d rows)", len(rows))
    return voter_table
