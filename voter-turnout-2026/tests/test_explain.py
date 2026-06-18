"""Tests for src/explain.py pure functions (no W&B I/O required)."""

import numpy as np
import pandas as pd


def test_risk_tier_boundaries():
    from src.explain import _risk_tier

    assert _risk_tier(0.90) == "High"
    assert _risk_tier(0.65) == "High"  # inclusive lower bound
    assert _risk_tier(0.64) == "Medium"
    assert _risk_tier(0.35) == "Medium"  # inclusive lower bound
    assert _risk_tier(0.34) == "Low"
    assert _risk_tier(0.0) == "Low"


def test_age_band_binning():
    from src.explain import _age_band

    assert _age_band(18) == "18-29"
    assert _age_band(29) == "18-29"
    assert _age_band(30) == "30-44"
    assert _age_band(44) == "30-44"
    assert _age_band(59) == "45-59"
    assert _age_band(74) == "60-74"
    assert _age_band(75) == "75+"
    assert _age_band(85) == "75+"


def test_generate_reason_nonempty_for_consistent_voter():
    from src.explain import generate_reason

    feature_names = [
        "midterm_turnout_count",
        "years_registered",
        "mail_ballot_history",
        "age",
    ]
    # Heaviest SHAP weight on midterm turnout, then registration tenure, then mail ballot
    shap_values = np.array([0.5, 0.3, 0.2, 0.05])
    row = pd.Series(
        {
            "midterm_turnout_count": 5,
            "years_registered": 18,
            "mail_ballot_history": 4,
            "age": 60,
        }
    )

    reason = generate_reason(shap_values, feature_names, row)
    assert isinstance(reason, str)
    assert len(reason) > 0
    assert "Consistent midterm voter (5/5)" in reason
    assert "Long-term registrant (18 yrs)" in reason


def test_generate_reason_newly_registered_segment():
    from src.explain import generate_reason

    feature_names = ["years_registered", "midterm_turnout_count", "age"]
    shap_values = np.array([0.4, 0.3, 0.3])
    row = pd.Series({"years_registered": 0, "midterm_turnout_count": 0, "age": 22})

    reason = generate_reason(shap_values, feature_names, row)
    assert "Newly registered" in reason
    # Young voter clause should appear given age=22 is in the top-3 by SHAP
    assert "Young voter" in reason or "Limited midterm" in reason


def test_generate_reason_handles_empty_gracefully():
    from src.explain import generate_reason

    reason = generate_reason(np.array([]), [], pd.Series(dtype=float))
    assert isinstance(reason, str)
    assert len(reason) > 0


def test_county_breakdown_md_with_stats():
    from src.report import _county_breakdown_md

    county_stats = {
        "Riverside": {"total": 1200, "high": 360, "medium": 500, "low": 340, "high_pct": 30.0, "avg_propensity": 0.512},
        "Summit": {"total": 980, "high": 200, "medium": 480, "low": 300, "high_pct": 20.4, "avg_propensity": 0.448},
    }
    md = _county_breakdown_md(county_stats, "my-team", "voter-turnout-2026", "abc123")

    assert "Riverside" in md
    assert "Summit" in md
    assert "1,200" in md  # thousands separator
    assert "30.0%" in md
    assert "voter_profiles/Riverside" in md  # per-county table key link
    assert "abc123" in md  # run id deep link


def test_county_breakdown_md_empty_stats_fallback():
    from src.report import _county_breakdown_md

    md = _county_breakdown_md({}, "my-team", "voter-turnout-2026", "abc123")
    assert isinstance(md, str)
    assert "build_county_tables" in md  # helpful guidance when stats missing


def test_column_constants_match_row_width():
    from src.explain import COUNTY_SUMMARY_COLUMNS, VOTER_TABLE_COLUMNS

    assert len(VOTER_TABLE_COLUMNS) == 9
    assert len(COUNTY_SUMMARY_COLUMNS) == 7
    assert VOTER_TABLE_COLUMNS[8] == "risk_tier"  # _build_rows indexes [8] for tier
    assert VOTER_TABLE_COLUMNS[1] == "county"  # _build_rows indexes [1] for county
    assert VOTER_TABLE_COLUMNS[4] == "predicted_propensity"  # indexed [4] for avg
