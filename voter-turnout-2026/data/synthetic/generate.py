"""Generate synthetic voter file data for development."""

import argparse
import logging
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

COUNTIES = ["Riverside", "Summit", "Lakewood", "Hillcrest", "Pinecrest"]
PARTIES = ["DEM", "REP", "IND", "NPA"]
N_ROWS = 50_000
POSITIVE_RATE = 0.30
RANDOM_SEED = 42


def generate(output_path: Path, n_rows: int = N_ROWS, seed: int = RANDOM_SEED) -> Path:
    """Generate synthetic voter file CSV.

    Args:
        output_path: Destination file path.
        n_rows: Number of rows to generate.
        seed: Random seed for reproducibility.

    Returns:
        Path to the generated CSV file.
    """
    rng = np.random.default_rng(seed)

    party = rng.choice(PARTIES, size=n_rows, p=[0.38, 0.33, 0.18, 0.11])
    age = rng.integers(18, 86, size=n_rows)
    years_registered = np.clip(rng.integers(0, 41, size=n_rows), 0, age - 18)
    midterm_turnout = rng.integers(0, 6, size=n_rows)
    presidential_turnout = rng.integers(0, 7, size=n_rows)
    address_changes = rng.integers(0, 5, size=n_rows)
    mail_ballot = rng.integers(0, 9, size=n_rows)
    median_income = rng.uniform(25_000, 150_001, size=n_rows)

    # Propensity derived from features with added noise
    propensity_raw = (
        0.30 * (midterm_turnout / 5)
        + 0.25 * (presidential_turnout / 6)
        + 0.15 * np.clip((age - 18) / 67, 0, 1)
        + 0.15 * (mail_ballot / 8)
        + 0.10 * np.clip(years_registered / 40, 0, 1)
        - 0.05 * (address_changes / 4)
        + 0.05 * np.clip((median_income - 25_000) / 125_000, 0, 1)
        + rng.normal(0, 0.1, size=n_rows)
    )
    propensity = np.clip(propensity_raw, 0.0, 1.0)

    # Deliberate ~30% positive rate on binary support target
    threshold = np.percentile(propensity, 100 * (1 - POSITIVE_RATE))
    support = (propensity >= threshold).astype(int)

    df = pd.DataFrame(
        {
            "voter_id": [str(uuid.uuid4()) for _ in range(n_rows)],
            "county": rng.choice(COUNTIES, size=n_rows),
            "age": age,
            "party_registration": party,
            "years_registered": years_registered,
            "midterm_turnout_count": midterm_turnout,
            "presidential_turnout_count": presidential_turnout,
            "address_changes_4yr": address_changes,
            "mail_ballot_history": mail_ballot,
            "median_income_tract": median_income.round(2),
            "target_vote_propensity": propensity.round(4),
            "target_support": support,
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(
        "Generated %d rows → %s  (positive rate: %.1f%%)",
        n_rows,
        output_path,
        support.mean() * 100,
    )
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic voter file")
    parser.add_argument("--output", default="data/synthetic/voters.csv")
    parser.add_argument("--rows", type=int, default=N_ROWS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    generate(Path(args.output), args.rows, args.seed)
