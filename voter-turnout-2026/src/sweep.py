"""W&B Sweep entrypoint — thin wrapper around train()."""

import argparse
import logging

from dotenv import load_dotenv

from src.train import train

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def sweep_train(base_config: dict | None = None) -> None:
    """Entry point called by each W&B sweep agent worker.

    The sweep agent runs this script as a subprocess and injects
    hyperparameters via environment variables. train() calls wandb.init()
    exactly once, picking those vars up automatically. Do NOT wrap this
    in a second wandb.init() call — that would create a duplicate run.

    Args:
        base_config: Non-hyperparameter overrides merged into DEFAULT_CONFIG
                     before the sweep agent's values take precedence
                     (e.g. target column, build_tables flag).
    """
    train(config=base_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="W&B Sweep entrypoint")
    parser.add_argument(
        "--target",
        choices=["support", "vote_propensity", "undecided"],
        default="support",
        help=(
            "Prediction target. 'undecided' restricts training to voters "
            "with low propensity scores (synthetic proxy for uncertain voters)."
        ),
    )
    args = parser.parse_args()

    # Sweep agents skip SHAP table building — too expensive per trial.
    # Run `make explain RUN_ID=<best-run-id>` once after the sweep to add tables.
    base: dict = {"build_tables": False}

    if args.target == "undecided":
        base["target"] = "target_support"
        base["undecided_only"] = True
    elif args.target == "vote_propensity":
        base["target"] = "target_vote_propensity"
    else:
        base["target"] = "target_support"

    sweep_train(base_config=base)
