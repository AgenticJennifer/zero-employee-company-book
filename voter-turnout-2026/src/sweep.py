"""W&B Sweep entrypoint — thin wrapper around train()."""

import argparse
import logging

import wandb
from dotenv import load_dotenv

from src.train import DEFAULT_CONFIG, train

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def sweep_train() -> None:
    """Entry point called by each W&B sweep agent worker.

    wandb.init() is called inside train(); the sweep agent injects
    hyperparameters via wandb.config before train() reads them.
    """
    with wandb.init():
        cfg = dict(wandb.config)
        train(config=cfg)


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

    override: dict = {}
    if args.target == "undecided":
        # Filter to low-propensity voters as a proxy for undecided segment
        override["target"] = "target_support"
        override["undecided_only"] = True
    elif args.target == "vote_propensity":
        override["target"] = "target_vote_propensity"
    else:
        override["target"] = "target_support"

    sweep_train()
