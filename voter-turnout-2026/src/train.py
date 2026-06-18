"""Main training loop with W&B experiment tracking."""

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import wandb
import xgboost as xgb
from dotenv import load_dotenv
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.artifacts import log_dataset_artifact, log_model_artifact
from src.features import FEATURE_COLUMNS, split_features
from src.ingest import load_voter_file

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CONFIG: dict[str, Any] = {
    "geographic_scope": "all_counties",
    "model_type": "xgboost",
    "target": "target_support",
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 2.3,  # ~1/positive_rate for 30% positive class
    "random_seed": 42,
    "train_split": 0.8,
    "val_split": 0.1,
    # False by default so sweep agents skip expensive SHAP; standalone `make train` sets True
    "build_tables": False,
}


class _WandbRoundLogger(xgb.callback.TrainingCallback):
    """Log per-round train/val logloss to an active W&B run."""

    def after_iteration(self, model: Any, epoch: int, evals_log: dict) -> bool:
        logs: dict[str, float] = {}
        for i, (dataset, metrics) in enumerate(evals_log.items()):
            prefix = "train/" if i == 0 else "val/"
            for metric_name, values in metrics.items():
                logs[f"{prefix}{metric_name}"] = values[-1]
        wandb.log(logs, step=epoch)
        return False


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, prefix: str) -> dict[str, float]:
    """Compute standard classification metrics.

    Args:
        y_true: Ground-truth binary labels.
        y_score: Predicted probabilities for the positive class.
        prefix: Metric key prefix, e.g. 'val/' or 'test/'.

    Returns:
        Dict mapping metric name to scalar value.
    """
    y_pred = (y_score >= 0.5).astype(int)
    return {
        f"{prefix}logloss": log_loss(y_true, y_score),
        f"{prefix}precision": precision_score(y_true, y_pred, zero_division=0),
        f"{prefix}recall": recall_score(y_true, y_pred, zero_division=0),
        f"{prefix}f1": f1_score(y_true, y_pred, zero_division=0),
        f"{prefix}pr_auc": average_precision_score(y_true, y_score),
        f"{prefix}roc_auc": roc_auc_score(y_true, y_score),
    }


def train(config: dict[str, Any] | None = None) -> None:
    """Run one training experiment and log everything to W&B.

    Args:
        config: Hyperparameter overrides merged on top of DEFAULT_CONFIG.
                Pass ``{"build_tables": True}`` to also build SHAP voter profile
                tables in the same run (skipped by default for sweep efficiency).
    """
    run_config = {**DEFAULT_CONFIG, **(config or {})}

    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "voter-turnout-2026"),
        entity=os.environ.get("WANDB_ENTITY") or None,
        config=run_config,
        tags=["baseline", run_config["geographic_scope"]],
        notes="XGBoost baseline on synthetic voter file",
    )
    cfg = wandb.config

    data_dir = Path(os.environ.get("DATA_PATH", "data/synthetic"))
    data_path = data_dir / "voters.csv"
    df, metadata = load_voter_file(data_path)

    if cfg.geographic_scope != "all_counties":
        df = df[df["county"] == cfg.geographic_scope].copy()
        metadata["county_scope"] = [cfg.geographic_scope]

    county_arg = cfg.geographic_scope if cfg.geographic_scope != "all_counties" else None
    log_dataset_artifact(run, data_path, county=county_arg, metadata=metadata)

    X_train, X_val, X_test, y_train, y_val, y_test = split_features(
        df,
        target=cfg.target,
        train_split=cfg.train_split,
        val_split=cfg.val_split,
        random_seed=cfg.random_seed,
    )

    model = xgb.XGBClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        scale_pos_weight=cfg.scale_pos_weight,
        random_state=cfg.random_seed,
        eval_metric="logloss",
        early_stopping_rounds=30,
    )

    # W&B captures CPU/memory metrics automatically each round — valuable for
    # sizing campaign field-office hardware where compute budgets are tight.
    model.fit(
        X_train.values,
        y_train.values,
        eval_set=[
            (X_train.values, y_train.values),
            (X_val.values, y_val.values),
        ],
        verbose=False,
        callbacks=[_WandbRoundLogger()],
    )

    val_score = model.predict_proba(X_val.values)[:, 1]
    val_metrics = compute_metrics(y_val.values, val_score, "val/")
    run.log(val_metrics)

    prec, rec, _ = precision_recall_curve(y_val.values, val_score)
    pr_table = wandb.Table(
        data=[[float(r), float(p)] for r, p in zip(rec, prec)],
        columns=["recall", "precision"],
    )
    run.log({"val/pr_curve": wandb.plot.line(pr_table, "recall", "precision", title="Precision-Recall Curve")})

    test_score = model.predict_proba(X_test.values)[:, 1]
    test_metrics = compute_metrics(y_test.values, test_score, "test/")
    run.log(test_metrics)

    val_meta = {k.replace("val/", ""): v for k, v in val_metrics.items()}
    log_model_artifact(run, model, FEATURE_COLUMNS, metadata=val_meta)

    if cfg.get("build_tables", False):
        from src.explain import build_county_tables, build_voter_table

        # X_test preserves the original df index, so we can recover voter metadata rows
        voter_meta_test = df.loc[X_test.index].reset_index(drop=True)
        X_test_reset = X_test.reset_index(drop=True)

        build_voter_table(model, X_test_reset, voter_meta_test, run)
        build_county_tables(model, X_test_reset, voter_meta_test, run)
        logger.info("Voter profile and county tables logged to run %s", run.id)

    logger.info(
        "Training complete | val/pr_auc=%.4f  test/pr_auc=%.4f",
        val_metrics["val/pr_auc"],
        test_metrics["test/pr_auc"],
    )
    run.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train XGBoost voter propensity model")
    parser.add_argument(
        "--no-tables",
        action="store_true",
        help="Skip voter profile and county table building (faster, for quick iterations)",
    )
    args = parser.parse_args()

    train(config={"build_tables": not args.no_tables})
