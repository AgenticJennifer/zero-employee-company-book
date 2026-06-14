"""Artifact versioning helpers for dataset and model lineage."""

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import wandb
import xgboost as xgb

from src.ingest import load_voter_file, sha256_of_file

logger = logging.getLogger(__name__)


def log_dataset_artifact(
    run: wandb.run.Run,
    path: Path,
    county: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> wandb.Artifact:
    """Create and log a dataset artifact with provenance metadata.

    Args:
        run: Active W&B run.
        path: Path to the voter file CSV.
        county: County scope string, or None for all counties.
        metadata: Pre-computed ingest metadata. Re-runs ingest if None.

    Returns:
        The logged Artifact.
    """
    scope = county or "all"
    artifact = wandb.Artifact(name=f"voterfile-{scope}", type="dataset")
    artifact.add_file(str(path))

    if metadata is None:
        _, metadata = load_voter_file(path)

    artifact.metadata.update(
        {
            "row_count": metadata.get("row_count"),
            "ingest_date": metadata.get("ingest_date"),
            "sha256_hash": metadata.get("sha256_hash") or sha256_of_file(path),
            "county_scope": county or "all",
            "positive_rate": metadata.get("positive_rate"),
        }
    )

    run.log_artifact(artifact)
    logger.info("Logged dataset artifact voterfile-%s @ %s", scope, path)
    return artifact


def log_model_artifact(
    run: wandb.run.Run,
    model: xgb.XGBClassifier,
    feature_names: list[str],
    metadata: dict[str, Any] | None = None,
) -> wandb.Artifact:
    """Save and log an XGBoost model artifact linked to the current run.

    Args:
        run: Active W&B run (used to set training_run_id in metadata).
        model: Trained XGBClassifier.
        feature_names: Ordered list of feature column names used at training time.
        metadata: Evaluation metrics to attach (val_pr_auc, recall, etc.).

    Returns:
        The logged Artifact.
    """
    artifact = wandb.Artifact(name="xgb-voter-model", type="model")

    with tempfile.TemporaryDirectory() as tmp:
        model_path = Path(tmp) / "model.json"
        model.save_model(str(model_path))

        features_path = Path(tmp) / "features.json"
        features_path.write_text(json.dumps(feature_names))

        artifact.add_file(str(model_path))
        artifact.add_file(str(features_path))

    artifact.metadata.update(
        {
            "n_features": len(feature_names),
            "training_run_id": run.id,
            **(metadata or {}),
        }
    )

    run.log_artifact(artifact)
    logger.info("Logged model artifact for run %s", run.id)
    return artifact


def load_dataset_artifact(
    run: wandb.run.Run,
    name: str,
    version: str = "latest",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Download and load a versioned dataset artifact.

    Args:
        run: Active W&B run (records usage lineage).
        name: Artifact name without version suffix, e.g. 'voterfile-all'.
        version: Version alias ('latest') or number string ('v0', 'v1').

    Returns:
        Tuple of (DataFrame, metadata dict).
    """
    artifact = run.use_artifact(f"{name}:{version}")
    artifact_dir = Path(artifact.download())
    csv_files = list(artifact_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV found in artifact {name}:{version}")
    df, _ = load_voter_file(csv_files[0])
    return df, dict(artifact.metadata)


def load_model_artifact(
    run: wandb.run.Run,
    name: str = "xgb-voter-model",
    version: str = "latest",
) -> tuple[xgb.XGBClassifier, dict[str, Any]]:
    """Download and load a versioned model artifact.

    Args:
        run: Active W&B run.
        name: Artifact name without version suffix.
        version: Version alias or number string.

    Returns:
        Tuple of (XGBClassifier, metadata dict).
    """
    artifact = run.use_artifact(f"{name}:{version}")
    artifact_dir = Path(artifact.download())

    model = xgb.XGBClassifier()
    model.load_model(str(artifact_dir / "model.json"))

    logger.info("Loaded model from artifact %s:%s", name, version)
    return model, dict(artifact.metadata)


def diff_datasets(artifact_v1: wandb.Artifact, artifact_v2: wandb.Artifact) -> str:
    """Produce a plain-English diff between two dataset artifact versions.

    Args:
        artifact_v1: Older artifact version.
        artifact_v2: Newer artifact version.

    Returns:
        Human-readable summary string, e.g.:
        'New version added 1,243 registrations (+2.5%). Positive rate shifted
        from 29.8% to 31.2%. Ingest date: 2026-06-13.'
    """
    m1, m2 = artifact_v1.metadata, artifact_v2.metadata

    rows1 = int(m1.get("row_count", 0))
    rows2 = int(m2.get("row_count", 0))
    delta = rows2 - rows1
    pct = (delta / rows1 * 100) if rows1 else 0.0

    pos1 = float(m1.get("positive_rate", 0.0))
    pos2 = float(m2.get("positive_rate", 0.0))
    date2 = str(m2.get("ingest_date", "unknown"))[:10]

    direction = "added" if delta >= 0 else "removed"
    sign = "+" if delta >= 0 else "-"

    return (
        f"New version {direction} {abs(delta):,} registrations "
        f"({sign}{abs(pct):.1f}%). "
        f"Positive rate shifted from {pos1 * 100:.1f}% to {pos2 * 100:.1f}%. "
        f"Ingest date: {date2}."
    )
