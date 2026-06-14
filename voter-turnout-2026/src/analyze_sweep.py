"""Pull W&B sweep results and print a plain-English summary."""

import logging
import os

from dotenv import load_dotenv

import wandb

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TOP_N = 5
PRIMARY_METRIC = "val/pr_auc"
SAMPLE_COUNTY_SIZE = 10_000  # rough voters-per-county for impact estimation
BASELINE_RECALL = 0.70  # assumed baseline recall before sweep


def analyze_sweep(
    sweep_id: str,
    entity: str | None = None,
    project: str | None = None,
) -> str:
    """Pull completed sweep runs and produce a ranked performance summary.

    Args:
        sweep_id: W&B sweep ID string.
        entity: W&B entity (team) name. Falls back to WANDB_ENTITY env var.
        project: W&B project name. Falls back to WANDB_PROJECT env var.

    Returns:
        Plain-English summary string.
    """
    api = wandb.Api()
    entity = entity or os.environ.get("WANDB_ENTITY")
    project = project or os.environ.get("WANDB_PROJECT", "voter-turnout-2026")

    path = f"{entity}/{project}/sweeps/{sweep_id}" if entity else f"{project}/sweeps/{sweep_id}"
    sweep = api.sweep(path)
    runs = sorted(sweep.runs, key=lambda r: r.summary.get(PRIMARY_METRIC, 0.0), reverse=True)

    if not runs:
        return "No completed runs found in sweep."

    header = (
        f"{'Rank':<5} {'PR-AUC':<10} {'depth':<7} {'lr':<9} "
        f"{'n_est':<7} {'sub':<7} {'col':<7} {'spw':<7}"
    )
    print(f"\nTop {min(TOP_N, len(runs))} configs by {PRIMARY_METRIC}:\n")
    print(header)
    print("-" * len(header))

    for rank, run in enumerate(runs[:TOP_N], 1):
        cfg = run.config
        pr = run.summary.get(PRIMARY_METRIC, float("nan"))
        print(
            f"{rank:<5} {pr:<10.4f} "
            f"{cfg.get('max_depth', '?'):<7} "
            f"{cfg.get('learning_rate', float('nan')):<9.4f} "
            f"{cfg.get('n_estimators', '?'):<7} "
            f"{cfg.get('subsample', float('nan')):<7.3f} "
            f"{cfg.get('colsample_bytree', float('nan')):<7.3f} "
            f"{cfg.get('scale_pos_weight', float('nan')):<7.2f}"
        )

    best = runs[0]
    best_cfg = best.config
    best_pr = best.summary.get(PRIMARY_METRIC, float("nan"))
    best_recall = best.summary.get("val/recall", BASELINE_RECALL)

    additional = max(0, int((best_recall - BASELINE_RECALL) * 0.30 * SAMPLE_COUNTY_SIZE))
    recall_note = (
        f"Compared to baseline, recall improved {(best_recall - BASELINE_RECALL) * 100:.1f}% "
        f"— roughly {additional:,} additional supporters identified in this county."
        if additional > 0
        else "Recall at baseline level."
    )

    summary = (
        f"\nBest config: max_depth={best_cfg.get('max_depth')}, "
        f"lr={best_cfg.get('learning_rate', 0):.3f}, "
        f"achieves {best_pr:.3f} PR-AUC. {recall_note}"
    )
    print(summary)
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Summarize W&B sweep results")
    parser.add_argument("sweep_id", help="W&B sweep ID")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--project", default=None)
    args = parser.parse_args()

    analyze_sweep(args.sweep_id, args.entity, args.project)
