"""W&B Report assembly for campaign decision-making."""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

import wandb

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _county_breakdown_md(county_stats: dict, entity: str, project: str, run_id: str) -> str:
    """Render county breakdown as a markdown table with direct W&B table links.

    Args:
        county_stats: Dict from run.summary['county_stats'], keyed by county name.
        entity: W&B entity (team).
        project: W&B project name.
        run_id: ID of the run that logged the county tables.

    Returns:
        Markdown string with county stats table and per-county deep links.
    """
    if not county_stats:
        return (
            "_County breakdown not yet available. Run `build_county_tables()` from "
            "`src/explain.py` after training, then re-run `src/report.py`._\n\n"
            "Stats are stored in `run.summary['county_stats']` for automatic pickup here."
        )

    header = "| County | Voters | High tier | Medium tier | Low tier | High% | Avg propensity |\n"
    divider = "|--------|--------|-----------|-------------|----------|-------|----------------|\n"
    rows_md = ""
    links_md = "\n**Per-county voter tables** (full SHAP detail, sortable/filterable):\n\n"

    base_url = f"https://wandb.ai/{entity}/{project}/runs/{run_id}/tables"

    for county, stats in sorted(county_stats.items()):
        rows_md += (
            f"| {county} "
            f"| {stats.get('total', 0):,} "
            f"| {stats.get('high', 0):,} "
            f"| {stats.get('medium', 0):,} "
            f"| {stats.get('low', 0):,} "
            f"| {stats.get('high_pct', 0):.1f}% "
            f"| {stats.get('avg_propensity', 0):.3f} |\n"
        )
        table_key = f"voter_profiles/{county}"
        links_md += f"- [{county}]({base_url}) — table key `{table_key}`\n"

    return header + divider + rows_md + links_md


def build_report(
    sweep_id: str | None = None,
    entity: str | None = None,
    project: str | None = None,
    output_path: Path = Path("outputs/report_url.txt"),
) -> str:
    """Assemble a W&B Report with campaign-ready analytics sections.

    Sections produced:
      1. Executive Summary
      2. Best Run Comparison (top 5 runs from project)
      3. Demographic Receptivity Analysis — automated county breakdown from
         run.summary['county_stats'] logged by build_county_tables()
      4. Precision vs. Recall Tradeoff
      5. Model Lineage note
      6. Recommendations (from SHAP feature importance)

    Args:
        sweep_id: Optional sweep ID to filter the comparison table.
        entity: W&B entity (team). Falls back to WANDB_ENTITY env var.
        project: W&B project name. Falls back to WANDB_PROJECT env var.
        output_path: File path to write the report share URL.

    Returns:
        URL of the created W&B Report.
    """
    try:
        import wandb.apis.reports as wr
    except ImportError as exc:
        raise ImportError(
            "wandb.apis.reports not available. Ensure wandb>=0.13 is installed."
        ) from exc

    entity = entity or os.environ.get("WANDB_ENTITY")
    project = project or os.environ.get("WANDB_PROJECT", "voter-turnout-2026")

    if not entity:
        raise ValueError("WANDB_ENTITY must be set via env var or --entity flag")

    api = wandb.Api()
    runs = list(api.runs(f"{entity}/{project}", order="-summary_metrics.val/pr_auc"))
    top5 = runs[:5]

    best_pr = top5[0].summary.get("val/pr_auc", 0.0) if top5 else 0.0
    best_recall = top5[0].summary.get("val/recall", 0.0) if top5 else 0.0
    best_precision = top5[0].summary.get("val/precision", 0.0) if top5 else 0.0
    best_run_id = top5[0].id if top5 else None

    # Fetch per-county stats stored by build_county_tables() in run.summary
    county_stats: dict = {}
    if top5:
        county_stats = top5[0].summary.get("county_stats", {})
        if not county_stats:
            logger.warning(
                "county_stats not found in best run summary. "
                "Call build_county_tables() after training to populate Section 3."
            )

    top5_ids = [r.id for r in top5]
    top5_filter = {"$or": [{"name": rid} for rid in top5_ids]} if top5_ids else {}
    best_filter = {"name": best_run_id} if best_run_id else {}

    report = wr.Report(
        project=project,
        entity=entity,
        title="Voter Propensity Model — Campaign Report",
        description="XGBoost propensity model trained on synthetic voter file. FDE portfolio demo.",
    )

    # ------------------------------------------------------------------
    # Section 1: Executive Summary
    # ------------------------------------------------------------------
    exec_md = (
        f"## 1. Executive Summary\n\n"
        f"The best XGBoost model achieves **{best_pr:.1%} PR-AUC** on the held-out test set, "
        f"with **{best_recall:.1%} recall** and **{best_precision:.1%} precision**.\n\n"
        f"This means ~{best_recall:.0%} of all likely supporters are correctly identified. "
        f"At this precision, campaign resources avoid non-supporters ~{best_precision:.0%} of the time, "
        f"directly improving mailer and canvassing budget efficiency."
    )

    # ------------------------------------------------------------------
    # Section 3: Demographic Receptivity — automated from county_stats
    # ------------------------------------------------------------------
    county_table_md = _county_breakdown_md(county_stats, entity, project, best_run_id or "")

    demographic_md = (
        "## 3. Demographic Receptivity Analysis\n\n"
        "### County-level propensity breakdown\n\n"
        f"{county_table_md}\n\n"
        "### Cross-cutting segments\n\n"
        "- **Age 60–74** consistently shows highest propensity across all counties\n"
        "- **18–29 + IND registration** = highest score uncertainty; treat model output as a "
        "floor, not a ceiling for this segment\n"
        "- **DEM + 4–5 midterm history** = most reliable High-tier block for GOTV investment\n"
        "- **Any party + 2+ address changes** = flag for address verification before mailing — "
        "undeliverable mail wastes budget and suppresses reported contact rates"
    )

    # ------------------------------------------------------------------
    # Section 4: PR Curve
    # ------------------------------------------------------------------
    pr_md = (
        "## 4. Precision vs. Recall Tradeoff\n\n"
        "The PR curve shows the mailer-budget tradeoff at every decision threshold:\n\n"
        "- **Move left** (higher recall): capture more supporters, but mail more non-supporters "
        "(lower precision = wasted spend)\n"
        "- **Move right** (higher precision): save budget, but miss some supporters\n\n"
        "**Recommended operating points:**\n"
        "- Field canvassing: **threshold 0.40** — labor is cheaper than missing a voter\n"
        "- Direct mail: **threshold 0.55–0.65** — printing + postage cost is real\n"
        "- Premium fundraising ask: **threshold 0.70+** — very high precision segment only"
    )

    # ------------------------------------------------------------------
    # Section 5: Model Lineage
    # ------------------------------------------------------------------
    lineage_md = (
        "## 5. Model Lineage\n\n"
        "W&B Artifact lineage links each model version to the exact dataset version that produced it.\n\n"
        "To view:\n"
        "1. Open artifact `xgb-voter-model` in the W&B UI\n"
        "2. Click the **Lineage** tab\n"
        "3. Graph shows: `voterfile-all (dataset)` → training run → `xgb-voter-model (model)`\n\n"
        "Any model can be traced back to the exact voter file snapshot and hyperparameter config "
        "that created it — critical for compliance audits and reproducibility."
    )

    # ------------------------------------------------------------------
    # Section 6: Recommendations
    # ------------------------------------------------------------------
    recommendations_md = (
        "## 6. Recommendations\n\n"
        "Based on SHAP feature importance from the best model:\n\n"
        "- **Midterm turnout history** is the strongest predictor. Prioritize consistent midterm voters "
        "(4–5/5) for phone banking and in-person GOTV events.\n"
        "- **Mail ballot history** (3+ uses) correlates with high accessibility. Offer absentee ballot "
        "assistance to Medium-tier voters in this group.\n"
        "- **Long-term registrants** (10+ years) are reliably more likely to turn out than newly "
        "registered voters of comparable demographics.\n"
        "- **Address instability** (2+ changes in 4 years) is a negative signal. Flag for address "
        "verification before any outreach — returned mail wastes budget.\n"
        "- **Young voters (18–29)** carry high uncertainty. Pair model scores with peer outreach "
        "programs rather than direct mail alone.\n\n"
        "### Risk Tier Targeting Strategy\n\n"
        "| Tier | Score range | Recommended tactic |\n"
        "|------|-------------|--------------------|\n"
        "| High | ≥ 0.65 | Phone bank + GOTV event invitations |\n"
        "| Medium | 0.35 – 0.65 | Direct mail + digital retargeting |\n"
        "| Low | < 0.35 | No action (protect budget for High/Medium) |\n"
    )

    report.blocks = [
        wr.TableOfContents(),
        wr.PanelGrid(panels=[wr.MarkdownPanel(markdown=exec_md)]),
        wr.HorizontalRule(),
        wr.PanelGrid(
            runsets=[wr.Runset(project=project, entity=entity, filters=top5_filter)],
            panels=[wr.RunComparer(diff_only="split", layout={"w": 24, "h": 9})],
        ),
        wr.HorizontalRule(),
        wr.PanelGrid(panels=[wr.MarkdownPanel(markdown=demographic_md)]),
        wr.HorizontalRule(),
        wr.PanelGrid(
            runsets=[wr.Runset(project=project, entity=entity, filters=best_filter)],
            panels=[
                wr.LinePlot(
                    x="recall",
                    y=["precision"],
                    title="Precision-Recall Curve",
                    title_x="Recall (supporters captured)",
                    title_y="Precision (mailer efficiency)",
                ),
                wr.MarkdownPanel(markdown=pr_md),
            ],
        ),
        wr.HorizontalRule(),
        wr.PanelGrid(panels=[wr.MarkdownPanel(markdown=lineage_md)]),
        wr.HorizontalRule(),
        wr.PanelGrid(panels=[wr.MarkdownPanel(markdown=recommendations_md)]),
    ]

    report.save()
    url = report.url
    logger.info("Report saved: %s", url)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(url)
    logger.info("Report URL written to %s", output_path)
    return url


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build W&B campaign report")
    parser.add_argument("--sweep-id", default=None)
    parser.add_argument("--entity", default=None)
    parser.add_argument("--project", default=None)
    args = parser.parse_args()

    build_report(sweep_id=args.sweep_id, entity=args.entity, project=args.project)
