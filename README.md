# voter-propensity-pipeline

XGBoost voter propensity model with full W&B experiment tracking, artifact versioning, hyperparameter sweeps, and campaign-ready reporting. Built as an AI Forward Deployed Engineer (FDE) portfolio project.

## Stack

Python · XGBoost · Weights & Biases · SHAP · Docker · Terraform (AWS)

## Project

All code lives in `voter-turnout-2026/` on the `claude/voter-propensity-wb-pipeline-92m8fh` branch.

## Quickstart

```bash
cd voter-turnout-2026
cp .env.example .env        # add WANDB_API_KEY + WANDB_ENTITY
make install
make generate-data
make train                  # full run: metrics + SHAP tables + artifacts
make sweep                  # Bayesian hyperparameter search
make report                 # assemble W&B Report → outputs/report_url.txt
make test
```

## Data Handling Policy

Voter files contain PII. `data/raw/` is gitignored and must never be committed.
Check your state's voter data usage statutes before production use.
