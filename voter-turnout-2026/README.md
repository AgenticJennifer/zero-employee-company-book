# voter-turnout-2026

XGBoost voter propensity model with full W&B experiment tracking, artifact versioning, and hyperparameter sweeps. Built as an AI Forward Deployed Engineer (FDE) portfolio project.

## Quickstart

```bash
# 1. Install dependencies
make install

# 2. Set credentials
cp .env.example .env
# Edit .env — add WANDB_API_KEY and WANDB_ENTITY

# 3. Generate synthetic voter file (50k rows)
make generate-data

# 4. Train baseline model
make train

# 5. Run hyperparameter sweep (20+ trials)
make sweep
# Follow the printed agent command: wandb agent <sweep-id>

# 6. Run tests
make test
```

## Architecture

```
data/synthetic/generate.py   → 50k synthetic voters with realistic distributions
src/ingest.py                → Load + validate voter CSVs
src/features.py              → Feature engineering (10 features)
src/train.py                 → XGBoost training loop + W&B logging
src/artifacts.py             → Dataset + model artifact versioning
src/sweep.py                 → W&B Sweep entrypoint
src/analyze_sweep.py         → Pull sweep results + plain-English summary
src/explain.py               → SHAP explanations + voter profile tables
src/report.py                → W&B Report assembly
```

## Model

- **Algorithm**: XGBoost classifier
- **Primary metric**: PR-AUC (appropriate for imbalanced classes at ~30% positive rate)
- **Targets**: `target_support` (binary) or `target_vote_propensity` (continuous)
- **Features**: turnout history, registration tenure, mail ballot history, demographics, income tract

## W&B Integration

| Feature | How it's used |
|---------|---------------|
| Experiment tracking | Every run logs train/val/test metrics per epoch |
| Artifacts | Dataset and model artifacts with lineage linking |
| Sweeps | Bayesian search over 6 hyperparameters |
| Tables | Voter profile table with SHAP explanations per voter |
| Reports | Campaign-ready report with PR curve and recommendations |

## Local W&B Server

For campaigns that cannot upload voter data to the cloud:

```bash
make server-up       # starts W&B at http://localhost:8080
# Set WANDB_BASE_URL=http://localhost:8080 in .env
make train           # logs to local server
make server-down
```

## Data Handling Policy

**IMPORTANT — read before using real voter files.**

- `data/raw/` is **gitignored and must never be committed**. Voter files contain PII.
- Access to raw voter files must be restricted to named individuals and logged.
- Model outputs (scores, SHAP values) must not contain raw voter PII fields.
- `data/synthetic/` contains only algorithmically generated fake data — safe to commit.
- Before running on a real voter file in production, verify compliance with your state's voter data usage statutes. Many states restrict commercial use, export, or AI-based profiling of voter registration data.
- Recommended: encrypt `data/raw/` at rest (e.g. with `age` or LUKS) and restrict filesystem permissions to the operator user.

## Stack

Python · XGBoost · Weights & Biases · SHAP · Docker · Terraform
