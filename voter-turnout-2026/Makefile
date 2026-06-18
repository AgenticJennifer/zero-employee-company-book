.PHONY: install generate-data train train-fast sweep analyze-sweep explain report test server-up server-down

install:
	pip install -r requirements.txt

generate-data:
	python data/synthetic/generate.py

# Full training run: builds voter profile + county tables in the same W&B run
train:
	python -m src.train

# Quick iteration: skip SHAP table building
train-fast:
	python -m src.train --no-tables

sweep:
	wandb sweep config/sweep.yaml

analyze-sweep:
	@test -n "$(SWEEP_ID)" || (echo "Usage: make analyze-sweep SWEEP_ID=<wandb-sweep-id>" && exit 1)
	python -m src.analyze_sweep $(SWEEP_ID)

# Retroactively build/rebuild county tables for an existing run
explain:
	@test -n "$(RUN_ID)" || (echo "Usage: make explain RUN_ID=<wandb-run-id>" && exit 1)
	python -m src.explain --run-id $(RUN_ID)

# Assemble W&B Report and write URL to outputs/report_url.txt
report:
	python -m src.report

test:
	pytest tests/ -v

server-up:
	docker-compose -f infra/docker-compose.yml up -d
	@echo "W&B Server running at http://localhost:8080"
	@echo "Set WANDB_BASE_URL=http://localhost:8080 in .env to use local server"

server-down:
	docker-compose -f infra/docker-compose.yml down
