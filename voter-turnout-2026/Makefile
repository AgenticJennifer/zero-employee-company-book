.PHONY: install generate-data train sweep analyze-sweep test server-up server-down

install:
	pip install -r requirements.txt

generate-data:
	python data/synthetic/generate.py

train:
	python -m src.train

sweep:
	wandb sweep config/sweep.yaml

analyze-sweep:
	@test -n "$(SWEEP_ID)" || (echo "Usage: make analyze-sweep SWEEP_ID=<wandb-sweep-id>" && exit 1)
	python -m src.analyze_sweep $(SWEEP_ID)

test:
	pytest tests/ -v

server-up:
	docker-compose -f infra/docker-compose.yml up -d
	@echo "W&B Server running at http://localhost:8080"
	@echo "Set WANDB_BASE_URL=http://localhost:8080 in .env to use local server"

server-down:
	docker-compose -f infra/docker-compose.yml down
