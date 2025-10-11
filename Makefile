PY=python3

.PHONY: install test api cli docker-build docker-run docs-serve daily papers paper-clean paper-list

install:
	$(PY) -m pip install --upgrade pip
	pip install -r configs/python/requirements.txt
	pip install -e packages/pac_system

test:
	pytest -q

api:
	uvicorn services.api:app --host 0.0.0.0 --port 8080 --reload

cli:
	pacctl run-unified --mode auto

docker-build:
	docker build -t pac-system:latest -f deploy/docker/Dockerfile .

docker-run:
	docker compose -f deploy/docker/docker-compose.dev.yml up --build

docs-serve:
	mkdocs serve -f docs/mkdocs.yml

daily:
	$(PY) scripts/run_daily.py

papers:
	$(PY) scripts/build_papers.py --all

paper-clean:
	$(PY) scripts/build_papers.py --clean

paper-list:
	$(PY) scripts/build_papers.py --list

# Environment and reproducibility targets
.PHONY: env env-update env-export validate-data validate-experiments reproduce-experiment

env:
	@echo "Setting up reproducible environment..."
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install pip-tools
	pip-compile configs/python/requirements.in -o configs/python/requirements.txt
	pip-sync configs/python/requirements.txt

env-update:
	@echo "Updating environment dependencies..."
	pip-compile --upgrade configs/python/requirements.in -o configs/python/requirements.txt
	pip-sync configs/python/requirements.txt

env-export:
	@echo "Exporting current environment..."
	$(PY) -m pip freeze > configs/python/requirements-frozen.txt
	$(PY) --version > configs/python/python-version.txt

validate-data:
	@echo "Validating data integrity..."
	$(PY) scripts/validate_data.py

validate-experiments:
	@echo "Running experiment validation suite..."
	$(PY) -m pytest experiments/ -v --tb=short --durations=10

reproduce-experiment:
	@echo "Reproducing experiment: $(EXP)"
	@if [ -z "$(EXP)" ]; then \
		echo "Usage: make reproduce-experiment EXP=experiment_name"; \
		exit 1; \
	fi
	$(PY) experiments/$(EXP).py --reproduce

# Research workflow targets
.PHONY: research-daily research-weekly research-backup

research-daily:
	@echo "Running daily research tasks..."
	$(PY) scripts/run_daily.py
	make papers
	make validate-data

research-weekly:
	@echo "Running weekly research validation..."
	make validate-experiments
	make env-export
	$(PY) scripts/generate_research_summary.py

research-backup:
	@echo "Creating research backup..."
	tar -czf research_backup_$(shell date +%Y%m%d).tar.gz research/ artifacts/ data/ --exclude='*.log' --exclude='__pycache__'

