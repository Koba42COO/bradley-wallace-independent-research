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

