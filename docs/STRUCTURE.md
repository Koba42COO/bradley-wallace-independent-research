## Repository structure

```
dev/
├── packages/                 # Reusable Python packages (installable)
│   └── pac_system/
│       ├── __init__.py
│       ├── pyproject.toml
│       └── [... core modules ...]
├── services/                 # Apps/microservices
│   ├── backend/
│   ├── primality-api/
│   ├── cudnt-production/
│   └── monitoring/
├── research/                 # Papers, experiments, notebooks
│   ├── papers/
│   ├── experiments/
│   └── notebooks/
├── experiments/              # One-off runnable prototypes/benchmarks
├── scripts/                  # Dev/ops scripts
├── configs/                  # Centralized configs
│   └── python/
├── deploy/                   # Docker, compose, k8s
│   ├── docker/
│   └── k8s/
├── data/                     # Versioned inputs/outputs (gitignored)
│   ├── raw/
│   ├── external/
│   ├── interim/
│   └── processed/
├── artifacts/                # Generated models/reports/images/logs (gitignored)
│   ├── models/
│   ├── reports/
│   ├── figures/
│   └── logs/
├── docs/                     # Docs and guides
├── tests/                    # Unified test tree
└── deploy/docker/*.yml       # Compose files
```

## Common commands

- Install for local dev (editable):
  - `pip install -r configs/python/requirements.txt`
  - `pip install -e packages/pac_system`

- Run API locally:
  - `make api` (uvicorn)

- Docker (dev):
  - `docker compose -f deploy/docker/docker-compose.dev.yml up --build`

- Docker (prod-like):
  - `docker compose -f deploy/docker/docker-compose.prod.yml up --build -d`

## Notes

- Large outputs (csv, db, pkl, logs, images) live under `data/` or `artifacts/` and are gitignored.
- Per-service or per-package deps can stay local; shared Python deps live under `configs/python/`.

