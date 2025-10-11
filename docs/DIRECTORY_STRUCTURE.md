# Directory Structure

This document outlines the standardized directory structure for the Bradley Wallace Research Framework.

## Root Level Organization

```
/
├── research/           # Core research work and papers
├── experiments/        # Runnable experiments and demos
├── data/              # Input datasets and raw data
├── artifacts/         # Generated outputs and results
├── figures/           # Visualizations and plots
├── services/          # Service implementations
├── tools/             # Development and utility tools
├── docs/              # Documentation
├── scripts/           # Build and automation scripts
└── configs/           # Configuration files
```

## Research Directory (`research/`)

```
research/
├── papers/                    # Formal research papers
│   ├── templates/            # LaTeX templates
│   ├── consciousness_mathematics/
│   ├── ancient_sites/
│   ├── skyrmion_research/
│   └── unified_synthesis/
├── experiments/              # Research experiments
│   ├── cosmic_spirals_research/
│   └── [other-research-experiments]/
└── notebooks/                # Jupyter notebooks
```

## Experiments Directory (`experiments/`)

```
experiments/
├── [experiment_name].py      # Main experiment scripts
├── projects/                 # Multi-file experiment projects
├── benchmarks/               # Performance benchmarking
└── demos/                    # Demonstration scripts
```

## Data Directory (`data/`)

```
data/
├── raw/                      # Original, immutable data
├── interim/                  # Cleaned, transformed data
├── processed/                # Final datasets for modeling
└── external/                 # Third-party datasets
```

## Artifacts Directory (`artifacts/`)

```
artifacts/
├── papers/                   # Compiled PDF papers
├── figures/                  # Generated plots and visualizations
├── models/                   # Trained model artifacts
├── reports/                  # Analysis reports and summaries
├── run-data/                 # Runtime data and logs
└── [experiment-specific]/    # Experiment outputs
```

## Figures Directory (`figures/`)

```
figures/
├── [domain_name]/            # Domain-specific visualizations
│   ├── [figure_name].png
│   └── [figure_name].pdf
├── archangels/               # Religious/art figures
└── [other-domains]/
```

## Services Directory (`services/`)

```
services/
├── [service_name]/           # Service implementation
│   ├── scripts/
│   ├── config/
│   └── src/
├── api/                      # REST API services
├── monitoring/               # Monitoring and logging
└── [other-services]/
```

## Tools Directory (`tools/`)

```
tools/
├── [tool_name]/              # Development tools
│   ├── scripts/
│   ├── config/
│   └── docs/
├── aiva-ide/                 # IDE integrations
├── gpt_teams_exporter/       # Export utilities
└── [other-tools]/
```

## File Naming Conventions

### Scripts and Code
- `snake_case.py` for Python files
- `kebab-case.js` for JavaScript files
- `CamelCase.java` for Java files

### Data Files
- `{dataset}_{version}_{date}.{ext}`
- Example: `prime_gaps_v1_20251011.csv`

### Figures
- `{experiment}_{figure}_{date}.{ext}`
- Example: `wallace_analysis_20251002.png`

### Papers
- `{discovery}_{date}.tex`
- Example: `79_21_consciousness_rule.tex`

## Git LFS Tracking

Large files are automatically tracked via Git LFS:

- **Images**: `*.png`, `*.jpg`, `*.jpeg`, `*.gif`, `*.webp`
- **Data**: `*.npz`, `*.npy`, `*.db`, `*.csv`, `*.parquet`
- **Models**: `*.pth`, `*.onnx`, `*.h5`
- **Archives**: `*.zip`, `*.tar`, `*.gz`
- **Documents**: `*.pdf`

## Directory Creation Guidelines

1. **Research First**: New research goes in `research/`
2. **Experiment Validation**: Code that validates claims goes in `experiments/`
3. **Data Preservation**: Raw data in `data/raw/`, processed in `data/processed/`
4. **Output Organization**: Generated files in appropriate `artifacts/` subdirectories
5. **Tool Separation**: Development tools in `tools/`, production services in `services/`

## Maintenance

- Regular cleanup of temporary files
- Archive old experiments to avoid clutter
- Update this document when structure changes
- Ensure new directories follow these conventions
