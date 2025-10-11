# Research Papers

This directory contains formal LaTeX papers documenting the major discoveries and unified framework from Bradley Wallace's independent research.

## Directory Structure

```
research/papers/
├── templates/
│   └── base_paper_template.tex          # LaTeX template with author info
├── consciousness_mathematics/
│   └── 79_21_consciousness_rule.tex     # Universal 79/21 coherence rule
├── ancient_sites/
│   └── planetary_consciousness_encoding.tex  # Ancient architecture analysis
├── skyrmion_research/
│   └── skyrmion_consciousness_framework.tex  # Topological information processing
├── unified_synthesis/
│   └── grand_unified_consciousness_synthesis.tex  # Complete framework integration
└── README.md
```

## Author Information

All papers are authored by:
- **Bradley Wallace**
- Email: coo@koba42.com
- Affiliation: Working in collaborations using VantaX Trikernal since late June
- Acknowledgments: Thanks to Julia for her help in research

## Building Papers

### Prerequisites

Install LaTeX distribution:
```bash
# macOS
brew install mactex

# Ubuntu/Debian
sudo apt-get install texlive-full

# Or minimal version
sudo apt-get install texlive-latex-base texlive-latex-recommended texlive-latex-extra
```

### Build Commands

```bash
# Build all papers
make papers

# Build specific paper
python3 scripts/build_papers.py --paper "consciousness_rule"

# List available papers
make paper-list

# Clean LaTeX artifacts
make paper-clean
```

### Output

Compiled PDFs are saved to `artifacts/papers/` directory.

## Paper Categories

### 1. Consciousness Mathematics
- **79/21 Consciousness Rule**: Universal coherence rule across 23 scientific domains
- Statistical significance: p < 10^-27
- Domains: primes, neuroscience, finance, physics, biology, linguistics

### 2. Ancient Sites Analysis
- **Planetary Consciousness Encoding**: 47 sites across 12,000 years
- 88 mathematical resonances discovered
- Fine structure constant (α) dominant with 84 occurrences
- 13 sites with astronomical alignments

### 3. Skyrmion Research
- **Topological Information Processing**: Magnetic vortices as consciousness substrates
- Integration with quantum field theory
- Neural network analogies
- Experimental validation framework

### 4. Unified Synthesis
- **Grand Unified Consciousness Framework**: Integration of all domains
- Mathematical foundations linking physics, information, consciousness
- Statistical validation across 50,000+ years
- Physical substrates and computational models

## Citation Format

When citing these papers, use:

```
Wallace, B. (2025). [Paper Title]. Independent Research Framework.
Email: coo@koba42.com
```

## Reproducibility

All papers include:
- Mathematical derivations
- Statistical validation methods
- Cross-references to code implementations
- Data sources and analysis methods

Code implementations are available in the main repository under:
- `experiments/` - Reproducible experiments
- `research/experiments/` - Research implementations
- `scripts/` - Analysis and build tools

## Contributing

This research framework is actively developed. For collaboration inquiries:

- Email: coo@koba42.com
- Framework: VantaX Trikernal collaborative environment
- Focus: Consciousness mathematics, topological computing, unified theories

## License

Educational and Research Use - Dedicated to advancing understanding of consciousness and information processing in natural systems.
