# LaTeX Compilation Instructions

## Quick Start

```bash
cd papers
pdflatex STEGANOGRAPHY_NLP_SUBLINGUAL_PRIMING_COMPLETE.tex
pdflatex STEGANOGRAPHY_NLP_SUBLINGUAL_PRIMING_COMPLETE.tex
```

## Complete Compilation

```bash
# Navigate to papers directory
cd papers

# First pass (generates .aux file)
pdflatex STEGANOGRAPHY_NLP_SUBLINGUAL_PRIMING_COMPLETE.tex

# Second pass (resolves references)
pdflatex STEGANOGRAPHY_NLP_SUBLINGUAL_PRIMING_COMPLETE.tex

# Output: STEGANOGRAPHY_NLP_SUBLINGUAL_PRIMING_COMPLETE.pdf
```

## Required LaTeX Packages

The document requires these packages (usually included in full TeX distributions):

- `amsmath`, `amssymb`, `amsthm` - Mathematics
- `graphicx` - Graphics
- `hyperref` - Hyperlinks
- `listings` - Code listings
- `xcolor` - Colors
- `geometry` - Page layout
- `fancyhdr` - Headers/footers
- `booktabs` - Professional tables
- `algorithm`, `algpseudocode` - Algorithms
- `natbib` - Bibliography (optional)

## Installation

### macOS

```bash
# Install MacTeX (full distribution)
brew install --cask mactex

# Or install BasicTeX (smaller)
brew install --cask basictex
```

### Linux

```bash
# Ubuntu/Debian
sudo apt-get install texlive-full

# Or minimal installation
sudo apt-get install texlive-latex-base texlive-latex-extra
```

### Windows

Download and install MiKTeX or TeX Live from:
- https://miktex.org/
- https://www.tug.org/texlive/

## Troubleshooting

### Missing Packages

If compilation fails with missing package errors:

```bash
# MiKTeX (Windows/Linux)
miktex install <package-name>

# TeX Live
tlmgr install <package-name>
```

### Code Listings Issues

If code listings cause issues, the document references external files. Ensure:
1. Source files exist in `../src/` relative to papers directory
2. Or comment out `\lstinputlisting` commands and use inline code blocks

### Large File Size

The document may be large due to code listings. Options:
1. Use `\lstinputlisting` with `firstline` and `lastline` to include only key sections
2. Move code to separate appendix files
3. Reference code files instead of including them

## Alternative: Generate PDF from Markdown

If LaTeX is not available, you can use:

```bash
# Using pandoc
pandoc STEGANOGRAPHY_NLP_SUBLINGUAL_PRIMING_COMPLETE.tex -o output.pdf

# Or convert markdown documentation
pandoc docs/STEGANOGRAPHY_NLP_SUBLINGUAL_PRIMING_INVESTIGATION.md -o paper.pdf
```

## Verification

After compilation, verify:
- [ ] PDF generated successfully
- [ ] All sections present
- [ ] Figures/tables render correctly
- [ ] References resolve
- [ ] Code listings display properly
- [ ] Page numbers correct

---

**Status:** ✅ LaTeX Document Complete  
**Framework:** Universal Prime Graph Protocol φ.1

