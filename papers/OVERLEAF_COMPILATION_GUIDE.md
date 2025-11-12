# 📄 Overleaf Compilation Guide

## Quick Start: Compile Paper to PDF

### Step 1: Create Overleaf Account
1. Go to [https://www.overleaf.com](https://www.overleaf.com)
2. Sign up (free account works fine)
3. Log in

### Step 2: Create New Project
1. Click **"New Project"** → **"Upload Project"**
2. Select `crypto_market_analyzer_pell_cycles.tex` from `papers/` directory
3. Click **"Upload"**

### Step 3: Compile
1. Overleaf will automatically detect it's a LaTeX file
2. Click **"Recompile"** button (top left)
3. Wait for compilation (usually 5-10 seconds)
4. View PDF in the right panel

### Step 4: Download PDF
1. Click **"Download PDF"** button (top menu)
2. Save to your computer
3. PDF is ready for submission!

---

## Troubleshooting

### Missing Packages
If compilation fails with "Package not found":
- Overleaf usually installs packages automatically
- If not, click **"Logs and output files"** → **"Other logs & files"** → **"Compilation log"**
- Look for missing package names
- Add `\usepackage{package_name}` if needed

### Common Issues

**Issue**: "File not found" errors
- **Solution**: Make sure all files are in the same directory or adjust paths

**Issue**: Bibliography errors
- **Solution**: The paper uses manual bibliography, should work as-is

**Issue**: Algorithm package errors
- **Solution**: Overleaf should have `algorithm` and `algorithmic` packages

---

## Alternative: Local Compilation

If you prefer local compilation:

### macOS
```bash
# Install MacTeX (large download ~4GB)
brew install --cask mactex

# Compile
cd papers
pdflatex crypto_market_analyzer_pell_cycles.tex
pdflatex crypto_market_analyzer_pell_cycles.tex  # Run twice for references
```

### Linux
```bash
sudo apt-get install texlive-full
cd papers
pdflatex crypto_market_analyzer_pell_cycles.tex
pdflatex crypto_market_analyzer_pell_cycles.tex
```

---

## Paper Information

- **Title**: Advanced Cryptocurrency Market Analysis Using Pell Sequence Cycles, Tri-Gemini Temporal Inference, and Prime Pattern Detection: Applications to Futures Markets
- **Author**: Bradley Wallace
- **Pages**: ~25-30 pages (estimated)
- **Sections**: 11 main sections + bibliography

---

## Submission Checklist

Before submitting to arXiv:

- [ ] PDF compiles without errors
- [ ] All equations render correctly
- [ ] Tables are properly formatted
- [ ] Algorithms display correctly
- [ ] Bibliography is complete
- [ ] Figures (if any) are included
- [ ] Abstract is 200-300 words
- [ ] Keywords are included

---

**Ready to compile!** Use Overleaf for the easiest experience.

