#!/bin/bash
# Compile LaTeX paper to PDF

echo "📄 Compiling LaTeX paper to PDF..."
echo "===================================="

cd "$(dirname "$0")"

PAPER="crypto_market_analyzer_pell_cycles.tex"

if [ ! -f "$PAPER" ]; then
    echo "❌ Error: $PAPER not found"
    exit 1
fi

# Check for pdflatex
if ! command -v pdflatex &> /dev/null; then
    echo "⚠️  pdflatex not found. Install MacTeX or use Overleaf.com"
    echo ""
    echo "To compile manually:"
    echo "  1. Go to https://www.overleaf.com"
    echo "  2. Upload $PAPER"
    echo "  3. Click 'Recompile'"
    echo "  4. Download PDF"
    echo ""
    echo "Or install MacTeX:"
    echo "  brew install --cask mactex"
    exit 0
fi

echo "Compiling $PAPER..."
pdflatex -interaction=nonstopmode "$PAPER" > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "✅ First pass complete"
    # Run again for references
    pdflatex -interaction=nonstopmode "$PAPER" > /dev/null 2>&1
    echo "✅ Second pass complete"
    
    PDF="${PAPER%.tex}.pdf"
    if [ -f "$PDF" ]; then
        echo "✅ PDF created: $PDF"
        ls -lh "$PDF"
    else
        echo "❌ PDF not created"
    fi
else
    echo "❌ Compilation failed. Check for LaTeX errors."
    echo "Run: pdflatex $PAPER (without redirect) to see errors"
fi

