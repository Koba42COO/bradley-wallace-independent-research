#!/bin/bash
# Script to push crypto market analyzer to independent research repository

echo "🚀 Pushing Crypto Market Analyzer to Independent Research Repository"
echo "===================================================================="
echo ""

# Check if we're in the right directory
if [ ! -f "crypto_analyzer_complete.py" ]; then
    echo "❌ Error: crypto_analyzer_complete.py not found"
    echo "   Please run this script from the project root directory"
    exit 1
fi

# Check if bradley-research remote exists
if ! git remote | grep -q "bradley-research"; then
    echo "📡 Adding bradley-research remote..."
    git remote add bradley-research https://github.com/Koba42COO/bradley-wallace-independent-research.git
fi

# Show current branch
CURRENT_BRANCH=$(git branch --show-current)
echo "📍 Current branch: $CURRENT_BRANCH"
echo ""

# Option 1: Push to existing branch
echo "Option 1: Pushing to existing branch..."
git push bradley-research $CURRENT_BRANCH

# Option 2: Create dedicated branch for crypto analyzer
echo ""
echo "Option 2: Creating dedicated branch 'crypto-market-analyzer'..."
git checkout -b crypto-market-analyzer 2>/dev/null || git checkout crypto-market-analyzer
git push -u bradley-research crypto-market-analyzer

echo ""
echo "✅ Push complete!"
echo ""
echo "📋 Repository URLs:"
echo "   Main: https://github.com/Koba42COO/bradley-wallace-independent-research"
echo "   Branch: https://github.com/Koba42COO/bradley-wallace-independent-research/tree/crypto-market-analyzer"
echo ""
echo "💡 Next steps:"
echo "   1. Create a pull request to merge into main"
echo "   2. Or create a new release/tag for this research"
echo "   3. Add repository description and topics"

