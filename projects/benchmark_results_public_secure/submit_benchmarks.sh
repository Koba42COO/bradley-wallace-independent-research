#!/bin/bash
# AIVA Benchmark Results - Automated Submission Helper

echo "🧠 AIVA Benchmark Results Submission"
echo "===================================="
echo ""

# Check if files exist
if [ ! -f "papers_with_code_secure.json" ]; then
    echo "⚠️  papers_with_code_secure.json not found"
    exit 1
fi

if [ ! -f "huggingface_leaderboard_secure.json" ]; then
    echo "⚠️  huggingface_leaderboard_secure.json not found"
    exit 1
fi

echo "✅ All submission files found"
echo ""
echo "📋 Next steps:"
echo "1. Papers with Code:"
echo "   - Visit: https://paperswithcode.com/"
echo "   - Submit: papers_with_code_secure.json"
echo ""
echo "2. HuggingFace:"
echo "   - Visit: https://huggingface.co/spaces"
echo "   - Submit: huggingface_leaderboard_secure.json"
echo ""
echo "3. GitHub:"
echo "   - Create release with: github_release_notes_secure.md"
echo ""
echo "✅ Submission helper complete"
