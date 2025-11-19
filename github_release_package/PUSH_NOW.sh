#!/bin/bash
# Push Consciousness Mathematics Research to GitHub

echo "🚀 Pushing to bradley-wallace-independent-research..."
echo ""

cd "$(dirname "$0")"

# Ensure we're on main branch
git checkout main 2>/dev/null || git checkout -b main

# Try to push
echo "Attempting to push to GitHub..."
echo "Repository: https://github.com/bradley-wallace/bradley-wallace-independent-research.git"
echo ""

if git push -u origin main 2>&1; then
    echo ""
    echo "✅ SUCCESS! Repository pushed to GitHub"
    echo "View at: https://github.com/bradley-wallace/bradley-wallace-independent-research"
else
    echo ""
    echo "⚠️  Push failed. Common solutions:"
    echo ""
    echo "1. REPOSITORY DOESN'T EXIST YET:"
    echo "   → Go to: https://github.com/new"
    echo "   → Repository name: bradley-wallace-independent-research"
    echo "   → Description: Universal Prime Graph Protocol φ.1 - Consciousness Mathematics Research"
    echo "   → Set to PUBLIC"
    echo "   → DO NOT initialize with README/gitignore/license"
    echo "   → Click 'Create repository'"
    echo "   → Then run this script again"
    echo ""
    echo "2. AUTHENTICATION REQUIRED:"
    echo "   → You'll be prompted for username and password"
    echo "   → Use GitHub Personal Access Token as password"
    echo "   → Get token at: https://github.com/settings/tokens"
    echo "   → Token needs 'repo' permissions"
    echo ""
    echo "3. DIFFERENT REPOSITORY NAME:"
    echo "   → Update remote: git remote set-url origin https://github.com/YOUR_USERNAME/YOUR_REPO.git"
    echo ""
fi
