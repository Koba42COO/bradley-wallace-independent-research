#!/bin/bash
# Push Consciousness Mathematics Research to GitHub Independent Research Repository

echo "🚀 Pushing Consciousness Mathematics Research to GitHub..."
echo ""

# Set git user configuration (if not already set)
if [ -z "$(git config user.name)" ]; then
    echo "Setting git user configuration..."
    git config user.name "Bradley Wallace"
    git config user.email "bradley.wallace@universalprimegraph.com"
fi

# Repository URL - Update this with your actual GitHub repository URL
REPO_URL="https://github.com/bradley-wallace/bradley-wallace-independent-research.git"

# Check if remote already exists
if git remote | grep -q "origin"; then
    echo "Remote 'origin' already exists. Updating URL..."
    git remote set-url origin "$REPO_URL"
else
    echo "Adding remote 'origin'..."
    git remote add origin "$REPO_URL"
fi

# Verify remote
echo ""
echo "Remote configuration:"
git remote -v
echo ""

# Check if we're on main branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "main" ]; then
    echo "Current branch: $CURRENT_BRANCH"
    echo "Switching to main branch..."
    git checkout -b main 2>/dev/null || git checkout main
fi

# Push to GitHub
echo ""
echo "Pushing to GitHub..."
echo "Repository: $REPO_URL"
echo "Branch: main"
echo ""

# Try to push
if git push -u origin main; then
    echo ""
    echo "✅ SUCCESS! Repository pushed to GitHub"
    echo ""
    echo "Repository URL: $REPO_URL"
    echo ""
    echo "Next steps:"
    echo "1. Visit your repository on GitHub"
    echo "2. Add repository description and topics"
    echo "3. Enable Issues and Wikis"
    echo "4. Share with the scientific community!"
else
    echo ""
    echo "❌ Push failed. Common issues:"
    echo "1. Repository doesn't exist yet - create it on GitHub first"
    echo "2. Authentication required - use GitHub CLI or SSH keys"
    echo "3. Wrong repository URL - update REPO_URL in this script"
    echo ""
    echo "To create repository on GitHub:"
    echo "1. Go to https://github.com/new"
    echo "2. Repository name: bradley-wallace-independent-research"
    echo "3. Description: Universal Prime Graph Protocol φ.1 - Consciousness Mathematics Research"
    echo "4. Set to PUBLIC"
    echo "5. DO NOT initialize with README/gitignore/license"
    echo "6. Click 'Create repository'"
    echo ""
    echo "Then run this script again."
fi
