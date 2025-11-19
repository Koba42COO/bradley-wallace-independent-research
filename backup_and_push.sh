#!/bin/bash

# Full Dev Folder Backup, Chunking, and Git Push Script
# This script creates a compressed backup, chunks it, and pushes to the backup repo

set -e  # Exit on error

BACKUP_DIR="/Users/coo-koba42/dev"
BACKUP_NAME="dev_folder_backup_$(date +%Y%m%d_%H%M%S)"
TEMP_DIR="/tmp/${BACKUP_NAME}"
CHUNK_SIZE=50M  # 50MB chunks (GitHub has 100MB file limit)
BACKUP_REPO="backup"
BACKUP_BRANCH="backups"

echo "=========================================="
echo "Dev Folder Backup & Push Script"
echo "=========================================="
echo ""

# Create temp directory
mkdir -p "${TEMP_DIR}"
cd "${BACKUP_DIR}"

echo "Step 1: Creating compressed backup..."
echo "Excluding: __pycache__, .git, node_modules, build, dist, .env files, etc."

# Create tar archive excluding common unnecessary files
tar --exclude='__pycache__' \
    --exclude='.git' \
    --exclude='node_modules' \
    --exclude='build' \
    --exclude='dist' \
    --exclude='*.pyc' \
    --exclude='.DS_Store' \
    --exclude='.env' \
    --exclude='*.log' \
    --exclude='logs' \
    --exclude='*.db' \
    --exclude='*.sqlite' \
    --exclude='*.csv' \
    --exclude='.benchmark_cache' \
    --exclude='.aiva_storage' \
    -czf "${TEMP_DIR}/${BACKUP_NAME}.tar.gz" \
    -C "${BACKUP_DIR}" .

echo "✓ Backup created: ${TEMP_DIR}/${BACKUP_NAME}.tar.gz"
BACKUP_SIZE=$(du -h "${TEMP_DIR}/${BACKUP_NAME}.tar.gz" | cut -f1)
echo "  Size: ${BACKUP_SIZE}"
echo ""

# Check if chunking is needed
BACKUP_SIZE_BYTES=$(stat -f%z "${TEMP_DIR}/${BACKUP_NAME}.tar.gz" 2>/dev/null || stat -c%s "${TEMP_DIR}/${BACKUP_NAME}.tar.gz" 2>/dev/null)
CHUNK_SIZE_BYTES=$((50 * 1024 * 1024))  # 50MB in bytes

if [ "${BACKUP_SIZE_BYTES}" -gt "${CHUNK_SIZE_BYTES}" ]; then
    echo "Step 2: Chunking backup (size exceeds 50MB)..."
    cd "${TEMP_DIR}"
    split -b "${CHUNK_SIZE}" -d -a 3 "${BACKUP_NAME}.tar.gz" "${BACKUP_NAME}.tar.gz.part"
    
    # Create manifest file
    echo "${BACKUP_NAME}" > "${BACKUP_NAME}.manifest"
    echo "$(date +%Y-%m-%d\ %H:%M:%S)" >> "${BACKUP_NAME}.manifest"
    echo "${BACKUP_SIZE_BYTES}" >> "${BACKUP_NAME}.manifest"
    ls -1 "${BACKUP_NAME}.tar.gz.part"* | wc -l | tr -d ' ' >> "${BACKUP_NAME}.manifest"
    
    CHUNK_COUNT=$(ls -1 "${BACKUP_NAME}.tar.gz.part"* | wc -l | tr -d ' ')
    echo "✓ Backup chunked into ${CHUNK_COUNT} parts"
    echo ""
    
    CHUNKED=true
else
    echo "Step 2: Backup is small enough, no chunking needed"
    echo ""
    CHUNKED=false
fi

echo "Step 3: Preparing git repository..."
cd "${BACKUP_DIR}"

# Ensure we're on the backup branch or create it
git fetch "${BACKUP_REPO}" 2>/dev/null || true
if git show-ref --verify --quiet refs/heads/"${BACKUP_BRANCH}"; then
    git checkout "${BACKUP_BRANCH}"
    git pull "${BACKUP_REPO}" "${BACKUP_BRANCH}" 2>/dev/null || true
else
    git checkout -b "${BACKUP_BRANCH}" 2>/dev/null || git checkout "${BACKUP_BRANCH}"
fi

# Create backup directory in repo
mkdir -p "backups/${BACKUP_NAME}"

echo "Step 4: Copying backup files to repository..."
if [ "$CHUNKED" = true ]; then
    cp "${TEMP_DIR}/${BACKUP_NAME}.tar.gz.part"* "backups/${BACKUP_NAME}/"
    cp "${TEMP_DIR}/${BACKUP_NAME}.manifest" "backups/${BACKUP_NAME}/"
    echo "✓ Copied ${CHUNK_COUNT} chunks and manifest"
else
    cp "${TEMP_DIR}/${BACKUP_NAME}.tar.gz" "backups/${BACKUP_NAME}/"
    echo "✓ Copied single backup file"
fi

# Create README for this backup
cat > "backups/${BACKUP_NAME}/README.md" << EOF
# Dev Folder Backup: ${BACKUP_NAME}

**Created:** $(date +"%Y-%m-%d %H:%M:%S")
**Size:** ${BACKUP_SIZE}
**Chunked:** ${CHUNKED}

## Restoration Instructions

### If Chunked:
\`\`\`bash
# Combine chunks
cat ${BACKUP_NAME}.tar.gz.part* > ${BACKUP_NAME}.tar.gz

# Extract
tar -xzf ${BACKUP_NAME}.tar.gz
\`\`\`

### If Single File:
\`\`\`bash
tar -xzf ${BACKUP_NAME}.tar.gz
\`\`\`

## Excluded Files
- \`__pycache__/\`
- \`.git/\`
- \`node_modules/\`
- \`build/\`, \`dist/\`
- \`*.pyc\`
- \`.env\` files
- Log files
- Database files
EOF

echo "Step 5: Committing and pushing to ${BACKUP_REPO}..."
git add "backups/${BACKUP_NAME}/"
git commit -m "Add backup: ${BACKUP_NAME} (${BACKUP_SIZE})" || echo "No changes to commit"

echo ""
echo "Pushing to ${BACKUP_REPO}/${BACKUP_BRANCH}..."
git push "${BACKUP_REPO}" "${BACKUP_BRANCH}"

echo ""
echo "=========================================="
echo "✓ Backup complete and pushed!"
echo "=========================================="
echo "Backup location: backups/${BACKUP_NAME}/"
echo "Remote: ${BACKUP_REPO}/${BACKUP_BRANCH}"
echo ""
echo "Cleaning up temp files..."
rm -rf "${TEMP_DIR}"
echo "✓ Done!"

