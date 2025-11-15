# Dev Folder Backup: dev_folder_backup_20251115_155936

**Created:** 2025-11-15 16:01:20
**Size:** 1.3G
**Chunked:** true

## Restoration Instructions

### If Chunked:
```bash
# Combine chunks
cat dev_folder_backup_20251115_155936.tar.gz.part* > dev_folder_backup_20251115_155936.tar.gz

# Extract
tar -xzf dev_folder_backup_20251115_155936.tar.gz
```

### If Single File:
```bash
tar -xzf dev_folder_backup_20251115_155936.tar.gz
```

## Excluded Files
- `__pycache__/`
- `.git/`
- `node_modules/`
- `build/`, `dist/`
- `*.pyc`
- `.env` files
- Log files
- Database files
