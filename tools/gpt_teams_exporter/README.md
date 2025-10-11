# GPT Teams Archive Exporter

Exports conversations from ChatGPT Teams and integrates them into AIVA memory with semantic classification.

## Features

- **Browser Automation**: Automated login and conversation scraping using Playwright
- **Semantic Classification**: Automatically categorizes conversations by scientific discipline
- **AIVA Integration**: Seamlessly imports conversations into AIVA's memory system
- **Deduplication**: Tracks processed conversations to avoid duplicates
- **Flexible Export**: Raw JSON + Markdown formats with organized directory structure
- **Web UI**: Optional web interface for easy control and monitoring
- **Dry Run**: Preview what would be exported without making changes

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
playwright install chromium
```

2. Copy environment configuration:
```bash
cp configs/env.example .env
```

3. Edit `.env` with your ChatGPT credentials:
```bash
CHATGPT_EMAIL=your-email@example.com
CHATGPT_PASSWORD=your-password-here
```

## Usage

### Command Line

```bash
# Basic export
python tools/gpt_teams_exporter/main.py

# Dry run (preview only)
python tools/gpt_teams_exporter/main.py --dry-run

# Export with limits
python tools/gpt_teams_exporter/main.py --since 2024-01-01 --limit 100

# Include personal conversations
python tools/gpt_teams_exporter/main.py --include-personal

# Use seed folder instead of scraping
python tools/gpt_teams_exporter/main.py --seed-folder /path/to/chat_gpt_folder
```

### Web UI

```bash
python tools/gpt_teams_exporter/main.py --web-ui
```

Then open http://localhost:8765 in your browser.

## Output Structure

```
artifacts/gpt_convos/
├── index.jsonl                 # Deduplication index
├── raw/                        # Raw exports
│   └── 2024/
│       └── 01/
│           └── 2024-01-15_14-30-00__conversation-title__conv-id.json
├── raw_md/                     # Raw markdown
│   └── 2024/
│       └── 01/
│           └── 2024-01-15_14-30-00__conversation-title__conv-id.md
├── science/                    # Classified by discipline
│   ├── math/
│   ├── physics/
│   ├── ml/
│   ├── cryptography/
│   ├── systems/
│   ├── philosophy_theory/
│   └── application/
└── personal_sanitized/         # Personal conversations (optional)
```

## Classification System

Conversations are automatically classified using keyword matching:

### Science Disciplines
- **math**: Mathematics, algebra, geometry, number theory, cryptography
- **physics**: Physics, quantum mechanics, relativity, field theory
- **ml**: Machine learning, AI, neural networks, deep learning
- **cryptography**: Encryption, security protocols, zero-knowledge proofs
- **systems**: Distributed systems, architecture, infrastructure
- **philosophy_theory**: Consciousness, metaphysics, spirituality, theory
- **application**: Implementation, coding, software development

### Content Filtering
- **Creative/Spiritual**: Goes to `philosophy_theory` (unless private)
- **Personal/Trauma**: Excluded entirely for privacy
- **Default**: Personal conversations can be included with `--include-personal`

## AIVA Memory Integration

Exported conversations are automatically integrated into AIVA's memory:

- **Episodic Memory**: Conversation metadata and file references
- **Timeline**: Chronological events with timestamps
- **Artifacts**: File metadata, checksums, and topic tags
- **Knowledge Graph**: Concept extraction and linking

## Configuration

### Classification Rules

Edit `configs/gpt_scraper_classification.yaml` to customize:

```yaml
# Add new science disciplines
science_disciplines:
  - neuroscience: ["neural", "brain", "cognition", "consciousness"]

# Adjust exclusion keywords
exclude_keywords:
  - "medical"
  - "therapy"

# Modify similarity threshold
similarity_threshold: 0.8
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CHATGPT_EMAIL` | Your ChatGPT account email | Required |
| `CHATGPT_PASSWORD` | Your ChatGPT password | Required |
| `CHATGPT_TEAM_NAME` | Team workspace name | Optional |
| `SCRAPE_HEADFUL` | Run browser visibly | `true` |
| `EXPORT_SINCE` | Export conversations after date | `2024-01-01` |
| `EXPORT_LIMIT` | Maximum conversations | `500` |
| `INCLUDE_PERSONAL_SANITIZED` | Include personal convos | `false` |
| `CLASSIFICATION_CONFIG` | Path to config YAML | `configs/gpt_scraper_classification.yaml` |

## Security & Privacy

- **Credentials**: Stored in `.env` (not committed to git)
- **2FA Support**: Manual completion required for two-factor authentication
- **Content Filtering**: Personal/trauma content automatically excluded
- **Browser Data**: Persistent browser context maintains login state
- **Rate Limiting**: Polite delays between requests to avoid detection

## Troubleshooting

### Login Issues
- Ensure credentials are correct in `.env`
- Complete 2FA manually when prompted
- Clear browser data: `rm -rf artifacts/run-data/browser_data/`

### Classification Issues
- Edit `configs/gpt_scraper_classification.yaml`
- Check logs in `artifacts/run-data/gpt_teams_export.log`
- Use `--dry-run` to preview classifications

### Memory Issues
- Large exports may require increasing system memory
- Use `--limit` to process in smaller batches
- Check AIVA memory files for import status

## Development

### Adding New Features

1. **Classification Rules**: Update `configs/gpt_scraper_classification.yaml`
2. **Export Formats**: Modify `_save_conversation()` in `main.py`
3. **AIVA Integration**: Update `_update_aiva_memory()` method
4. **Browser Automation**: Extend Playwright logic in `_get_conversations()`

### Testing

```bash
# Test classification
python -c "
from tools.gpt_teams_exporter.main import GPTTeamsExporter
exporter = GPTTeamsExporter({})
test_conv = {'mapping': {'node1': {'message': {'content': {'parts': ['quantum physics theory']}}}}}}
print(exporter._classify_conversation(test_conv))
"
```

## License

This tool is part of the Wallace Research Suite. See project LICENSE for details.
