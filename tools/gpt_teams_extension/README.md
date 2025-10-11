# GPT Teams Archive Brave Extension

A Brave browser extension that provides a banner interface for exporting ChatGPT Teams conversations directly to AIVA memory.

## Features

- **Banner Interface**: Click the extension icon to open a beautiful banner with export controls
- **Direct Browser Integration**: Works with your existing ChatGPT browser session
- **Real-time Progress**: Live progress updates during export
- **Smart Classification**: Automatically categorizes conversations by scientific discipline
- **AIVA Memory Integration**: Seamlessly imports into AIVA's episodic, timeline, and artifact memory

## Installation

### 1. Install Python Backend

First, make sure the Python exporter is working:

```bash
cd tools/gpt_teams_exporter
pip install -r ../../../requirements.txt
python test_installation.py  # Should pass
```

### 2. Load Extension in Brave

1. Open Brave and go to `brave://extensions/`
2. Enable "Developer mode" (toggle in top-right)
3. Click "Load unpacked"
4. Select the `tools/gpt_teams_extension` folder

The extension icon (🧠) should appear in your toolbar.

### 3. Start the Backend API

```bash
cd tools/gpt_teams_exporter
python main.py --extension-api
```

Keep this terminal running - the extension needs it.

## Usage

1. **Log into ChatGPT**: Open https://chatgpt.com and log in normally
2. **Click Extension Icon**: Click the 🧠 icon in your toolbar
3. **Configure Export**:
   - Set max conversations to export
   - Choose start date
   - Toggle personal conversation inclusion
4. **Start Export**: Click "🚀 Start Export" or "👀 Dry Run"
5. **Monitor Progress**: Watch real-time progress in the banner
6. **View Results**: Check artifacts folder and AIVA memory

## Interface Overview

### Status Banner
- **🟢 Connected**: Backend API is running
- **🔴 Disconnected**: Start the Python API server first

### Export Controls
- **🚀 Start Export**: Begin full export with file saving
- **👀 Dry Run**: Preview what would be exported
- **⏹️ Stop Export**: Cancel running export

### Settings
- **Max Conversations**: Limit export volume (1-1000)
- **Since Date**: Only export conversations after this date
- **Include Personal**: Include non-science conversations

## File Structure

```
tools/gpt_teams_extension/
├── manifest.json          # Extension configuration
├── popup.html            # Banner interface
├── popup.js              # UI logic and API communication
├── content.js            # ChatGPT page integration
├── background.js         # Background service worker
├── icons/                # Extension icons (16x16, 48x48, 128x128)
├── generate_icons.py     # Icon generation script
└── README.md            # This file
```

## Technical Details

### Communication Flow
1. **Extension → Python API**: HTTP requests to localhost:8765
2. **Popup ↔ Background**: Chrome extension messaging
3. **Background ↔ Content**: DOM access for conversation extraction
4. **Content → ChatGPT**: Direct browser session access

### Security
- Localhost-only API communication
- No external data transmission
- Browser session reuse (no credential storage)
- AIVA memory integration respects privacy filters

### Data Processing
- **Raw Extraction**: JSON API calls or DOM scraping
- **Classification**: Keyword-based science discipline detection
- **Filtering**: Exclude personal/trauma content by default
- **Storage**: Chronological file organization
- **Memory Integration**: Episodic, timeline, and artifact updates

## Troubleshooting

### Extension Not Loading
- Check that all files are in the correct structure
- Verify manifest.json syntax
- Try reloading the extension

### Backend Connection Failed
```bash
# Check if API is running
curl http://localhost:8765/status

# Start API server
python tools/gpt_teams_exporter/main.py --extension-api
```

### Export Not Working
- Ensure you're logged into ChatGPT
- Check browser console for errors (F12 → Console)
- Try a dry run first to test connectivity

### No Conversations Found
- Navigate to https://chatgpt.com first
- Check that conversations exist in your account
- Adjust the "Since Date" setting

## Development

### Modifying the Extension
1. Edit files in `tools/gpt_teams_extension/`
2. Go to `brave://extensions/`
3. Click "Reload" on the extension
4. Test changes

### Adding Features
- **UI Changes**: Modify `popup.html` and `popup.js`
- **ChatGPT Integration**: Update `content.js`
- **Backend Communication**: Modify `background.js`
- **API Endpoints**: Update Python `main.py`

### Testing
```bash
# Test backend API
curl -X POST http://localhost:8765/start \
  -H "Content-Type: application/json" \
  -d '{"dry_run": true, "limit": 5}'

# Check progress
curl http://localhost:8765/progress
```

## Privacy & Ethics

- **No Data Collection**: All processing happens locally
- **Session Reuse**: Uses your existing ChatGPT login
- **Content Filtering**: Personal conversations excluded by default
- **Local Storage**: Files saved to your local artifacts folder
- **AIVA Integration**: Memory enhancements for your personal AI

## Requirements

- **Brave Browser**: Latest version recommended
- **Python 3.8+**: With required dependencies
- **ChatGPT Account**: Teams access preferred
- **AIVA System**: For memory integration (optional)

---

**Built for the Wallace Research Suite** 🧠⚡
