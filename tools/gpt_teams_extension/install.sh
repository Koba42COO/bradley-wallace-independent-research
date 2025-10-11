#!/bin/bash
# GPT Teams Archive Extension Installer for Brave

echo "🧠 GPT Teams Archive Extension Installer"
echo "========================================"

# Check if we're in the right directory
if [ ! -f "manifest.json" ]; then
    echo "❌ Error: Please run this script from the tools/gpt_teams_extension directory"
    exit 1
fi

echo "📁 Extension directory: $(pwd)"

# Check if icons exist
if [ ! -d "icons" ] || [ ! -f "icons/icon48.png" ]; then
    echo "🎨 Generating extension icons..."
    python3 generate_icons.py
fi

echo "✅ Extension files ready"
echo ""
echo "📋 Manual Installation Instructions:"
echo "===================================="
echo ""
echo "1. Open Brave browser"
echo "2. Navigate to: brave://extensions/"
echo "3. Enable 'Developer mode' (top-right toggle)"
echo "4. Click 'Load unpacked'"
echo "5. Select this folder: $(pwd)"
echo "6. The 🧠 extension icon should appear in your toolbar"
echo ""
echo "🚀 Starting Backend API..."
echo "=========================="
echo ""
echo "Keep this terminal open. The extension needs the API running."
echo "Press Ctrl+C to stop the API server."
echo ""
echo "Starting Python API server..."

# Start the Python API server
cd ../gpt_teams_exporter
python3 main.py --extension-api
