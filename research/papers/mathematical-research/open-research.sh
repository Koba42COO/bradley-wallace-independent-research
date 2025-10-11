#!/bin/bash

# Open Wallace Transform Ratio Analysis Research Framework
# Accessible from anywhere in your development environment

echo "🧮 Opening Wallace Transform Ratio Analysis Research Framework..."
echo "📊 Framework will be available at: http://localhost:3001/mathematical-research/"
echo ""

# Check if file server is running
if curl -s http://localhost:3001/health > /dev/null 2>&1; then
    echo "✅ File server is running"
else
    echo "❌ File server is not running"
    echo "🔧 Start it with: cd /path/to/vibesdk && node scripts/file-server.js"
    exit 1
fi

# Open in default browser
if command -v open >/dev/null 2>&1; then
    open "http://localhost:3001/mathematical-research/"
elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "http://localhost:3001/mathematical-research/"
elif command -v start >/dev/null 2>&1; then
    start "http://localhost:3001/mathematical-research/"
else
    echo "🌐 Open this URL in your browser:"
    echo "http://localhost:3001/mathematical-research/"
fi

echo ""
echo "📈 Research Framework Features:"
echo "   • Bradley's Formula Test: g_n = W_φ(p_n) · φ^k"
echo "   • Log-Space Frequency Matching"
echo "   • Spectral Peak Detection (FFT)"
echo "   • Comparative Analysis with Published Results"
echo "   • Interactive Data Visualizations"
echo ""
echo "🎯 Click 'Begin Analysis' to start computation"
