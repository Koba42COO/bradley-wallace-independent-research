#!/bin/bash

# WebRTC Self-Hosted P2P Server Startup Script
# Starts the complete TangTalk-style peer-to-peer communication system

echo "🔗 Starting WebRTC Self-Hosted P2P System"
echo "=========================================="

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+"
    exit 1
fi

# Check if required packages are installed
echo "📦 Checking dependencies..."
python3 -c "
try:
    import fastapi, uvicorn, websockets
    print('✅ FastAPI, Uvicorn, WebSockets OK')
except ImportError as e:
    print(f'❌ Missing dependency: {e}')
    print('Run: pip install fastapi uvicorn websockets')
    exit(1)
"

# Check if aiortc is available (optional for full functionality)
python3 -c "
try:
    import aiortc
    print('✅ aiortc available (full WebRTC support)')
except ImportError:
    print('⚠️  aiortc not available (limited functionality)')
    print('Install with: pip install aiortc')
"

echo ""
echo "🚀 Starting servers..."
echo "STUN Server: localhost:3478"
echo "TURN Server: localhost:3479"
echo "Signaling Server: ws://localhost:8080"
echo ""
echo "Press Ctrl+C to stop all servers"
echo ""

# Start the unified P2P server
python3 webrtc_p2p_server.py
