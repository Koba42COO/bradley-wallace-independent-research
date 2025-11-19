#!/bin/bash
# 🛡️ Sentinel Security - Quick Network Scan
# Run this script to scan your network for devices

echo "🛡️  Sentinel Security - Network Device Scan"
echo "============================================"
echo ""
echo "Starting 30-second network scan..."
echo "Press Ctrl+C to stop early"
echo ""

# Run Sentinel Security for 30 seconds
sudo python3 sentinel_security.py -i en0 &
SENTINEL_PID=$!

# Wait 30 seconds
sleep 30

# Stop Sentinel
sudo kill $SENTINEL_PID 2>/dev/null
wait $SENTINEL_PID 2>/dev/null

echo ""
echo "✅ Scan complete!"
echo ""
echo "📊 Exporting results..."
python3 sentinel_security.py --export

echo ""
echo "📋 Device Summary:"
python3 -c "
from sentinel_security import DeviceDatabase
db = DeviceDatabase()
devices = db.get_all_devices()
print(f'Total devices found: {len(devices)}')
for d in devices:
    print(f\"  • {d['ip_address']:15s} {d['mac_address']:17s} {d.get('vendor', 'Unknown'):20s}\")
"

