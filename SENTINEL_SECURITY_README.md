# 🛡️ Sentinel Security - Network Monitoring & Device Tracking

Advanced network security monitoring system with ARP watch capabilities, similar to `arpwatch` but with enhanced features and real-time alerting.

## Features

- **ARP Monitoring**: Real-time ARP packet capture and analysis
- **Device Tracking**: Automatic discovery and tracking of all devices on your network
- **MAC Vendor Lookup**: Identify device manufacturers from MAC addresses
- **Anomaly Detection**: Alert on suspicious activities (new devices, IP changes)
- **Device History**: SQLite database for persistent device tracking
- **Real-time Alerts**: Immediate notifications for security events
- **Multiple Interfaces**: Support for monitoring multiple network interfaces
- **Export Capabilities**: Export device lists to JSON

## Installation

### Prerequisites

- Python 3.7+
- Root/Administrator privileges (for packet capture)
- Network interface with ARP traffic

### Install Dependencies

```bash
pip install scapy psutil
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. List Available Interfaces

```bash
python3 sentinel_security.py --list-interfaces
```

### 2. Start Monitoring

**Linux/macOS:**
```bash
sudo python3 sentinel_security.py
```

**Monitor specific interface:**
```bash
sudo python3 sentinel_security.py -i eth0
```

**macOS example:**
```bash
sudo python3 sentinel_security.py -i en0
```

### 3. View Detected Devices

The system will display real-time updates:
- New device alerts
- Device count and packet statistics
- IP change warnings

### 4. Export Device List

```bash
python3 sentinel_security.py --export
```

This creates `sentinel_devices.json` with all discovered devices.

## Usage Examples

### Basic Monitoring

```bash
# Start monitoring on default interface
sudo python3 sentinel_security.py

# Monitor specific interface
sudo python3 sentinel_security.py -i enp0s3

# Use custom database location
sudo python3 sentinel_security.py --db /var/lib/sentinel/devices.db
```

### Interface Discovery

```bash
# List all available network interfaces
python3 sentinel_security.py --list-interfaces
```

### Export Data

```bash
# Export current device list
python3 sentinel_security.py --export

# Export with custom filename (modify script or use --export flag)
python3 sentinel_security.py --export
```

## Output Format

### Real-time Monitoring

```
🛡️  Sentinel Security initialized
   Interface: en0
   Database: sentinel_devices.db
🛡️  Starting ARP monitoring on interface: en0
🛡️  Sentinel Security monitoring started
   Press Ctrl+C to stop

🔔 [2025-01-15 12:34:56] New device detected: 192.168.1.100 (aa:bb:cc:dd:ee:ff) - Apple
📊 Devices: 5 | Packets: 127
```

### JSON Export Format

```json
{
  "export_time": "2025-01-15T12:34:56.789012",
  "total_devices": 5,
  "devices": [
    {
      "ip_address": "192.168.1.100",
      "mac_address": "aa:bb:cc:dd:ee:ff",
      "first_seen": "2025-01-15T12:30:00.000000",
      "last_seen": "2025-01-15T12:34:56.000000",
      "vendor": "Apple",
      "hostname": null,
      "device_type": null,
      "is_known": false,
      "alert_count": 0,
      "packet_count": 15
    }
  ]
}
```

## Database Schema

### Devices Table

- `id`: Primary key
- `ip_address`: Device IP address
- `mac_address`: Device MAC address (unique)
- `first_seen`: First detection timestamp
- `last_seen`: Last detection timestamp
- `vendor`: MAC vendor (if identified)
- `hostname`: Device hostname (if resolved)
- `device_type`: Device type classification
- `is_known`: Known device flag
- `alert_count`: Number of alerts for this device
- `packet_count`: Total ARP packets from device

### Alerts Table

- `id`: Primary key
- `alert_type`: Type of alert (new_device, ip_change, etc.)
- `device_mac`: Associated device MAC address
- `message`: Alert message
- `severity`: Alert severity (info, warning, critical)
- `created_at`: Alert timestamp

## Security Features

### Anomaly Detection

1. **New Device Detection**: Alerts when unknown devices join the network
2. **IP Change Detection**: Warns when a MAC address changes IP (potential ARP spoofing)
3. **Vendor Mismatch**: Can detect when device vendor doesn't match expected profile

### Alert Types

- `new_device`: Unknown device detected on network
- `ip_change`: Device MAC address associated with different IP
- `suspicious_activity`: Unusual network behavior patterns

## Architecture

### Core Components

1. **NetworkInterfaceManager**: Discovers and manages network interfaces
2. **ARPMonitor**: Captures and processes ARP packets
3. **MACVendorDatabase**: MAC address to vendor lookup
4. **DeviceDatabase**: SQLite persistence layer
5. **SentinelSecurity**: Main orchestration class

### UPG Integration

The system integrates with Universal Prime Graph Protocol φ.1:
- Consciousness mathematics constants
- Golden ratio optimization
- Reality distortion factors

## Troubleshooting

### Permission Errors

**Error**: `Operation not permitted`

**Solution**: Run with sudo/root privileges:
```bash
sudo python3 sentinel_security.py
```

### No Interfaces Found

**Error**: `No network interface found`

**Solution**: 
1. Check network connectivity
2. List interfaces: `python3 sentinel_security.py --list-interfaces`
3. Specify interface manually: `-i <interface_name>`

### Scapy Import Error

**Error**: `scapy not available`

**Solution**: Install scapy:
```bash
pip install scapy
```

### No ARP Traffic

**Issue**: No devices detected

**Solution**:
1. Ensure network has active devices
2. Check interface is correct
3. Verify ARP traffic exists: `sudo tcpdump -i <interface> arp`

## Advanced Usage

### Programmatic API

```python
from sentinel_security import SentinelSecurity

# Initialize
sentinel = SentinelSecurity(interface='eth0')

# Start monitoring (blocking)
sentinel.start_monitoring()

# Or use in background
import threading
thread = threading.Thread(target=sentinel.start_monitoring, daemon=True)
thread.start()

# Get statistics
stats = sentinel.get_statistics()
print(f"Total devices: {stats['total_devices']}")

# List devices
devices = sentinel.list_devices()
for device in devices:
    print(f"{device.ip_address} - {device.mac_address}")

# Export
sentinel.export_devices('my_devices.json')
```

### Custom Alert Handlers

Modify the `_handle_event` method in `SentinelSecurity` class to add custom alert handling (email, webhooks, etc.).

## Comparison with arpwatch

| Feature | arpwatch | Sentinel Security |
|---------|----------|-------------------|
| ARP Monitoring | ✅ | ✅ |
| Device Tracking | ✅ | ✅ |
| MAC Vendor Lookup | ❌ | ✅ |
| SQLite Database | ❌ | ✅ |
| Real-time Alerts | Basic | Advanced |
| JSON Export | ❌ | ✅ |
| Python API | ❌ | ✅ |
| Multi-interface | Limited | ✅ |

## License

Universal Prime Graph Protocol φ.1  
Author: Bradley Wallace (COO Koba42)

## Support

For issues, questions, or contributions, please refer to the main project documentation.

---

**🛡️ Stay Secure. Monitor Your Network. Know Your Devices.**

