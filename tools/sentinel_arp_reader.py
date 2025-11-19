#!/usr/bin/env python3
"""
🛡️ Sentinel Security - ARP Table Reader (No Root Required)
===========================================================

Reads system ARP table to show current network devices without requiring root.
"""

import subprocess
import re
from typing import List, Dict
from sentinel_security import MACVendorDatabase, NetworkDevice
from datetime import datetime

def read_arp_table() -> List[Dict[str, str]]:
    """Read system ARP table"""
    devices = []
    
    try:
        # Use arp -a command (works on macOS and Linux)
        result = subprocess.run(['arp', '-a'], capture_output=True, text=True)
        
        # Parse ARP table output
        # Format: hostname (ip) at mac [ether] on interface
        pattern = r'(\S+)\s+\((\d+\.\d+\.\d+\.\d+)\)\s+at\s+([0-9a-fA-F:]+)'
        
        for line in result.stdout.split('\n'):
            match = re.search(pattern, line)
            if match:
                hostname, ip, mac = match.groups()
                devices.append({
                    'hostname': hostname,
                    'ip_address': ip,
                    'mac_address': mac.upper()
                })
    except Exception as e:
        print(f"⚠️  Error reading ARP table: {e}")
    
    return devices

def main():
    """Display current network devices from ARP table"""
    print("🛡️  Sentinel Security - Network Device Scan")
    print("=" * 60)
    print("\nReading system ARP table...\n")
    
    devices = read_arp_table()
    vendor_db = MACVendorDatabase()
    
    if not devices:
        print("⚠️  No devices found in ARP table")
        print("   This could mean:")
        print("   - No recent network activity")
        print("   - ARP table is empty")
        print("   - Need to run with sudo for live monitoring")
        return
    
    print(f"📊 Found {len(devices)} device(s) in ARP table:\n")
    print(f"{'IP Address':<18} {'MAC Address':<20} {'Vendor':<25} {'Hostname'}")
    print("-" * 80)
    
    for device in devices:
        vendor = vendor_db.lookup_vendor(device['mac_address']) or 'Unknown'
        hostname = device['hostname'] or 'N/A'
        print(f"{device['ip_address']:<18} {device['mac_address']:<20} {vendor:<25} {hostname}")
    
    print("\n" + "=" * 60)
    print("\n💡 For real-time monitoring and new device detection, run:")
    print("   sudo python3 sentinel_security.py -i en0")
    print()

if __name__ == '__main__':
    main()

