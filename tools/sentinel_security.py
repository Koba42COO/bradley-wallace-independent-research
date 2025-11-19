#!/usr/bin/env python3
"""
🛡️ Sentinel Security - Advanced Network Monitoring & Device Tracking
====================================================================

A comprehensive network security monitoring system with ARP watch capabilities,
device tracking, anomaly detection, and real-time alerting.

Features:
- ARP monitoring (arpwatch-like functionality)
- Real-time device discovery and tracking
- MAC address fingerprinting
- Network anomaly detection
- Device history and logging
- Alert system for new/unknown devices
- Multiple interface support
- Web dashboard (optional)

Author: Bradley Wallace (COO Koba42)
Framework: Universal Prime Graph Protocol φ.1
"""

import os
import sys
import json
import time
import signal
import sqlite3
import hashlib
import subprocess
import threading
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import re

try:
    import scapy.all as scapy
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    print("⚠️  Warning: scapy not available. Install with: pip install scapy")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️  Warning: psutil not available. Install with: pip install psutil")


# ============================================================================
# UPG FOUNDATIONS - Universal Prime Graph Protocol φ.1
# ============================================================================
from decimal import Decimal, getcontext
import math

getcontext().prec = 50

class UPGConstants:
    """Universal Prime Graph consciousness mathematics constants"""
    PHI = Decimal('1.618033988749895')
    DELTA = Decimal('2.414213562373095')
    CONSCIOUSNESS = Decimal('0.79')  # 79/21 universal coherence rule
    REALITY_DISTORTION = Decimal('1.1808')  # Quantum amplification factor
    CONSCIOUSNESS_DIMENSIONS = 21  # Prime topology dimension
    COHERENCE_THRESHOLD = Decimal('1e-15')


@dataclass
class NetworkDevice:
    """Represents a device on the network"""
    ip_address: str
    mac_address: str
    first_seen: datetime
    last_seen: datetime
    vendor: Optional[str] = None
    hostname: Optional[str] = None
    device_type: Optional[str] = None
    is_known: bool = False
    alert_count: int = 0
    packet_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'ip_address': self.ip_address,
            'mac_address': self.mac_address,
            'first_seen': self.first_seen.isoformat(),
            'last_seen': self.last_seen.isoformat(),
            'vendor': self.vendor,
            'hostname': self.hostname,
            'device_type': self.device_type,
            'is_known': self.is_known,
            'alert_count': self.alert_count,
            'packet_count': self.packet_count
        }


class MACVendorDatabase:
    """MAC address vendor lookup database"""
    
    def __init__(self):
        self.vendor_db = {}
        self._load_vendor_database()
    
    def _load_vendor_database(self):
        """Load MAC vendor database from common prefixes"""
        # Common vendor prefixes (simplified - in production, use full OUI database)
        common_vendors = {
            '00:50:56': 'VMware',
            '00:0c:29': 'VMware',
            '00:05:69': 'VMware',
            '08:00:27': 'VirtualBox',
            '52:54:00': 'QEMU',
            '00:1b:21': 'Apple',
            '00:23:12': 'Apple',
            '00:25:00': 'Apple',
            '00:26:4a': 'Apple',
            '00:26:bb': 'Apple',
            'ac:de:48': 'Apple',
            'f0:18:98': 'Apple',
            'f4:f5:e8': 'Apple',
            '00:50:f2': 'Microsoft',
            '00:15:5d': 'Microsoft',
            '00:03:ff': 'Microsoft',
            '00:0d:3a': 'Microsoft',
            '00:1d:7d': 'Samsung',
            '00:23:39': 'Samsung',
            '00:26:e8': 'Samsung',
            '00:15:99': 'Samsung',
            '00:1e:13': 'Samsung',
            '00:1f:cc': 'Samsung',
            '00:23:6c': 'Samsung',
            '00:24:90': 'Samsung',
            '00:25:66': 'Samsung',
            '00:26:5e': 'Samsung',
            '00:50:43': 'Sony',
            '00:13:15': 'Sony',
            '00:16:fe': 'Sony',
            '00:1a:80': 'Sony',
            '00:1b:0d': 'Sony',
            '00:1c:bf': 'Sony',
            '00:1e:ec': 'Sony',
            '00:21:9b': 'Sony',
            '00:24:be': 'Sony',
            '00:26:4c': 'Sony',
        }
        
        # Normalize MAC addresses for lookup
        for prefix, vendor in common_vendors.items():
            self.vendor_db[prefix.lower()] = vendor
    
    def lookup_vendor(self, mac_address: str) -> Optional[str]:
        """Look up vendor from MAC address"""
        mac_upper = mac_address.upper().replace('-', ':')
        prefix = ':'.join(mac_upper.split(':')[:3])
        return self.vendor_db.get(prefix.lower())


class NetworkInterfaceManager:
    """Manages network interfaces"""
    
    def __init__(self):
        self.interfaces = []
        self._discover_interfaces()
    
    def _discover_interfaces(self):
        """Discover available network interfaces"""
        if PSUTIL_AVAILABLE:
            try:
                interfaces = psutil.net_if_addrs()
                for iface_name, addrs in interfaces.items():
                    for addr in addrs:
                        if addr.family == 2:  # IPv4
                            if not iface_name.startswith('lo') and not iface_name.startswith('docker'):
                                self.interfaces.append({
                                    'name': iface_name,
                                    'ip': addr.address,
                                    'netmask': addr.netmask
                                })
            except Exception as e:
                print(f"⚠️  Error discovering interfaces: {e}")
        
        # Fallback to system commands
        if not self.interfaces:
            try:
                if sys.platform == 'linux':
                    result = subprocess.run(['ip', 'addr', 'show'], 
                                           capture_output=True, text=True)
                    # Parse ip addr output
                    for line in result.stdout.split('\n'):
                        if 'inet ' in line and '127.0.0.1' not in line:
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                ip = parts[1].split('/')[0]
                                # Try to get interface name from previous lines
                                self.interfaces.append({
                                    'name': 'eth0',  # Default fallback
                                    'ip': ip,
                                    'netmask': '255.255.255.0'
                                })
                elif sys.platform == 'darwin':  # macOS
                    result = subprocess.run(['ifconfig'], 
                                           capture_output=True, text=True)
                    # Parse ifconfig output
                    current_iface = None
                    for line in result.stdout.split('\n'):
                        if not line.startswith(' ') and not line.startswith('\t'):
                            current_iface = line.split(':')[0]
                        elif 'inet ' in line and '127.0.0.1' not in line:
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                ip = parts[1]
                                if current_iface:
                                    self.interfaces.append({
                                        'name': current_iface,
                                        'ip': ip,
                                        'netmask': '255.255.255.0'
                                    })
            except Exception as e:
                print(f"⚠️  Error in fallback interface discovery: {e}")
    
    def list_interfaces(self) -> List[Dict[str, str]]:
        """List all available interfaces"""
        return self.interfaces
    
    def get_default_interface(self) -> Optional[str]:
        """Get the default network interface"""
        if self.interfaces:
            return self.interfaces[0]['name']
        return None


class ARPMonitor:
    """ARP packet monitoring and device tracking"""
    
    def __init__(self, interface: str, callback=None):
        self.interface = interface
        self.callback = callback
        self.running = False
        self.devices: Dict[str, NetworkDevice] = {}
        self.mac_vendor_db = MACVendorDatabase()
        self.packet_count = 0
        
    def start_monitoring(self):
        """Start ARP monitoring"""
        if not SCAPY_AVAILABLE:
            print("❌ Error: scapy is required for ARP monitoring")
            print("   Install with: pip install scapy")
            return False
        
        if os.geteuid() != 0:
            print("⚠️  Warning: ARP monitoring requires root privileges")
            print("   Run with: sudo python3 sentinel_security.py")
        
        self.running = True
        print(f"🛡️  Starting ARP monitoring on interface: {self.interface}")
        
        try:
            # Use scapy to sniff ARP packets
            scapy.sniff(
                iface=self.interface,
                filter="arp",
                prn=self._process_arp_packet,
                stop_filter=lambda x: not self.running
            )
        except Exception as e:
            print(f"❌ Error in ARP monitoring: {e}")
            return False
        
        return True
    
    def _process_arp_packet(self, packet):
        """Process an ARP packet"""
        if not self.running:
            return
        
        try:
            if packet.haslayer(scapy.ARP):
                arp = packet[scapy.ARP]
                
                # Only process ARP replies and requests
                if arp.op in [1, 2]:  # 1 = request, 2 = reply
                    ip_addr = arp.psrc
                    mac_addr = arp.hwsrc
                    
                    self.packet_count += 1
                    self._update_device(ip_addr, mac_addr)
                    
        except Exception as e:
            print(f"⚠️  Error processing ARP packet: {e}")
    
    def _update_device(self, ip_address: str, mac_address: str):
        """Update or create device record"""
        now = datetime.now()
        
        if mac_address in self.devices:
            # Update existing device
            device = self.devices[mac_address]
            device.last_seen = now
            device.packet_count += 1
            
            # Check for IP change (potential ARP spoofing)
            if device.ip_address != ip_address:
                device.alert_count += 1
                if self.callback:
                    self.callback('ip_change', {
                        'mac': mac_address,
                        'old_ip': device.ip_address,
                        'new_ip': ip_address
                    })
        else:
            # New device discovered
            vendor = self.mac_vendor_db.lookup_vendor(mac_address)
            device = NetworkDevice(
                ip_address=ip_address,
                mac_address=mac_address,
                first_seen=now,
                last_seen=now,
                vendor=vendor,
                is_known=False,
                alert_count=0,
                packet_count=1
            )
            self.devices[mac_address] = device
            
            if self.callback:
                self.callback('new_device', device.to_dict())
    
    def stop_monitoring(self):
        """Stop ARP monitoring"""
        self.running = False
    
    def get_devices(self) -> List[NetworkDevice]:
        """Get all discovered devices"""
        return list(self.devices.values())
    
    def get_device_count(self) -> int:
        """Get total device count"""
        return len(self.devices)


class DeviceDatabase:
    """SQLite database for device history"""
    
    def __init__(self, db_path: str = "sentinel_devices.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS devices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ip_address TEXT NOT NULL,
                mac_address TEXT NOT NULL UNIQUE,
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                vendor TEXT,
                hostname TEXT,
                device_type TEXT,
                is_known INTEGER DEFAULT 0,
                alert_count INTEGER DEFAULT 0,
                packet_count INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_type TEXT NOT NULL,
                device_mac TEXT,
                message TEXT,
                severity TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (device_mac) REFERENCES devices(mac_address)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_device(self, device: NetworkDevice):
        """Save or update device in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO devices 
            (ip_address, mac_address, first_seen, last_seen, vendor, 
             hostname, device_type, is_known, alert_count, packet_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            device.ip_address,
            device.mac_address,
            device.first_seen.isoformat(),
            device.last_seen.isoformat(),
            device.vendor,
            device.hostname,
            device.device_type,
            1 if device.is_known else 0,
            device.alert_count,
            device.packet_count
        ))
        
        conn.commit()
        conn.close()
    
    def save_alert(self, alert_type: str, device_mac: str, message: str, severity: str = "info"):
        """Save alert to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO alerts (alert_type, device_mac, message, severity)
            VALUES (?, ?, ?, ?)
        ''', (alert_type, device_mac, message, severity))
        
        conn.commit()
        conn.close()
    
    def get_all_devices(self) -> List[Dict[str, Any]]:
        """Get all devices from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM devices ORDER BY last_seen DESC')
        columns = [desc[0] for desc in cursor.description]
        devices = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return devices
    
    def get_recent_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent alerts"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM alerts 
            ORDER BY created_at DESC 
            LIMIT ?
        ''', (limit,))
        
        columns = [desc[0] for desc in cursor.description]
        alerts = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return alerts


class SentinelSecurity:
    """
    🛡️ Sentinel Security - Main Security Monitoring System
    """
    
    def __init__(self, interface: Optional[str] = None, db_path: str = "sentinel_devices.db"):
        self.interface_manager = NetworkInterfaceManager()
        self.db = DeviceDatabase(db_path)
        self.monitor = None
        self.running = False
        
        # Determine interface
        if interface:
            self.interface = interface
        else:
            self.interface = self.interface_manager.get_default_interface()
            if not self.interface:
                print("❌ Error: No network interface found")
                sys.exit(1)
        
        print(f"🛡️  Sentinel Security initialized")
        print(f"   Interface: {self.interface}")
        print(f"   Database: {db_path}")
    
    def start_monitoring(self):
        """Start network monitoring"""
        if self.running:
            print("⚠️  Monitoring already running")
            return
        
        self.running = True
        
        # Initialize ARP monitor with callback
        self.monitor = ARPMonitor(self.interface, callback=self._handle_event)
        
        # Start monitoring in separate thread
        monitor_thread = threading.Thread(target=self.monitor.start_monitoring, daemon=True)
        monitor_thread.start()
        
        print("🛡️  Sentinel Security monitoring started")
        print("   Press Ctrl+C to stop")
        
        # Main loop - display updates
        try:
            while self.running:
                time.sleep(5)
                self._display_status()
        except KeyboardInterrupt:
            print("\n🛛 Stopping Sentinel Security...")
            self.stop_monitoring()
    
    def _handle_event(self, event_type: str, data: Dict[str, Any]):
        """Handle monitoring events"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if event_type == 'new_device':
            device = data
            message = f"New device detected: {device['ip_address']} ({device['mac_address']})"
            if device.get('vendor'):
                message += f" - {device['vendor']}"
            
            print(f"\n🔔 [{timestamp}] {message}")
            
            # Save to database
            device_obj = NetworkDevice(
                ip_address=device['ip_address'],
                mac_address=device['mac_address'],
                first_seen=datetime.fromisoformat(device['first_seen']),
                last_seen=datetime.fromisoformat(device['last_seen']),
                vendor=device.get('vendor'),
                is_known=False
            )
            self.db.save_device(device_obj)
            self.db.save_alert('new_device', device['mac_address'], message, 'warning')
        
        elif event_type == 'ip_change':
            message = f"IP change detected: {data['mac']} changed from {data['old_ip']} to {data['new_ip']}"
            print(f"\n⚠️  [{timestamp}] {message}")
            self.db.save_alert('ip_change', data['mac'], message, 'warning')
    
    def _display_status(self):
        """Display current monitoring status"""
        if self.monitor:
            device_count = self.monitor.get_device_count()
            packet_count = self.monitor.packet_count
            
            print(f"\r📊 Devices: {device_count} | Packets: {packet_count}", end='', flush=True)
    
    def stop_monitoring(self):
        """Stop network monitoring"""
        self.running = False
        if self.monitor:
            self.monitor.stop_monitoring()
        print("\n✅ Sentinel Security stopped")
    
    def list_devices(self) -> List[NetworkDevice]:
        """List all discovered devices"""
        if self.monitor:
            return self.monitor.get_devices()
        return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get monitoring statistics"""
        devices = self.list_devices()
        
        stats = {
            'total_devices': len(devices),
            'known_devices': sum(1 for d in devices if d.is_known),
            'unknown_devices': sum(1 for d in devices if not d.is_known),
            'total_packets': self.monitor.packet_count if self.monitor else 0,
            'interface': self.interface,
            'uptime': time.time() if self.running else 0
        }
        
        return stats
    
    def export_devices(self, filename: str = "sentinel_devices.json"):
        """Export devices to JSON"""
        devices = self.list_devices()
        data = {
            'export_time': datetime.now().isoformat(),
            'total_devices': len(devices),
            'devices': [d.to_dict() for d in devices]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Exported {len(devices)} devices to {filename}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='🛡️ Sentinel Security - Network Monitoring & Device Tracking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  sudo python3 sentinel_security.py                    # Start monitoring on default interface
  sudo python3 sentinel_security.py -i eth0            # Monitor specific interface
  sudo python3 sentinel_security.py --list-interfaces   # List available interfaces
  python3 sentinel_security.py --export                # Export device list
        '''
    )
    
    parser.add_argument('-i', '--interface', help='Network interface to monitor')
    parser.add_argument('--list-interfaces', action='store_true', 
                       help='List available network interfaces')
    parser.add_argument('--export', action='store_true',
                       help='Export devices to JSON and exit')
    parser.add_argument('--db', default='sentinel_devices.db',
                       help='Database file path (default: sentinel_devices.db)')
    
    args = parser.parse_args()
    
    # List interfaces
    if args.list_interfaces:
        manager = NetworkInterfaceManager()
        interfaces = manager.list_interfaces()
        print("📡 Available Network Interfaces:")
        for iface in interfaces:
            print(f"   {iface['name']}: {iface['ip']}")
        return
    
    # Export devices
    if args.export:
        sentinel = SentinelSecurity(interface=args.interface, db_path=args.db)
        sentinel.export_devices()
        return
    
    # Start monitoring
    sentinel = SentinelSecurity(interface=args.interface, db_path=args.db)
    sentinel.start_monitoring()


if __name__ == '__main__':
    main()

