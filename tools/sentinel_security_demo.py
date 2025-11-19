#!/usr/bin/env python3
"""
🛡️ Sentinel Security - Demo Script
===================================

Quick demonstration of Sentinel Security capabilities
"""

from sentinel_security import (
    SentinelSecurity,
    NetworkInterfaceManager,
    DeviceDatabase
)

def demo_interface_discovery():
    """Demonstrate interface discovery"""
    print("=" * 60)
    print("📡 Network Interface Discovery")
    print("=" * 60)
    
    manager = NetworkInterfaceManager()
    interfaces = manager.list_interfaces()
    
    if interfaces:
        print(f"\nFound {len(interfaces)} network interface(s):\n")
        for i, iface in enumerate(interfaces, 1):
            print(f"  {i}. {iface['name']}")
            print(f"     IP: {iface['ip']}")
            print(f"     Netmask: {iface['netmask']}")
            print()
    else:
        print("\n⚠️  No network interfaces found")
    
    print("=" * 60)
    print()


def demo_database_operations():
    """Demonstrate database operations"""
    print("=" * 60)
    print("💾 Database Operations Demo")
    print("=" * 60)
    
    db = DeviceDatabase("demo_sentinel.db")
    
    # Get all devices from database
    devices = db.get_all_devices()
    print(f"\n📊 Devices in database: {len(devices)}")
    
    if devices:
        print("\nRecent devices:")
        for device in devices[:5]:  # Show first 5
            print(f"  • {device.get('ip_address', 'N/A')} - {device.get('mac_address', 'N/A')}")
            if device.get('vendor'):
                print(f"    Vendor: {device['vendor']}")
    
    # Get recent alerts
    alerts = db.get_recent_alerts(limit=5)
    print(f"\n🔔 Recent alerts: {len(alerts)}")
    
    if alerts:
        for alert in alerts:
            print(f"  • [{alert.get('severity', 'info').upper()}] {alert.get('message', 'N/A')}")
    
    print("=" * 60)
    print()


def demo_statistics():
    """Show how to get statistics"""
    print("=" * 60)
    print("📊 Statistics Demo")
    print("=" * 60)
    
    print("\nTo get real-time statistics, run:")
    print("  sudo python3 sentinel_security.py")
    print("\nOr use the API:")
    print("""
from sentinel_security import SentinelSecurity

sentinel = SentinelSecurity()
stats = sentinel.get_statistics()
print(f"Total devices: {stats['total_devices']}")
print(f"Unknown devices: {stats['unknown_devices']}")
print(f"Total packets: {stats['total_packets']}")
    """)
    
    print("=" * 60)
    print()


def main():
    """Run all demos"""
    print("\n" + "=" * 60)
    print("🛡️  SENTINEL SECURITY - DEMONSTRATION")
    print("=" * 60)
    print()
    
    demo_interface_discovery()
    demo_database_operations()
    demo_statistics()
    
    print("\n" + "=" * 60)
    print("✅ Demo Complete")
    print("=" * 60)
    print("\nTo start monitoring, run:")
    print("  sudo python3 sentinel_security.py")
    print("\nFor help:")
    print("  python3 sentinel_security.py --help")
    print()


if __name__ == '__main__':
    main()

