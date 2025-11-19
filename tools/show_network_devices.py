#!/usr/bin/env python3
"""Display network devices from Sentinel Security database"""

from sentinel_security import DeviceDatabase

db = DeviceDatabase()
devices = db.get_all_devices()

print('\n🛡️  SENTINEL SECURITY - NETWORK DEVICE REPORT')
print('=' * 70)
print(f'\n📊 Total Devices Found: {len(devices)}\n')

if devices:
    print(f'{"IP Address":<18} {"MAC Address":<20} {"Vendor":<25} {"Hostname":<30}')
    print('-' * 100)
    
    for d in devices:
        ip = d.get('ip_address', 'N/A')
        mac = d.get('mac_address', 'N/A')
        vendor = d.get('vendor') or 'Unknown'
        hostname = d.get('hostname') or 'N/A'
        
        print(f'{ip:<18} {mac:<20} {vendor:<25} {hostname:<30}')
    
    print('\n' + '=' * 70)
    print('\n💡 Device Details:\n')
    
    for d in devices:
        print(f'  📱 {d.get("ip_address")} ({d.get("hostname", "Unknown Hostname")})')
        print(f'     MAC: {d.get("mac_address")}')
        if d.get('vendor'):
            print(f'     Vendor: {d.get("vendor")}')
        print(f'     First Seen: {d.get("first_seen", "N/A")}')
        print(f'     Last Seen: {d.get("last_seen", "N/A")}')
        print()
else:
    print('⚠️  No devices found in database')
    print('   Run: sudo python3 sentinel_security.py -i en0')
    print('   Or: python3 sentinel_arp_reader.py')

print('=' * 70)
print('\n💡 For real-time monitoring:')
print('   sudo python3 sentinel_security.py -i en0\n')

