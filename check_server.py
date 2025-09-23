#!/usr/bin/env python3
"""
Quick script to check if SquashPlot server is running on localhost:5000
Usage: python check_server.py
"""

import requests
import sys

def check_server():
    """Check if the server is running on localhost:5000"""
    try:
        response = requests.get("http://localhost:5000", timeout=5)
        print("✅ Server is running!")
        print(f"Status Code: {response.status_code}")
        print(f"🌐 Open in browser: http://localhost:5000")
        return True
    except requests.exceptions.ConnectionError:
        print("❌ Server is not running on localhost:5000")
        print("💡 Start the server with: python main.py --web")
        return False
    except requests.exceptions.Timeout:
        print("⏰ Server connection timed out")
        return False
    except Exception as e:
        print(f"❌ Error testing server: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Checking SquashPlot server on localhost:5000...")
    check_server()
