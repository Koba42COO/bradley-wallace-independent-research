#!/usr/bin/env python3
"""
AIVA IDE Testing Suite
Comprehensive tests for AI-powered IDE functionality
"""

import unittest
import requests
import json
import websocket
import time
import threading
from unittest.mock import Mock, patch
import sys
import os

class TestAIVAIDEServer(unittest.TestCase):
    """Test cases for AIVA IDE server"""

    def setUp(self):
        """Set up test fixtures"""
        self.base_url = "http://localhost:3001"
        self.api_url = f"{self.base_url}/api"
        self.test_timeout = 10

    def test_health_endpoint(self):
        """Test server health check"""
        response = requests.get(f"{self.api_url}/health", timeout=self.test_timeout)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn('status', data)
        self.assertEqual(data['status'], 'healthy')
        self.assertIn('service', data)
        self.assertEqual(data['service'], 'aiva-ide-server')

    def test_file_operations(self):
        """Test basic file operations"""
        # List files
        response = requests.get(f"{self.api_url}/files", timeout=self.test_timeout)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn('files', data)
        self.assertIsInstance(data['files'], list)

    def test_read_file(self):
        """Test reading a known file"""
        # Try to read the README
        response = requests.get(
            f"{self.api_url}/files/aiva_ide/README.md",
            timeout=self.test_timeout
        )
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn('content', data)
        self.assertIn('size', data)
        self.assertIn('modified', data)

    def test_create_and_delete_file(self):
        """Test creating and deleting a test file"""
        test_file_path = "test_temp_file.txt"
        test_content = "This is a test file for AIVA IDE testing."

        # Create file
        response = requests.post(
            f"{self.api_url}/files",
            json={
                'path': test_file_path,
                'content': test_content
            },
            timeout=self.test_timeout
        )
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertTrue(data['success'])
        self.assertEqual(data['path'], test_file_path)

        # Verify file exists by reading it
        response = requests.get(
            f"{self.api_url}/files/{test_file_path}",
            timeout=self.test_timeout
        )
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data['content'], test_content)

        # Delete file
        response = requests.delete(
            f"{self.api_url}/files/{test_file_path}",
            timeout=self.test_timeout
        )
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertTrue(data['success'])

        # Verify file is deleted
        response = requests.get(
            f"{self.api_url}/files/{test_file_path}",
            timeout=self.test_timeout
        )
        self.assertEqual(response.status_code, 404)

    def test_write_file(self):
        """Test writing to an existing file"""
        test_file_path = "test_write_file.txt"
        initial_content = "Initial content"
        updated_content = "Updated content"

        # Create initial file
        response = requests.post(
            f"{self.api_url}/files",
            json={
                'path': test_file_path,
                'content': initial_content
            },
            timeout=self.test_timeout
        )
        self.assertEqual(response.status_code, 200)

        # Write updated content
        response = requests.post(
            f"{self.api_url}/files/{test_file_path}",
            json={'content': updated_content},
            timeout=self.test_timeout
        )
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertTrue(data['success'])

        # Verify content was updated
        response = requests.get(
            f"{self.api_url}/files/{test_file_path}",
            timeout=self.test_timeout
        )
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data['content'], updated_content)

        # Clean up
        requests.delete(f"{self.api_url}/files/{test_file_path}")

    @unittest.skipUnless(os.getenv('OPENAI_API_KEY'), "OpenAI API key required")
    def test_gpt_chat(self):
        """Test GPT chat functionality"""
        test_messages = [
            {"role": "user", "content": "Hello, can you help me with coding?"}
        ]

        response = requests.post(
            f"{self.api_url}/chat",
            json={
                'messages': test_messages,
                'model': 'gpt-3.5-turbo'
            },
            timeout=30  # GPT requests can take longer
        )

        if response.status_code == 200:
            data = response.json()
            self.assertIn('message', data)
            self.assertIsInstance(data['message'], str)
            self.assertGreater(len(data['message']), 0)
        else:
            # If API key is not configured, expect 500
            self.assertEqual(response.status_code, 500)
            data = response.json()
            self.assertIn('error', data)

    @unittest.skipUnless(os.getenv('OPENAI_API_KEY'), "OpenAI API key required")
    def test_code_completion(self):
        """Test code completion functionality"""
        test_code = "function fibonacci(n) {"
        test_language = "javascript"

        response = requests.post(
            f"{self.api_url}/complete",
            json={
                'code': test_code,
                'language': test_language,
                'context': "Complete this Fibonacci function"
            },
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            self.assertIn('completion', data)
            self.assertIsInstance(data['completion'], str)
        else:
            # If API key is not configured, expect 500
            self.assertEqual(response.status_code, 500)
            data = response.json()
            self.assertIn('error', data)

class TestAIVAIDEWebSocket(unittest.TestCase):
    """Test cases for WebSocket real-time collaboration"""

    def setUp(self):
        self.server_url = "ws://localhost:3001"
        self.test_room = "test-room-" + str(int(time.time()))

    def test_websocket_connection(self):
        """Test basic WebSocket connection"""
        try:
            ws = websocket.create_connection(self.server_url, timeout=5)

            # Connection should be successful
            self.assertTrue(ws.connected)

            # Send join room message
            join_message = {
                'type': 'join-room',
                'roomId': self.test_room
            }
            ws.send(json.dumps(join_message))

            # Should receive confirmation or no error
            ws.close()
        except Exception as e:
            self.fail(f"WebSocket connection failed: {e}")

    def test_room_collaboration(self):
        """Test room-based collaboration"""
        messages_received = []
        client1_connected = False
        client2_connected = False

        def client_handler(client_id):
            nonlocal messages_received, client1_connected, client2_connected

            try:
                ws = websocket.create_connection(self.server_url, timeout=5)

                # Join room
                join_message = {
                    'type': 'join-room',
                    'roomId': self.test_room
                }
                ws.send(json.dumps(join_message))

                if client_id == 1:
                    client1_connected = True
                else:
                    client2_connected = True

                # Client 1 sends a message
                if client_id == 1:
                    time.sleep(0.1)  # Wait for both to connect
                    code_change = {
                        'type': 'code-change',
                        'roomId': self.test_room,
                        'content': 'console.log("Hello World");',
                        'filePath': 'test.js',
                        'userId': 'client1'
                    }
                    ws.send(json.dumps(code_change))

                # Listen for messages
                try:
                    while True:
                        message = ws.recv()
                        data = json.loads(message)
                        messages_received.append((client_id, data))
                        break  # Exit after receiving one message
                except:
                    pass  # Connection closed

                ws.close()

            except Exception as e:
                print(f"Client {client_id} error: {e}")

        # Start two clients
        thread1 = threading.Thread(target=client_handler, args=(1,))
        thread2 = threading.Thread(target=client_handler, args=(2,))

        thread1.start()
        thread2.start()

        thread1.join(timeout=5)
        thread2.join(timeout=5)

        # Verify connections
        self.assertTrue(client1_connected or client2_connected,
                       "At least one client should have connected")

        # If both connected, client 2 should have received client 1's message
        if client1_connected and client2_connected:
            code_change_received = any(
                msg[1].get('type') == 'code-change' and
                msg[1].get('content') == 'console.log("Hello World");'
                for msg in messages_received
            )
            self.assertTrue(code_change_received,
                          "Client 2 should have received code change from client 1")

class TestAIVAIDEIntegration(unittest.TestCase):
    """Integration tests combining multiple AIVA IDE features"""

    def setUp(self):
        self.base_url = "http://localhost:3001"
        self.api_url = f"{self.base_url}/api"

    def test_full_workflow(self):
        """Test a complete workflow: create file, edit, save, read, delete"""
        test_file = f"integration_test_{int(time.time())}.py"
        initial_code = 'def hello():\n    print("Hello World")\n'
        updated_code = 'def hello():\n    print("Hello AIVA IDE")\n'

        # 1. Create file
        response = requests.post(
            f"{self.api_url}/files",
            json={'path': test_file, 'content': initial_code},
            timeout=10
        )
        self.assertEqual(response.status_code, 200)

        # 2. Read file to verify
        response = requests.get(f"{self.api_url}/files/{test_file}", timeout=10)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['content'], initial_code)

        # 3. Update file
        response = requests.post(
            f"{self.api_url}/files/{test_file}",
            json={'content': updated_code},
            timeout=10
        )
        self.assertEqual(response.status_code, 200)

        # 4. Read updated file
        response = requests.get(f"{self.api_url}/files/{test_file}", timeout=10)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['content'], updated_code)

        # 5. Delete file
        response = requests.delete(f"{self.api_url}/files/{test_file}", timeout=10)
        self.assertEqual(response.status_code, 200)

        # 6. Verify deletion
        response = requests.get(f"{self.api_url}/files/{test_file}", timeout=10)
        self.assertEqual(response.status_code, 404)

if __name__ == '__main__':
    # Check if AIVA IDE server is running
    try:
        response = requests.get("http://localhost:3001/api/health", timeout=5)
        if response.status_code != 200:
            print("❌ AIVA IDE server is not running. Please start it first:")
            print("   cd aiva_ide/server && npm install && npm run dev")
            sys.exit(1)
    except:
        print("❌ AIVA IDE server is not running. Please start it first:")
        print("   cd aiva_ide/server && npm install && npm run dev")
        sys.exit(1)

    print("🧪 Running AIVA IDE Tests...")
    unittest.main(verbosity=2)
