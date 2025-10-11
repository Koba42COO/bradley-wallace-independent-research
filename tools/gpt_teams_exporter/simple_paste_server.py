#!/usr/bin/env python3
"""
Simple paste server for GPT Teams Archive - bypasses Flask issues
"""

import json
import time
import hashlib
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

class PasteRequestHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, exporter=None, **kwargs):
        self.exporter = exporter
        super().__init__(*args, **kwargs)

    def do_GET(self):
        if self.path == '/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {'status': 'ready', 'backend_connected': True, 'aiva_available': False}
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == '/paste':
            self.handle_paste()
        elif self.path == '/capture':
            self.handle_capture()
        else:
            self.send_response(404)
            self.end_headers()

    def handle_paste(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))

            # Create conversation object
            conversation = {
                'id': f"pasted_{int(time.time())}_{hash(data.get('content', ''))[:8]}",
                'title': data.get('title', 'Pasted Conversation'),
                'create_time': time.time(),
                'update_time': time.time(),
                'pasted_at': data.get('pasted_at'),
                'content': data.get('content', ''),
                'messages': data.get('messages', []),
                'message_count': data.get('message_count', 1),
                'word_count': data.get('word_count', 0),
                'source': 'pasted',
                'mapping': {}
            }

            # Convert messages to mapping format
            if data.get('messages'):
                for i, msg in enumerate(data['messages']):
                    conversation['mapping'][f'pasted_{i}'] = {
                        'message': {
                            'content': {'parts': [msg.get('content', '')]},
                            'author': {'role': msg.get('role', 'user')},
                            'create_time': time.time()
                        }
                    }

            # Process with exporter if available
            if self.exporter:
                classification = self.exporter._classify_conversation(conversation)
                saved_files = self.exporter._save_conversation(conversation, classification)
                self.exporter._save_to_index(conversation)

                # Update AIVA memory
                results = [{
                    'id': conversation['id'],
                    'title': conversation['title'],
                    'classification': classification,
                    'saved_files': saved_files,
                    'message_count': conversation['message_count']
                }]
                self.exporter._update_aiva_memory(results)
            else:
                classification = 'unknown'

            response = {
                'status': 'pasted',
                'id': conversation['id'],
                'title': conversation['title'],
                'classification': classification,
                'message_count': conversation['message_count'],
                'word_count': conversation['word_count'],
                'pasted_at': conversation['pasted_at'],
                'source': 'pasted'
            }

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        except Exception as e:
            print(f"Paste error: {e}")
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode())

    def handle_capture(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'received', 'data_length': len(str(data))}).encode())

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode())

    def log_message(self, format, *args):
        # Suppress default logging
        pass

def run_simple_server(port=8765, exporter=None):
    """Run the simple paste server"""
    server_address = ('127.0.0.1', port)

    def handler_class(*args, **kwargs):
        return PasteRequestHandler(*args, exporter=exporter, **kwargs)

    httpd = HTTPServer(server_address, handler_class)
    print(f"🚀 Simple Paste Server running at http://127.0.0.1:{port}")
    print("📝 Ready for paste requests from Brave extension")
    print("Press Ctrl+C to stop")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
        httpd.shutdown()

if __name__ == '__main__':
    # Try to load the exporter if possible
    exporter = None
    try:
        from main import GPTTeamsExporter
        config = {
            'ARTIFACTS_DIR': 'artifacts/gpt_convos',
            'AIVA_MEMORY_DIR': 'aiva-core/data/memories',
            'RUN_DATA_DIR': 'artifacts/run-data',
            'CLASSIFICATION_CONFIG': 'configs/gpt_scraper_classification.yaml'
        }
        exporter = GPTTeamsExporter(config)
        print("✅ GPT Teams Exporter loaded")
    except Exception as e:
        print(f"⚠️ Could not load exporter: {e}")
        print("Server will run in basic mode without AIVA integration")

    run_simple_server(exporter=exporter)
