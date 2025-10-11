#!/usr/bin/env python3
"""
Working GPT Teams Archive Server - Simple and reliable
"""

import json
import time
import hashlib
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

class GPTArchiveHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, exporter=None, **kwargs):
        self.exporter = exporter
        super().__init__(*args, **kwargs)

    def do_GET(self):
        if self.path == '/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            response = {
                'status': 'ready',
                'backend_connected': True,
                'aiva_available': self.exporter is not None,
                'server': 'working_server'
            }
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        print(f"DEBUG: do_POST called with path: {self.path}")
        if self.path == '/capture':
            print("DEBUG: Calling handle_paste")
            self.handle_paste()
        else:
            print(f"DEBUG: Unknown path: {self.path}")
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        # Handle CORS preflight requests
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def handle_paste(self):
        try:
            print("DEBUG: Starting handle_paste")
            content_length = int(self.headers['Content-Length'])
            print(f"DEBUG: Content length: {content_length}")
            post_data = self.rfile.read(content_length)
            print(f"DEBUG: Raw post data: {post_data[:100]}...")
            data = json.loads(post_data.decode('utf-8'))
            print(f"DEBUG: Parsed JSON: {data}")

            # Process the conversation
            result = self.process_conversation(data)

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())

        except Exception as e:
            print(f"Paste error: {e}")
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode())

    def process_conversation(self, data):
        """Process a pasted conversation"""
        print(f"DEBUG: Processing conversation data: {data.keys()}")

        # Create ID safely
        try:
            content_str = data.get('content', '')
            content_hash = str(abs(hash(content_str)))[-8:]  # Use abs to avoid negative hashes
            conv_id = f"pasted_{int(time.time())}_{content_hash}"
        except Exception as e:
            print(f"DEBUG: Hash error: {e}")
            conv_id = f"pasted_{int(time.time())}_{int(time.time()) % 10000}"

        conversation = {
            'id': conv_id,
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

        print(f"DEBUG: Created conversation: {conversation['id']}")

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

        # Classify the conversation
        classification = self.classify_conversation(conversation)

        # Save files if exporter available
        saved_files = []
        if self.exporter:
            try:
                print(f"DEBUG: Saving conversation with classification: {classification}")
                saved_files = self.exporter._save_conversation(conversation, classification)
                print(f"DEBUG: Saved files: {saved_files}")
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
                print("DEBUG: AIVA memory updated")
            except Exception as e:
                print(f"File save error: {e}")
                import traceback
                traceback.print_exc()

        return {
            'status': 'pasted',
            'id': conversation['id'],
            'title': conversation['title'],
            'classification': classification,
            'message_count': conversation['message_count'],
            'word_count': conversation['word_count'],
            'pasted_at': conversation['pasted_at'],
            'file_path': saved_files[0] if saved_files else None,
            'source': 'pasted',
            'saved_files': len(saved_files)
        }

    def classify_conversation(self, conversation):
        """Simple classification based on content"""
        content = conversation.get('content', '').lower()

        # Define classification rules
        rules = {
            'philosophy_theory': ['consciousness', 'quantum', 'reality', 'existence', 'theory', 'philosophy', 'mind', 'intelligence'],
            'physics': ['physics', 'quantum', 'relativity', 'electromagnetic', 'thermodynamics', 'mechanics'],
            'ml': ['machine learning', 'neural network', 'deep learning', 'ai', 'artificial intelligence', 'model', 'training'],
            'cryptography': ['encryption', 'crypto', 'security', 'rsa', 'diffie-hellman', 'hash', 'signature'],
            'math': ['mathematics', 'algebra', 'calculus', 'geometry', 'topology', 'number theory', 'prime'],
            'systems': ['distributed', 'architecture', 'scalability', 'performance', 'infrastructure', 'kubernetes', 'docker']
        }

        for category, keywords in rules.items():
            if any(keyword in content for keyword in keywords):
                return category

        return 'philosophy_theory'  # Default

    def log_message(self, format, *args):
        # Suppress default logging
        pass

def run_server(port=8765):
    """Run the working GPT Archive server"""
    server_address = ('127.0.0.1', port)

    # Try to load the exporter
    exporter = None
    try:
        from main import GPTTeamsExporter
        config = {
            'ARTIFACTS_DIR': 'artifacts/gpt_convos',
            'AIVA_MEMORY_DIR': '../../../aiva-core/data/memories',
            'RUN_DATA_DIR': 'artifacts/run-data',
            'CLASSIFICATION_CONFIG': '../../../configs/gpt_scraper_classification.yaml'
        }
        exporter = GPTTeamsExporter(config)
        print("✅ GPT Teams Exporter loaded")
    except Exception as e:
        print(f"⚠️ Could not load exporter: {e}")
        print("Server will run in basic mode")

    def handler_class(*args, **kwargs):
        return GPTArchiveHandler(*args, exporter=exporter, **kwargs)

    httpd = HTTPServer(server_address, handler_class)

    print("🚀 Working GPT Archive Server")
    print(f"📍 Running at: http://127.0.0.1:{port}")
    print("📝 Ready for paste requests from Brave extension")
    print("🎯 Supports: /status, /capture")
    print("Press Ctrl+C to stop")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
        httpd.shutdown()

if __name__ == '__main__':
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    run_server(port)
