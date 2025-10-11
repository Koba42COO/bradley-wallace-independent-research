#!/usr/bin/env python3
"""
Super simple echo server to test HTTP handling
"""

import json
from http.server import HTTPServer, BaseHTTPRequestHandler

class EchoHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            response = {'status': 'ready', 'server': 'echo_server'}
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == '/capture':
            try:
                print("ECHO: Received POST to /capture")
                content_length = int(self.headers['Content-Length'])
                print(f"ECHO: Content length: {content_length}")
                post_data = self.rfile.read(content_length)
                print(f"ECHO: Raw data: {post_data}")

                # Just echo back what we received
                data = json.loads(post_data.decode('utf-8'))
                print(f"ECHO: Parsed data: {data}")

                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                response = {
                    'status': 'echo',
                    'received': data,
                    'echo': 'Hello from echo server!'
                }
                self.wfile.write(json.dumps(response).encode())
                print("ECHO: Response sent")

            except Exception as e:
                print(f"ECHO: Error: {e}")
                import traceback
                traceback.print_exc()
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def log_message(self, format, *args):
        # Suppress default logging
        pass

def run_echo_server(port=8765):
    """Run the echo server"""
    server_address = ('127.0.0.1', port)

    def handler_class(*args, **kwargs):
        return EchoHandler(*args, **kwargs)

    httpd = HTTPServer(server_address, handler_class)

    print("🔊 ECHO SERVER")
    print(f"📍 Running at: http://127.0.0.1:{port}")
    print("🎯 Just echoes back what you send")
    print("Press Ctrl+C to stop")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Echo server stopped")
        httpd.shutdown()

if __name__ == '__main__':
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    run_echo_server(port)
