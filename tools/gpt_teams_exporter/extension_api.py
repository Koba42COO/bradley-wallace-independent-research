"""
Simple extension API server for GPT Teams Archive
"""

from flask import Flask, request, jsonify

def create_simple_api():
    """Create a simple Flask API for testing"""
    app = Flask(__name__)

    @app.route('/status')
    def status():
        return jsonify({
            'status': 'ready',
            'backend_connected': True,
            'aiva_available': False
        })

    @app.route('/capture', methods=['POST'])
    def capture():
        return jsonify({'status': 'received', 'message': 'capture endpoint works'})

    @app.route('/progress')
    def progress():
        return jsonify({'running': False, 'progress': 0})

    return app

if __name__ == '__main__':
    app = create_simple_api()
    from werkzeug.serving import make_server
    server = make_server('127.0.0.1', 8765, app, threaded=True)
    print("Starting simple extension API at http://localhost:8765")
    server.serve_forever()
