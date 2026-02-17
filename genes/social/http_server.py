"""
GENE: HTTP_SERVER
Category: social
Function: Exposes status via HTTP interface
Dependencies: []
Resource Cost: CPU=0.5%, MEM=5MB
"""

import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread
from typing import Callable, Dict

class HTTPServerGene:
    """Exposes organism state via HTTP API"""
    
    METADATA = {
        'name': 'HTTP_SERVER',
        'category': 'social',
        'essential': False,
        'priority': 20
    }
    
    def __init__(self, port: int = 8899, state_callback: Callable = None):
        self.port = port
        self.state_callback = state_callback or (lambda: {})
        self.server = None
        self.server_thread = None
        self.active = False
    
    def express(self):
        """Gene expression - start HTTP server"""
        state_callback = self.state_callback
        
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/status':
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    state = state_callback()
                    self.wfile.write(json.dumps(state, indent=2).encode())
                elif self.path == '/health':
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/plain')
                    self.end_headers()
                    self.wfile.write(b'OK')
                else:
                    self.send_response(404)
                    self.end_headers()
            
            def log_message(self, format, *args):
                pass  # Suppress logs
        
        try:
            self.server = HTTPServer(('0.0.0.0', self.port), Handler)
            self.server_thread = Thread(target=self.server.serve_forever, daemon=True)
            self.server_thread.start()
            self.active = True
            print(f"[HTTP_SERVER] Listening on port {self.port}")
        except Exception as e:
            print(f"[HTTP_SERVER] Failed to start: {e}")
    
    def shutdown(self):
        """Stop HTTP server"""
        if self.server:
            self.server.shutdown()
            self.active = False
