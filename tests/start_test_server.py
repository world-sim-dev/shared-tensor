#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test Server Starter

A lightweight server starter specifically for testing purposes.
This script can be used by test runners to automatically start/stop servers.
"""

import sys
import os
import time
import threading
import atexit
from contextlib import contextmanager

# Add parent directory to path for test imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared_tensor.server import SharedTensorServer


class TestServerManager:
    """Manages test server lifecycle"""
    
    def __init__(self, host="localhost", port=8080):
        self.host = host
        self.port = port
        self.server = None
        self.server_thread = None
        self.is_running = False
    
    def start_server(self, timeout=5):
        """Start server in background thread"""
        if self.is_running:
            return True
            
        print(f"🚀 Starting test server on {self.host}:{self.port}")
        
        self.server = SharedTensorServer(host=self.host, port=self.port)
        
        def run_server():
            try:
                self.server.start(blocking=True)
            except Exception as e:
                print(f"❌ Server error: {e}")
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        
        # Wait for server to start
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Try to connect to check if server is ready
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                result = sock.connect_ex((self.host, self.port))
                sock.close()
                if result == 0:
                    self.is_running = True
                    print(f"✅ Test server started successfully")
                    return True
            except:
                pass
            time.sleep(0.1)
        
        print(f"⚠️  Server startup timeout after {timeout}s")
        return False
    
    def stop_server(self):
        """Stop the test server"""
        if not self.is_running:
            return
            
        print("⏹️  Stopping test server...")
        if self.server:
            self.server.stop()
        
        if self.server_thread and self.server_thread.is_alive():
            # Give server time to shutdown gracefully
            time.sleep(0.5)
        
        self.is_running = False
        print("✅ Test server stopped")


# Global test server instance
_test_server = TestServerManager()

def start_test_server(host="localhost", port=8080, timeout=5):
    """Start test server (global instance)"""
    global _test_server
    _test_server.host = host
    _test_server.port = port
    return _test_server.start_server(timeout)

def stop_test_server():
    """Stop test server (global instance)"""
    global _test_server
    _test_server.stop_server()

@contextmanager
def test_server_context(host="localhost", port=8080):
    """Context manager for test server"""
    server_manager = TestServerManager(host, port)
    try:
        if server_manager.start_server():
            yield server_manager
        else:
            raise RuntimeError("Failed to start test server")
    finally:
        server_manager.stop_server()

# Register cleanup on exit
atexit.register(stop_test_server)


def main():
    """Run as standalone test server"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Start test server')
    parser.add_argument('--host', default='localhost', help='Server host')
    parser.add_argument('--port', type=int, default=8080, help='Server port')
    parser.add_argument('--timeout', type=int, default=30, help='Run timeout (seconds, 0=infinite)')
    
    args = parser.parse_args()
    
    if start_test_server(args.host, args.port):
        try:
            if args.timeout > 0:
                print(f"⏱️  Server will run for {args.timeout} seconds...")
                time.sleep(args.timeout)
            else:
                print("⏱️  Server running indefinitely (Ctrl+C to stop)...")
                while True:
                    time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            stop_test_server()
    else:
        print("❌ Failed to start test server")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

