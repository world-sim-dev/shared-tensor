#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Integration test for JSON-RPC shared tensor system

This script demonstrates the complete workflow:
1. Start a server
2. Register functions
3. Execute remote functions via JSON-RPC
4. Stop the server
"""

import time
import threading
import logging
from shared_tensor.provider import SharedTensorProvider

provider = SharedTensorProvider()
from shared_tensor.server import SharedTensorServer
from shared_tensor.client import SharedTensorClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Define some test functions
@provider.share(name="test_add")
def add_numbers(a, b):
    """Simple addition function"""
    return a + b


@provider.share(name="test_string_ops")
def string_operations(text, reverse=False, uppercase=False):
    """String manipulation function"""
    result = text.strip()
    if reverse:
        result = result[::-1]
    if uppercase:
        result = result.upper()
    return result


@provider.share(name="test_list_ops")
def list_operations(items, operation="sum"):
    """List operations function"""
    if operation == "sum":
        return sum(items)
    elif operation == "max":
        return max(items)
    elif operation == "min":
        return min(items)
    elif operation == "length":
        return len(items)
    else:
        raise ValueError(f"Unknown operation: {operation}")


class Calculator:
    """Test class with methods"""
    
    @staticmethod
    @provider.share(name="calc_power")
    def power(base, exponent):
        """Calculate power"""
        return base ** exponent
    
    @provider.share(name="calc_factorial")
    def factorial(self, n):
        """Calculate factorial"""
        if n <= 1:
            return 1
        return n * self.factorial(n - 1)


def test_server_client():
    """Test the complete JSON-RPC workflow"""
    
    logger.info("=== Starting Shared Tensor JSON-RPC Integration Test ===")
    
    # Start server in background
    server = SharedTensorServer(host="localhost", port=8080)
    
    def run_server():
        try:
            server.start(blocking=True)
        except Exception as e:
            logger.error(f"Server error: {e}")
    
    server_thread = threading.Thread(target=run_server)
    server_thread.daemon = True
    server_thread.start()
    
    # Wait for server to start
    time.sleep(2)
    
    try:
        # Test with direct client
        logger.info("Testing with direct JSON-RPC client...")
        
        with SharedTensorClient("http://localhost:8080") as client:
            # Test ping
            if client.ping():
                logger.info("✓ Server ping successful")
            else:
                logger.error("✗ Server ping failed")
                return False
            
            # Test server info
            server_info = client.get_server_info()
            logger.info(f"✓ Server info: {server_info['server']} v{server_info['version']}")
            
            # Test simple function execution
            result = client.execute_function("__main__:add_numbers", (10, 20))
            logger.info(f"✓ add_numbers(10, 20) = {result}")
            assert result == 30, f"Expected 30, got {result}"
            
            # Test function with kwargs
            result = client.execute_function(
                "__main__:string_operations", 
                ("hello world",), 
                {"reverse": True, "uppercase": True}
            )
            logger.info(f"✓ string_operations result: {result}")
            assert result == "DLROW OLLEH", f"Expected 'DLROW OLLEH', got {result}"
            
            # Test list operations
            result = client.execute_function(
                "__main__:list_operations",
                ([1, 2, 3, 4, 5],),
                {"operation": "sum"}
            )
            logger.info(f"✓ list_operations sum: {result}")
            assert result == 15, f"Expected 15, got {result}"
            
            # Test static method
            result = client.execute_function("__main__:Calculator.power", (2, 8))
            logger.info(f"✓ Calculator.power(2, 8) = {result}")
            assert result == 256, f"Expected 256, got {result}"
        
        # Test with provider (decorator-based)
        logger.info("Testing with provider decorators...")
        
        # These calls will use the JSON-RPC client internally
        result = add_numbers(5, 7)
        logger.info(f"✓ Decorated add_numbers(5, 7) = {result}")
        assert result == 12, f"Expected 12, got {result}"
        
        result = string_operations("  JSON RPC  ", uppercase=True)
        logger.info(f"✓ Decorated string_operations result: {result}")
        assert result == "JSON RPC", f"Expected 'JSON RPC', got {result}"
        
        # Test error handling
        logger.info("Testing error handling...")
        try:
            with SharedTensorClient("http://localhost:8080") as client:
                client.execute_function("__main__:nonexistent_function", ())
        except RuntimeError as e:
            logger.info(f"✓ Error handling works: {e}")
        
        logger.info("=== All tests passed! ===")
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False
    
    finally:
        # Stop server
        logger.info("Stopping server...")
        server.stop()
        provider.close()


if __name__ == "__main__":
    success = test_server_client()
    if success:
        print("\n🎉 Integration test completed successfully!")
    else:
        print("\n❌ Integration test failed!")
        exit(1)
