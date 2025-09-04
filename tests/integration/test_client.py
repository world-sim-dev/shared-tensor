#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Client test script for shared tensor system
"""

from shared_tensor import SharedTensorClient
from shared_tensor.provider import SharedTensorProvider

provider = SharedTensorProvider()

# Define test functions using the provider decorator
@provider.share(name="add_test")
def add_numbers(a, b):
    """Simple addition function"""
    return a + b

@provider.share(name="string_test")
def process_string(text, uppercase=False):
    """String processing function"""
    result = text.strip()
    if uppercase:
        result = result.upper()
    return result

@provider.share(name="list_test")
def calculate_sum(numbers):
    """Calculate sum of a list"""
    return sum(numbers)

def test_direct_client():
    """Test using direct JSON-RPC client"""
    print("=== Testing Direct JSON-RPC Client ===")
    
    try:
        with SharedTensorClient("http://localhost:8080") as client:
            # Test ping
            if client.ping():
                print("✓ Server ping successful")
            else:
                print("✗ Server ping failed")
                return False
            
            # Test server info
            server_info = client.get_server_info()
            print(f"✓ Server: {server_info['server']} v{server_info.get('version', 'unknown')}")
            print(f"  Uptime: {server_info.get('uptime', 0):.2f} seconds")
            
            # Test function execution using built-in functions
            print("\n📋 Testing built-in function execution:")
            
            # Test os.path.join
            result = client.execute_function("os.path:join", ("/home", "user", "test.txt"))
            print(f"✓ os.path.join('/home', 'user', 'test.txt') = {result}")
            
            # Test len function (this might not work as len is built-in)
            try:
                result = client.execute_function("builtins:len", ("hello",))
                print(f"✓ len('hello') = {result}")
            except Exception as e:
                print(f"⚠️  len test expected to fail: {e}")
            
            return True
            
    except Exception as e:
        print(f"❌ Direct client test failed: {e}")
        return False

def test_provider_decorator():
    """Test using provider decorators"""
    print("\n=== Testing Provider Decorators ===")
    
    try:
        # Show registered functions
        print("📋 Registered functions:")
        for name, info in provider._registered_functions.items():
            print(f"  {name} -> {info['function_path']}")
        
        print("\n🔧 Testing remote execution via decorators:")
        
        # Test simple addition
        result = add_numbers(10, 20)
        print(f"✓ add_numbers(10, 20) = {result}")
        
        # Test string processing
        result = process_string("  hello world  ", uppercase=True)
        print(f"✓ process_string('  hello world  ', uppercase=True) = '{result}'")
        
        # Test list sum
        result = calculate_sum([1, 2, 3, 4, 5])
        print(f"✓ calculate_sum([1, 2, 3, 4, 5]) = {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Provider decorator test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Starting Client Tests")
    print("🔗 Connecting to server at http://localhost:8080")
    print("-" * 50)
    
    # Test direct client
    direct_ok = test_direct_client()
    
    # Test provider decorators
    provider_ok = test_provider_decorator()
    
    print("\n" + "=" * 50)
    if direct_ok and provider_ok:
        print("🎉 All client tests passed!")
        return 0
    else:
        print("❌ Some client tests failed!")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
