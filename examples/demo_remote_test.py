#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Demo: How to test remote execution with server

This script demonstrates the proper way to test remote execution functionality.
It shows both manual server management and the functions needed for testing.
"""

import os
import sys
import time
import subprocess
import socket

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Set environment variable for protobuf
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("❌ PyTorch not available")
    exit(1)

from shared_tensor.async_provider import async_provider


# Define remote functions
@async_provider.share_async(name="demo_create_tensor", wait=True)
def demo_create_tensor(shape, value=1.0, device="cpu"):
    """Remote function that creates a PyTorch tensor"""
    import torch
    tensor = torch.full(shape, value, device=device)
    print(f"🔄 Created tensor on {device} with shape {shape}")
    return tensor


@async_provider.share_async(name="demo_tensor_add", wait=True)
def demo_tensor_add(tensor1, tensor2):
    """Remote function that adds two tensors"""
    import torch
    result = tensor1 + tensor2
    print(f"🔄 Added tensors: {tensor1.shape} + {tensor2.shape}")
    return result


def check_server(host="localhost", port=8080):
    """Check if server is running"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


def test_remote_functions():
    """Test the remote functions"""
    print("\n🧪 Testing Remote Functions")
    print("=" * 40)
    
    try:
        # Test 1: Create tensors
        print("1️⃣ Testing tensor creation...")
        tensor1 = demo_create_tensor((2, 3), value=2.0)
        tensor2 = demo_create_tensor((2, 3), value=3.0)
        
        print(f"   Tensor 1: {tensor1}")
        print(f"   Tensor 2: {tensor2}")
        
        # Test 2: Add tensors
        print("\n2️⃣ Testing tensor addition...")
        result = demo_tensor_add(tensor1, tensor2)
        print(f"   Result: {result}")
        
        # Test 3: GPU tensor (if available)
        if torch.cuda.is_available():
            print("\n3️⃣ Testing GPU tensor...")
            gpu_tensor = demo_create_tensor((2, 2), value=5.0, device="cuda:0")
            print(f"   GPU Tensor: {gpu_tensor}")
        else:
            print("\n3️⃣ GPU not available, skipping GPU test")
        
        print("\n✅ All remote function tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Remote function test failed: {e}")
        return False


def main():
    """Main demo function"""
    print("🚀 Shared Tensor Remote Execution Demo")
    print("=" * 50)
    
    # Check if server is running
    if check_server():
        print("✅ Server is already running")
        print("🧪 Running remote execution tests...")
        
        success = test_remote_functions()
        
        if success:
            print("\n🎉 Demo completed successfully!")
        else:
            print("\n💔 Demo failed!")
            
    else:
        print("❌ Server not running!")
        print("\n📋 To run this demo:")
        print("1. Start the server in another terminal:")
        print("   python3 scripts/run_server.py")
        print("\n2. Import remote functions by running this script:")
        print("   python3 demo_remote_test.py")
        print("\n3. Or use the automated test script:")
        print("   ./test_with_server.sh")


if __name__ == "__main__":
    main()
