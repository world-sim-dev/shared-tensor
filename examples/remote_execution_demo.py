#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Remote Execution Demo

This example demonstrates how to use remote execution with PyTorch tensors.
It shows the complete workflow of defining remote functions, starting a server,
and executing functions remotely.

Usage:
1. Start server: python3 scripts/run_server.py
2. Run this demo: python3 examples/remote_execution_demo.py
"""

import sys
import os
import time

# Add parent directory to path  
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

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


# Define remote functions for demonstration
@async_provider.share_async(name="demo_create_tensor", wait=True)
def demo_create_tensor(shape, value=1.0, device="cpu"):
    """Create a tensor remotely"""
    import torch
    tensor = torch.full(shape, value, device=device)
    print(f"🔄 Created tensor on {device} with shape {shape}")
    return tensor


@async_provider.share_async(name="demo_add_tensors", wait=True)  
def demo_add_tensors(a, b):
    """Add two tensors remotely"""
    import torch
    result = a + b
    print(f"🔄 Added tensors: {result}")
    return result


@async_provider.share_async(name="demo_matrix_multiply", wait=True)
def demo_matrix_multiply(a, b):
    """Multiply two matrices remotely"""
    import torch
    result = torch.matmul(a, b)
    print(f"🔄 Matrix multiplication result shape: {result.shape}")
    return result


@async_provider.share_async(name="demo_gpu_computation", wait=True)
def demo_gpu_computation(size=1000):
    """Perform GPU computation if available"""
    import torch
    
    if torch.cuda.is_available():
        device = "cuda:0"
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("💻 Using CPU")
    
    # Create large tensors
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    # Perform computation
    start_time = time.time()
    result = torch.matmul(a, b)
    end_time = time.time()
    
    print(f"⏱️  Computation time: {end_time - start_time:.3f}s")
    print(f"📊 Result shape: {result.shape}")
    print(f"📍 Result device: {result.device}")
    
    return {
        'device': str(result.device),
        'shape': list(result.shape),
        'computation_time': end_time - start_time,
        'mean': result.mean().item(),
        'std': result.std().item()
    }


def check_server_connection():
    """Check if server is available"""
    try:
        import requests
        response = requests.get("http://localhost:8080", timeout=5)
        return True
    except Exception:
        return False


def demo_basic_tensor_operations():
    """Demonstrate basic tensor operations"""
    print("1️⃣ Basic Tensor Operations")
    print("-" * 30)
    
    try:
        # Create tensors
        print("Creating tensors...")
        tensor1 = demo_create_tensor((3, 3), 2.0)
        tensor2 = demo_create_tensor((3, 3), 3.0)
        
        print(f"Tensor 1:\n{tensor1}")
        print(f"Tensor 2:\n{tensor2}")
        
        # Add tensors
        print("\nAdding tensors...")
        result = demo_add_tensors(tensor1, tensor2)
        print(f"Sum:\n{result}")
        
        # Verify result
        expected = torch.full((3, 3), 5.0)
        if torch.allclose(result, expected):
            print("✅ Basic operations successful!")
            return True
        else:
            print("❌ Unexpected result!")
            return False
            
    except Exception as e:
        print(f"❌ Basic operations failed: {e}")
        return False


def demo_matrix_operations():
    """Demonstrate matrix operations"""
    print("\n2️⃣ Matrix Operations")
    print("-" * 30)
    
    try:
        # Create matrices
        print("Creating matrices...")
        matrix_a = demo_create_tensor((4, 3), 1.0)
        matrix_b = demo_create_tensor((3, 2), 2.0)
        
        print(f"Matrix A shape: {matrix_a.shape}")
        print(f"Matrix B shape: {matrix_b.shape}")
        
        # Matrix multiplication
        print("Performing matrix multiplication...")
        result = demo_matrix_multiply(matrix_a, matrix_b)
        
        print(f"Result shape: {result.shape}")
        print(f"Result:\n{result}")
        
        # Verify dimensions
        if result.shape == (4, 2):
            print("✅ Matrix operations successful!")
            return True
        else:
            print("❌ Unexpected result shape!")
            return False
            
    except Exception as e:
        print(f"❌ Matrix operations failed: {e}")
        return False


def demo_gpu_performance():
    """Demonstrate GPU performance if available"""
    print("\n3️⃣ GPU Performance Test")
    print("-" * 30)
    
    try:
        # Test different sizes
        sizes = [100, 500, 1000]
        
        for size in sizes:
            print(f"\nTesting {size}x{size} matrix multiplication...")
            result = demo_gpu_computation(size)
            
            print(f"Device: {result['device']}")
            print(f"Time: {result['computation_time']:.3f}s")
            print(f"Mean: {result['mean']:.6f}")
            print(f"Std: {result['std']:.6f}")
        
        print("✅ GPU performance test completed!")
        return True
        
    except Exception as e:
        print(f"❌ GPU performance test failed: {e}")
        return False


def main():
    """Main demo function"""
    print("🚀 Remote Execution Demo")
    print("=" * 50)
    
    # Check server connection
    if not check_server_connection():
        print("❌ Server not available!")
        print("\n📋 To run this demo:")
        print("1. Start the server:")
        print("   python3 scripts/run_server.py")
        print("\n2. Run this demo in another terminal:")
        print("   python3 examples/remote_execution_demo.py")
        return False
    
    print("✅ Server is available")
    
    # Show registered functions
    print(f"\n📝 Registered remote functions:")
    for func_name, func_info in async_provider._registered_functions.items():
        print(f"  {func_name} -> {func_info['function_path']}")
    
    # Run demos
    success_count = 0
    total_demos = 3
    
    if demo_basic_tensor_operations():
        success_count += 1
    
    if demo_matrix_operations():
        success_count += 1
    
    if demo_gpu_performance():
        success_count += 1
    
    # Summary
    print(f"\n🎉 Demo Summary")
    print("=" * 50)
    print(f"Successful demos: {success_count}/{total_demos}")
    
    if success_count == total_demos:
        print("🎊 All demos completed successfully!")
        print("Remote execution is working perfectly!")
    else:
        print("⚠️  Some demos failed. Check the server and network connection.")
    
    return success_count == total_demos


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

