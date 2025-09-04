#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for async shared tensor system

Demonstrates long-running task execution without HTTP timeout limitations.
"""

import time
import threading
from shared_tensor.server import SharedTensorServer
from shared_tensor.async_client import AsyncSharedTensorClient
from shared_tensor.async_provider import AsyncSharedTensorProvider
from shared_tensor.async_task import TaskStatus

async_provider = AsyncSharedTensorProvider()


# Define some long-running test functions
@async_provider.share_async(name="slow_computation", wait=False)
def slow_computation(duration: int, message: str = "Computing"):
    """Simulate a slow computation"""
    import time
    print(f"{message}... (will take {duration} seconds)")
    
    for i in range(duration):
        time.sleep(1)
        print(f"  Progress: {i+1}/{duration}")
    
    result = f"Completed: {message} after {duration} seconds"
    print(result)
    return result


@async_provider.share_async(name="cpu_intensive", wait=True)
def cpu_intensive_task(iterations: int):
    """CPU intensive task"""
    import math
    
    result = 0
    for i in range(iterations):
        result += math.sqrt(i * math.pi)
        if i % 100000 == 0:
            print(f"  CPU task progress: {i}/{iterations}")
    
    return f"CPU task completed: {result:.2f}"


@async_provider.share_async(name="data_processing")
def process_large_dataset(size: int, factor: float = 1.5):
    """Simulate processing a large dataset"""
    import time
    import random
    
    print(f"Processing dataset of size {size}")
    
    # Simulate processing
    data = []
    for i in range(size):
        if i % 1000 == 0:
            print(f"  Processing item {i}/{size}")
            time.sleep(0.1)  # Simulate I/O delay
        
        value = random.random() * factor
        data.append(value)
    
    # Calculate statistics
    total = sum(data)
    average = total / len(data)
    
    return {
        "size": size,
        "total": total,
        "average": average,
        "factor": factor
    }


def status_callback(task_info):
    """Callback function to show task progress"""
    elapsed = time.time() - task_info.created_at
    status = task_info.status.value
    print(f"📊 Task {task_info.task_id[:8]}... | Status: {status} | Elapsed: {elapsed:.1f}s")


def test_async_direct_client():
    """Test using async client directly"""
    print("=== Testing Async Direct Client ===")
    
    try:
        with AsyncSharedTensorClient("http://localhost:8080", poll_interval=0.5) as client:
            print("🚀 Submitting slow computation task...")
            
            # Submit a long-running task
            task_id = client.submit_task("__main__:slow_computation", (5, "Direct client test"))
            print(f"✓ Task submitted: {task_id[:8]}...")
            
            # Check initial status
            status = client.get_task_status(task_id)
            print(f"✓ Initial status: {status.status.value}")
            
            # Wait for completion with progress callback
            print("⏳ Waiting for completion...")
            result = client.wait_for_task(task_id, timeout=30, callback=status_callback)
            print(f"✅ Result: {result}")
            
            return True
            
    except Exception as e:
        print(f"❌ Async direct client test failed: {e}")
        return False


def test_async_provider():
    """Test using async provider decorators"""
    print("\n=== Testing Async Provider ===")
    
    try:
        print("📋 Registered async functions:")
        for name, info in async_provider._registered_functions.items():
            wait_default = info.get('async_default_wait', True)
            print(f"  {name} -> {info['function_path']} (wait={wait_default})")
        
        # Test 1: Function with wait=False (returns task ID)
        print("\n🔄 Test 1: Non-blocking execution...")
        task_id = slow_computation(3, "Provider test 1")
        print(f"✓ Got task ID: {task_id[:8]}...")
        
        # Monitor task manually
        print("📈 Monitoring task status...")
        while True:
            status = async_provider.get_task_status(task_id)
            print(f"  Status: {status.status.value}")
            
            if status.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                break
            
            time.sleep(1)
        
        if status.status == TaskStatus.COMPLETED:
            result = async_provider.get_task_result(task_id)
            print(f"✅ Result: {result}")
        else:
            print(f"❌ Task failed or cancelled: {status.error_message}")
        
        # Test 2: Function with wait=True (blocks until complete)
        print("\n🔄 Test 2: Blocking execution...")
        result = cpu_intensive_task(500000)
        print(f"✅ CPU task result: {result}")
        
        # Test 3: Using execute_async with custom options
        print("\n🔄 Test 3: Custom async execution...")
        result = process_large_dataset.execute_async(
            2000, 
            factor=2.0, 
            wait=True, 
            timeout=60,
            callback=status_callback
        )
        print(f"✅ Data processing result: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Async provider test failed: {e}")
        return False


def test_task_management():
    """Test task management features"""
    print("\n=== Testing Task Management ===")
    
    try:
        with AsyncSharedTensorClient("http://localhost:8080") as client:
            # Submit multiple tasks
            print("🚀 Submitting multiple tasks...")
            task_ids = []
            
            for i in range(3):
                task_id = client.submit_task("__main__:slow_computation", (2, f"Batch task {i+1}"))
                task_ids.append(task_id)
                print(f"  Task {i+1}: {task_id[:8]}...")
            
            # List all tasks
            print("\n📋 Listing all tasks...")
            all_tasks = client.list_tasks()
            print(f"✓ Total tasks on server: {len(all_tasks)}")
            
            # List running tasks
            running_tasks = client.list_tasks("running")
            print(f"✓ Running tasks: {len(running_tasks)}")
            
            # Cancel the last task
            if task_ids:
                last_task = task_ids[-1]
                print(f"\n❌ Cancelling task: {last_task[:8]}...")
                cancelled = client.cancel_task(last_task)
                print(f"✓ Cancellation {'successful' if cancelled else 'failed'}")
            
            # Wait for remaining tasks
            print("\n⏳ Waiting for remaining tasks...")
            for task_id in task_ids[:-1]:
                try:
                    result = client.wait_for_task(task_id, timeout=10)
                    print(f"✅ Task {task_id[:8]}... completed: {result}")
                except Exception as e:
                    print(f"❌ Task {task_id[:8]}... failed: {e}")
            
            return True
            
    except Exception as e:
        print(f"❌ Task management test failed: {e}")
        return False


def main():
    """Main test function"""
    print("🧪 Starting Async Shared Tensor Tests")
    print("🔗 Connecting to server at http://localhost:8080")
    print("⚠️  Make sure the server is running with: python3 scripts/run_server.py")
    print("-" * 60)
    
    # Run tests
    tests = [
        test_async_direct_client,
        test_async_provider, 
        test_task_management
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 All {total} async tests passed!")
        return 0
    else:
        print(f"❌ {total - passed} of {total} tests failed!")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
