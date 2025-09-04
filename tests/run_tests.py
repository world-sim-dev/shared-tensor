#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified Test Runner for Shared Tensor

This script provides a unified interface to run all tests in the shared-tensor project.
"""

import sys
import os
import argparse
import unittest
import subprocess
import importlib.util
import time
import signal
import socket
from pathlib import Path

# Set environment variable to fix protobuf issues
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestServerManager:
    """Manages test server lifecycle for remote execution tests"""
    
    def __init__(self, host="localhost", port=8080):
        self.host = host
        self.port = port
        self.server_process = None
        self.is_running = False
    
    def is_server_running(self):
        """Check if server is responding"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((self.host, self.port))
            sock.close()
            return result == 0
        except Exception:
            return False
    
    def start_server(self, timeout=10):
        """Start the test server"""
        if self.is_server_running():
            print(f"✅ Server already running on {self.host}:{self.port}")
            self.is_running = True
            return True
        
        print(f"🚀 Starting server on {self.host}:{self.port}...")
        
        # Start server process
        script_path = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'run_server.py')
        env = os.environ.copy()
        env['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
        
        try:
            self.server_process = subprocess.Popen(
                [sys.executable, script_path, '--host', self.host, '--port', str(self.port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env
            )
        except Exception as e:
            print(f"❌ Failed to start server: {e}")
            return False
        
        # Wait for server to be ready
        print("⏳ Waiting for server to be ready...")
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_server_running():
                print("✅ Server is ready!")
                self.is_running = True
                return True
            time.sleep(0.5)
        
        print(f"❌ Server failed to start within {timeout}s")
        self.stop_server()
        return False
    
    def stop_server(self):
        """Stop the test server"""
        if self.server_process:
            print("⏹️  Stopping server...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("🔨 Force killing server...")
                self.server_process.kill()
                self.server_process.wait()
            
            self.server_process = None
            print("✅ Server stopped")
        
        self.is_running = False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_server()


# Test configuration
TEST_CONFIG = {
    'unit_tests': [
        'tests.unit.test_function_path_resolution',
    ],
    'integration_tests': [
        'tests.integration.test_async_system',
        'tests.integration.test_client', 
        'tests.integration.test_jsonrpc_integration',
    ],
    'torch_tests': [
        'tests.pytorch_tests.test_tensor_serialization',
        'tests.pytorch_tests.test_model_serialization',
    ],

}


def check_dependencies():
    """Check if required dependencies are available"""
    dependencies = {
        'torch': False,
        'cuda': False,
        'server': False,
    }
    
    # Check PyTorch
    try:
        import torch
        dependencies['torch'] = True
        
        # Check CUDA
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            dependencies['cuda'] = True
    except ImportError:
        pass
    
    # Check if server is running (simple test)
    try:
        from shared_tensor.async_provider import async_provider
        
        @async_provider.share(name="test_server_check", wait=True)
        def test_server():
            return "server_available"
        
        result = test_server()
        if result == "server_available":
            dependencies['server'] = True
    except:
        pass
    
    return dependencies


def print_environment_info():
    """Print environment information"""
    print("🔧 Test Environment Information")
    print("=" * 50)
    
    deps = check_dependencies()
    
    print(f"🐍 Python: {sys.version.split()[0]}")
    print(f"📦 PyTorch: {'✅' if deps['torch'] else '❌'}")
    
    if deps['torch']:
        import torch
        version = getattr(torch, '__version__', 'unknown')
        print(f"   Version: {version}")
        print(f"🎮 CUDA: {'✅' if deps['cuda'] else '❌'}")
        
        if deps['cuda']:
            device_count = torch.cuda.device_count()
            print(f"   Devices: {device_count}")
            for i in range(device_count):
                name = torch.cuda.get_device_name(i)
                print(f"   Device {i}: {name}")
    
    print(f"🌐 Server: {'✅' if deps['server'] else '❌'}")
    print("=" * 50)
    
    return deps


def run_test_module(module_name, verbose=False):
    """Run a specific test module"""
    try:
        # Set up proper Python path
        original_path = sys.path[:]
        project_root = os.path.join(os.path.dirname(__file__), '..')
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        # Try to import the module
        module = importlib.import_module(module_name)
        
        # Create test suite
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(module)
        
        # Run tests
        runner = unittest.TextTestRunner(
            verbosity=2 if verbose else 1,
            stream=sys.stdout,
            buffer=True
        )
        
        result = runner.run(suite)
        
        result_success = result.wasSuccessful()
        
        # Restore original path
        sys.path[:] = original_path
        
        return result_success, result.testsRun, len(result.failures), len(result.errors)
        
    except ImportError as e:
        print(f"❌ Could not import {module_name}: {e}")
        # Restore original path
        sys.path[:] = original_path
        return False, 0, 0, 1
    except Exception as e:
        print(f"❌ Error running {module_name}: {e}")
        # Restore original path  
        sys.path[:] = original_path
        return False, 0, 0, 1


def run_test_category(category, test_modules, verbose=False, dependencies=None):
    """Run all tests in a category"""
    print(f"\n🧪 Running {category.upper()} Tests")
    print("-" * 40)
    
    total_tests = 0
    total_failures = 0
    total_errors = 0
    successful_modules = 0
    
    for module_name in test_modules:
        print(f"\n📋 {module_name}")
        
        # Check dependencies for specific modules
        if 'torch' in module_name and not dependencies.get('torch'):
            print(f"⚠️  Skipping {module_name} - PyTorch not available")
            continue
        
        if 'gpu' in module_name and not dependencies.get('cuda'):
            print(f"⚠️  Skipping {module_name} - CUDA not available")
            continue
        
        if 'remote' in module_name and not dependencies.get('server'):
            print(f"⚠️  Skipping {module_name} - Server not available")
            continue
        
        success, tests, failures, errors = run_test_module(module_name, verbose)
        
        total_tests += tests
        total_failures += failures
        total_errors += errors
        
        if success:
            successful_modules += 1
            print(f"✅ {module_name} - {tests} tests passed")
        else:
            print(f"❌ {module_name} - {failures} failures, {errors} errors")
    
    print(f"\n📊 {category.upper()} Summary:")
    print(f"   Modules: {successful_modules}/{len(test_modules)} successful")
    print(f"   Tests: {total_tests} total, {total_failures} failures, {total_errors} errors")
    
    return total_tests, total_failures, total_errors


def run_all_tests(categories=None, verbose=False):
    """Run all tests or specific categories"""
    print("🚀 Shared Tensor Test Suite")
    print("=" * 50)
    
    # Print environment info
    dependencies = print_environment_info()
    
    # Determine which categories to run
    if categories is None:
        categories = list(TEST_CONFIG.keys())
    
    # Run tests by category
    grand_total_tests = 0
    grand_total_failures = 0
    grand_total_errors = 0
    
    for category in categories:
        if category in TEST_CONFIG:
            tests, failures, errors = run_test_category(
                category, TEST_CONFIG[category], verbose, dependencies
            )
            grand_total_tests += tests
            grand_total_failures += failures
            grand_total_errors += errors
        else:
            print(f"⚠️  Unknown test category: {category}")
    
    # Final summary
    print("\n" + "=" * 50)
    print("🏁 Final Test Summary")
    print("=" * 50)
    print(f"📊 Total Tests: {grand_total_tests}")
    print(f"❌ Failures: {grand_total_failures}")
    print(f"💥 Errors: {grand_total_errors}")
    
    if grand_total_failures == 0 and grand_total_errors == 0:
        print("🎉 All tests passed!")
        return True
    else:
        print("💔 Some tests failed")
        return False


def run_specific_test(test_file):
    """Run a specific test file"""
    print(f"🧪 Running specific test: {test_file}")
    
    if test_file.endswith('.py'):
        # Run as Python script with proper environment
        env = os.environ.copy()
        env['PYTHONPATH'] = os.path.join(os.path.dirname(__file__), '..')
        env['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
        
        result = subprocess.run([sys.executable, test_file], 
                              capture_output=True, text=True, env=env)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result.returncode == 0
    else:
        # Try to import as module
        success, tests, failures, errors = run_test_module(test_file, verbose=True)
        return success


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Shared Tensor Test Runner")
    
    parser.add_argument(
        '--category', '-c',
        choices=['unit', 'integration', 'torch', 'all'],
        default='all',
        help='Test category to run'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    parser.add_argument(
        '--test', '-t',
        help='Run specific test file or module'
    )
    
    parser.add_argument(
        '--env-info', '-e',
        action='store_true',
        help='Show environment information only'
    )
    
    parser.add_argument(
        '--torch-only',
        action='store_true',
        help='Run only PyTorch-related tests'
    )
    
    parser.add_argument(
        '--with-server',
        action='store_true',
        help='Start server automatically for remote execution tests'
    )
    
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Skip GPU tests'
    )
    
    parser.add_argument(
        '--no-remote',
        action='store_true',
        help='Skip remote execution tests'
    )
    
    args = parser.parse_args()
    
    # Environment info only
    if args.env_info:
        print_environment_info()
        return
    
    # Specific test
    if args.test:
        # Check if this is a remote execution test
        if 'remote_execution' in args.test and args.with_server:
            with TestServerManager() as server:
                if server.start_server():
                    success = run_specific_test(args.test)
                else:
                    print("❌ Failed to start server for remote execution test")
                    success = False
        else:
            success = run_specific_test(args.test)
        sys.exit(0 if success else 1)
    
    # Determine categories
    if args.torch_only:
        categories = ['torch_tests']
    elif args.category == 'all':
        categories = None
    else:
        # Map category names to internal test config keys
        category_map = {
            'unit': 'unit_tests',
            'integration': 'integration_tests', 
            'torch': 'torch_tests'
        }
        categories = [category_map.get(args.category, f'{args.category}_tests')]
    
    # Run tests with optional server management
    if args.with_server and (categories is None or 'torch_tests' in categories):
        print("🖥️  Remote execution tests enabled - starting server...")
        with TestServerManager() as server:
            if server.start_server():
                success = run_all_tests(categories, args.verbose)
            else:
                print("❌ Failed to start server for remote execution tests")
                success = False
    else:
        success = run_all_tests(categories, args.verbose)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
