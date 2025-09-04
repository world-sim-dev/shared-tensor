#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unit tests for function path resolution

Tests the _get_function_path method in provider classes to ensure
proper handling of __main__ module and correct path generation.
"""

import sys
import os
import unittest

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Set environment variable for protobuf
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

from shared_tensor.provider import SharedTensorProvider
from shared_tensor.async_provider import AsyncSharedTensorProvider


class TestFunctionPathResolution(unittest.TestCase):
    """Test function path resolution for different scenarios"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sync_provider = SharedTensorProvider()
        self.async_provider = AsyncSharedTensorProvider()
    
    def test_sync_provider_main_module_resolution(self):
        """Test sync provider resolves __main__ module correctly"""
        
        @self.sync_provider.share(name="test_sync_function")
        def test_sync_function(x, y):
            """Test function for sync provider"""
            return x + y
        
        # Get function info
        func_info = self.sync_provider._registered_functions.get("test_sync_function")
        self.assertIsNotNone(func_info, "Function should be registered")
        
        function_path = func_info['function_path']
        print(f"Sync function path: {function_path}")
        
        # Check that __main__ is converted to actual module name
        self.assertNotIn("__main__", function_path, "Function path should not contain __main__")
        self.assertIn("test_function_path_resolution", function_path, "Should contain actual module name")
        self.assertIn("test_sync_function", function_path, "Should contain function name")
    
    def test_async_provider_main_module_resolution(self):
        """Test async provider resolves __main__ module correctly"""
        
        @self.async_provider.share(name="test_async_function", wait=True)
        def test_async_function(x, y):
            """Test function for async provider"""
            return x + y
        
        # Get function info
        func_info = self.async_provider._registered_functions.get("test_async_function")
        self.assertIsNotNone(func_info, "Function should be registered")
        
        function_path = func_info['function_path']
        print(f"Async function path: {function_path}")
        
        # Check that __main__ is converted to actual module name
        self.assertNotIn("__main__", function_path, "Function path should not contain __main__")
        self.assertIn("test_function_path_resolution", function_path, "Should contain actual module name")
        self.assertIn("test_async_function", function_path, "Should contain function name")
    
    def test_function_path_format(self):
        """Test that function paths follow the correct format"""
        
        @self.sync_provider.share(name="format_test_function")
        def format_test_function():
            """Test function for format validation"""
            return "test"
        
        func_info = self.sync_provider._registered_functions.get("format_test_function")
        function_path = func_info['function_path']
        
        # Should be in format "module:function"
        self.assertIn(":", function_path, "Function path should contain colon separator")
        
        module_part, function_part = function_path.split(":", 1)
        self.assertGreater(len(module_part), 0, "Module part should not be empty")
        self.assertEqual(function_part, "format_test_function", "Function part should match function name")
    
    def test_nested_function_handling(self):
        """Test handling of nested functions"""
        
        def outer_function():
            @self.sync_provider.share(name="nested_test_function")
            def nested_function():
                return "nested"
            return nested_function
        
        # This should work without issues
        nested_func = outer_function()
        
        func_info = self.sync_provider._registered_functions.get("nested_test_function")
        self.assertIsNotNone(func_info, "Nested function should be registered")
        
        function_path = func_info['function_path']
        print(f"Nested function path: {function_path}")
    
    def test_class_method_handling(self):
        """Test handling of class methods"""
        
        class TestClass:
            @self.sync_provider.share(name="method_test_function")
            def test_method(self, x):
                return x * 2
        
        func_info = self.sync_provider._registered_functions.get("method_test_function")
        self.assertIsNotNone(func_info, "Class method should be registered")
        
        function_path = func_info['function_path']
        print(f"Class method path: {function_path}")


def main():
    """Run the tests"""
    print("🧪 Testing Function Path Resolution")
    print("=" * 50)
    
    # Run tests
    unittest.main(verbosity=2)


if __name__ == "__main__":
    main()

