#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test Structure Validation

Quick validation script to ensure the new test structure is working correctly.
"""

import sys
import os
import unittest

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestStructureValidation(unittest.TestCase):
    """Validate the test structure"""
    
    def test_import_test_modules(self):
        """Test that all test modules can be imported"""
        print("\n🔍 Testing module imports...")
        
        # Test torch modules
        try:
            from tests.pytorch_tests.test_models import SimpleModel, create_simple_model
            print("   ✅ tests.pytorch_tests.test_models")
        except ImportError as e:
            print(f"   ❌ tests.pytorch_tests.test_models: {e}")
        
        # Test that we can import tests
        try:
            from tests.pytorch_tests import test_tensor_serialization
            print("   ✅ tests.pytorch_tests.test_tensor_serialization")
        except ImportError as e:
            print(f"   ❌ tests.pytorch_tests.test_tensor_serialization: {e}")
        
        try:
            from tests.pytorch_tests import test_gpu_tensors
            print("   ✅ tests.pytorch_tests.test_gpu_tensors")
        except ImportError as e:
            print(f"   ❌ tests.pytorch_tests.test_gpu_tensors: {e}")
        
        try:
            from tests.pytorch_tests import test_model_serialization
            print("   ✅ tests.pytorch_tests.test_model_serialization")
        except ImportError as e:
            print(f"   ❌ tests.pytorch_tests.test_model_serialization: {e}")
        
        try:
            from tests.pytorch_tests import test_remote_execution
            print("   ✅ tests.pytorch_tests.test_remote_execution")
        except ImportError as e:
            print(f"   ❌ tests.pytorch_tests.test_remote_execution: {e}")
    
    def test_shared_tensor_imports(self):
        """Test that shared_tensor can be imported from tests"""
        print("\n🔍 Testing shared_tensor imports...")
        
        try:
            from shared_tensor.utils import serialize_result
            print("   ✅ shared_tensor.utils")
        except ImportError as e:
            print(f"   ❌ shared_tensor.utils: {e}")
        
        try:
            from shared_tensor.async_provider import async_provider
            print("   ✅ shared_tensor.async_provider")
        except ImportError as e:
            print(f"   ❌ shared_tensor.async_provider: {e}")
    
    def test_torch_functionality(self):
        """Test basic torch functionality if available"""
        print("\n🔍 Testing PyTorch functionality...")
        
        try:
            import torch
            print("   ✅ PyTorch available")
            
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                print(f"   ✅ CUDA available ({torch.cuda.device_count()} devices)")
            else:
                print("   ⚠️  CUDA not available")
            
            # Test model creation
            from tests.pytorch_tests.test_models import create_simple_model
            model = create_simple_model()
            test_input = torch.randn(1, 10)
            output = model(test_input)
            print("   ✅ Model creation and execution")
            
        except ImportError:
            print("   ⚠️  PyTorch not available")
    
    def test_directory_structure(self):
        """Test that directory structure is correct"""
        print("\n🔍 Testing directory structure...")
        
        test_dir = os.path.dirname(__file__)
        
        # Check subdirectories exist
        subdirs = ['unit', 'integration', 'pytorch_tests']
        for subdir in subdirs:
            path = os.path.join(test_dir, subdir)
            if os.path.exists(path):
                print(f"   ✅ {subdir}/ directory exists")
            else:
                print(f"   ❌ {subdir}/ directory missing")
        
        # Check __init__.py files
        for subdir in subdirs:
            init_file = os.path.join(test_dir, subdir, '__init__.py')
            if os.path.exists(init_file):
                print(f"   ✅ {subdir}/__init__.py exists")
            else:
                print(f"   ❌ {subdir}/__init__.py missing")


def run_structure_test():
    """Run the structure validation test"""
    print("🚀 Test Structure Validation")
    print("=" * 50)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestStructureValidation)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("🎉 Test structure validation passed!")
        return True
    else:
        print("❌ Test structure validation failed!")
        return False


if __name__ == '__main__':
    success = run_structure_test()
    sys.exit(0 if success else 1)
