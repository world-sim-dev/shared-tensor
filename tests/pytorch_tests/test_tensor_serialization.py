"""
GPU Tensor Serialization Tests

Tests for GPU tensor serialization, device migration, and compatibility.
"""

import sys
import os
import unittest

import torch

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from tests.utils.serialization_test_utils import test_tensor_serialization


class TestTensorSerialization(unittest.TestCase):
    """Test GPU tensor serialization functionality"""
    
    def test_basic_gpu_tensor_serialization(self):
        """Test basic GPU tensor serialization/deserialization using subprocess"""
        print(f"\n🔄 Testing basic GPU tensor serialization...")
        
        # Create GPU tensor
        gpu_tensor = torch.randn(10, 10).cuda()
        print(f"   Original tensor device: {gpu_tensor.device}")
        print(f"   Original tensor shape: {gpu_tensor.shape}")
        
        # Use the common test utility
        result = test_tensor_serialization(gpu_tensor, tolerance_places=4)
        
        print(f"   Serialized size: {result['serialized_size']} bytes")
        print(f"   Deserialized device: {result['deserialized_info']['device']}")
        print(f"   Deserialized shape: {result['deserialized_info']['shape']}")
        print(f"   Deserialized dtype: {result['deserialized_info']['dtype']}")
        print(f"   ✅ GPU tensor serialization successful")
    

if __name__ == '__main__':
    torch.cuda.set_device(0)
    unittest.main(verbosity=2)
