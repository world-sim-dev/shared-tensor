"""
Model Serialization Tests

Tests for PyTorch model serialization functionality.
"""

import sys
import os
import unittest

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from tests.pytorch_tests.models import SimpleModel
from tests.utils.serialization_test_utils import test_model_serialization

class TestModelSerialization(unittest.TestCase):
    """Test model serialization functionality"""

    def test_gpu_model_serialization(self):
        """Test GPU model serialization"""
        print(f"\n🔄 Testing GPU model serialization...")
        model = SimpleModel().cuda()
        original_device = next(model.parameters()).device
        print(f"   Model device: {original_device}")
        result = test_model_serialization(model, timeout=60)
        print(f"   Serialized size: {result['serialized_size']} bytes")

        model_info = result['deserialized_info']
        self.assertEqual(str(original_device), model_info['device'])
        self.assertEqual(result['num_parameters'], model_info['num_parameters'])
        self.assertEqual(result['trainable_parameters'], model_info['trainable_parameters'])
        
        print(f"   Deserialized model device: {model_info['device']}")
        print(f"   Model parameters: {model_info['num_parameters']}")
        print(f"   Trainable parameters: {model_info['trainable_parameters']}")
        print(f"   ✅ GPU model serialization successful")


if __name__ == '__main__':
    unittest.main(verbosity=2)
