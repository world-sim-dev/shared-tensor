#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Serialization Test Utilities

Common utilities for testing tensor and model serialization across processes.
"""

import sys
import os
import json
import subprocess
from typing import Dict, Any, Optional

import torch


class SerializationTestHelper:
    """Helper class for testing serialization across processes"""
    
    def __init__(self):
        """Initialize the helper"""
        self.helper_script_path = os.path.join(
            os.path.dirname(__file__), '..', 'helpers', 'deserialize_helper.py'
        )
    
    def test_tensor_serialization(
        self,
        tensor: torch.Tensor,
        tolerance_places: int = 4,
        timeout: int = 30
    ) -> Dict[str, Any]:
        """
        Test tensor serialization by deserializing in a separate process
        
        Args:
            tensor: The tensor to test serialization for
            tolerance_places: Number of decimal places for statistical comparison
            timeout: Timeout in seconds for subprocess
            
        Returns:
            Dict containing test results and verification info
            
        Raises:
            AssertionError: If serialization test fails
            subprocess.TimeoutExpired: If subprocess times out
            Exception: For other unexpected errors
        """
        from shared_tensor.utils import serialize_result
        
        # Get original tensor properties
        original_device = tensor.device
        original_cpu_data = tensor.cpu().numpy() if tensor.numel() > 0 else None
        original_stats = self._compute_tensor_stats(tensor)
        
        # Serialize the tensor
        serialized = serialize_result(tensor)
        serialized_hex = serialized.hex()
        
        # Run deserialization in subprocess
        result = subprocess.run(
            [sys.executable, self.helper_script_path, serialized_hex],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.path.dirname(self.helper_script_path)
        )
        
        # Check subprocess result
        if result.returncode != 0:
            raise AssertionError(
                f"Helper script failed with return code {result.returncode}:\n"
                f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
            )
        
        # Parse JSON response
        try:
            response = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            raise AssertionError(f"Failed to parse JSON response: {e}\nOutput: {result.stdout}")
        
        # Verify deserialization success
        if not response['success']:
            raise AssertionError(f"Deserialization failed: {response.get('error', 'Unknown error')}")
        
        # Verify tensor properties
        tensor_info = response['tensor_info']
        self._verify_tensor_properties(tensor, tensor_info, original_stats, tolerance_places)
        
        return {
            'original_device': str(original_device),
            'original_shape': list(tensor.shape),
            'original_dtype': str(tensor.dtype),
            'original_stats': original_stats,
            'serialized_size': len(serialized),
            'deserialized_info': tensor_info,
            'verification_passed': True
        }
    
    def test_model_serialization(
        self,
        model: torch.nn.Module,
        test_input: Optional[torch.Tensor] = None,
        timeout: int = 30
    ) -> Dict[str, Any]:
        """
        Test model serialization by deserializing in a separate process
        
        Args:
            model: The model to test serialization for
            test_input: Optional test input for functional verification
            timeout: Timeout in seconds for subprocess
            
        Returns:
            Dict containing test results and verification info
        """
        from shared_tensor.utils import serialize_result
        
        # Get original model properties
        original_device = next(model.parameters()).device if list(model.parameters()) else 'unknown'
        num_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Serialize the model
        serialized = serialize_result(model)
        serialized_hex = serialized.hex()
        
        # Run deserialization in subprocess
        result = subprocess.run(
            [sys.executable, self.helper_script_path, serialized_hex],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.path.dirname(self.helper_script_path)
        )
        
        # Check subprocess result
        if result.returncode != 0:
            raise AssertionError(
                f"Helper script failed with return code {result.returncode}:\n"
                f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
            )
        
        # Parse JSON response
        try:
            response = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            raise AssertionError(f"Failed to parse JSON response: {e}\nOutput: {result.stdout}")
        
        # Verify deserialization success
        if not response['success']:
            raise AssertionError(f"Deserialization failed: {response.get('error', 'Unknown error')}")
        
        # Verify model properties
        model_info = response['tensor_info']
        if model_info['num_parameters'] != num_params:
            raise AssertionError(f"Parameter count mismatch: {num_params} != {model_info['num_parameters']}")
        
        if model_info['trainable_parameters'] != trainable_params:
            raise AssertionError(f"Trainable parameter count mismatch: {trainable_params} != {model_info['trainable_parameters']}")
        
        return {
            'original_device': str(original_device),
            'num_parameters': num_params,
            'trainable_parameters': trainable_params,
            'serialized_size': len(serialized),
            'deserialized_info': model_info,
            'verification_passed': True
        }
    
    def _compute_tensor_stats(self, tensor: torch.Tensor) -> Dict[str, float]:
        """Compute statistical properties of a tensor"""
        if tensor.numel() == 0:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
        
        cpu_tensor = tensor.cpu().float()
        return {
            'mean': float(cpu_tensor.mean()),
            'std': float(cpu_tensor.std()),
            'min': float(cpu_tensor.min()),
            'max': float(cpu_tensor.max())
        }
    
    def _verify_tensor_properties(
        self,
        original_tensor: torch.Tensor,
        tensor_info: Dict[str, Any],
        original_stats: Dict[str, float],
        tolerance_places: int
    ):
        """Verify that deserialized tensor properties match the original"""
        # Basic properties
        if str(original_tensor.device) != tensor_info['device']:
            raise AssertionError(f"Device mismatch: {original_tensor.device} != {tensor_info['device']}")
        
        if list(original_tensor.shape) != tensor_info['shape']:
            raise AssertionError(f"Shape mismatch: {original_tensor.shape} != {tensor_info['shape']}")
        
        if str(original_tensor.dtype) != tensor_info['dtype']:
            raise AssertionError(f"Dtype mismatch: {original_tensor.dtype} != {tensor_info['dtype']}")
        
        if original_tensor.is_cuda != tensor_info['is_cuda']:
            raise AssertionError(f"CUDA flag mismatch: {original_tensor.is_cuda} != {tensor_info['is_cuda']}")
        
        # Element count and memory size
        if original_tensor.numel() != tensor_info['element_count']:
            raise AssertionError(f"Element count mismatch: {original_tensor.numel()} != {tensor_info['element_count']}")
        
        expected_memory = original_tensor.element_size() * original_tensor.numel()
        if expected_memory != tensor_info['memory_size_bytes']:
            raise AssertionError(f"Memory size mismatch: {expected_memory} != {tensor_info['memory_size_bytes']}")
        
        # Statistical properties (with tolerance)
        for stat_name in ['mean', 'std', 'min', 'max']:
            original_val = original_stats[stat_name]
            deserialized_val = tensor_info[stat_name]
            tolerance = 10 ** (-tolerance_places)
            
            if abs(original_val - deserialized_val) > tolerance:
                raise AssertionError(
                    f"Statistical property '{stat_name}' mismatch: "
                    f"{original_val} != {deserialized_val} (tolerance: {tolerance})"
                )


# Global instance for convenience
serialization_test_helper = SerializationTestHelper()


def test_tensor_serialization(tensor: torch.Tensor, **kwargs) -> Dict[str, Any]:
    """
    Convenience function to test tensor serialization
    
    Args:
        tensor: The tensor to test
        **kwargs: Additional arguments for SerializationTestHelper.test_tensor_serialization
        
    Returns:
        Test results dictionary
    """
    return serialization_test_helper.test_tensor_serialization(tensor, **kwargs)


def test_model_serialization(model: torch.nn.Module, **kwargs) -> Dict[str, Any]:
    """
    Convenience function to test model serialization
    
    Args:
        model: The model to test
        **kwargs: Additional arguments for SerializationTestHelper.test_model_serialization
        
    Returns:
        Test results dictionary
    """
    return serialization_test_helper.test_model_serialization(model, **kwargs)


