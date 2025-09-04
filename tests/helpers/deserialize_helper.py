"""
Deserialization Helper Script

This script is used to deserialize PyTorch tensors in a separate process
to avoid CUDA multiprocessing issues during testing.
"""

import sys
import json
import os
import traceback

import torch
import pickle
import numpy as np

# Set environment variables before importing torch
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def deserialize_and_analyze(serialized_hex):
    """
    Deserialize tensor from hex string and return analysis
    
    Args:
        serialized_hex: Hex-encoded serialized tensor data
        
    Returns:
        Dict with analysis results
    """
    try:
        # Convert hex to bytes and deserialize
        serialized_bytes = bytes.fromhex(serialized_hex)
        tensor = pickle.loads(serialized_bytes)
        
        # Analyze the tensor
        result = {
            'success': True,
            'type': str(type(tensor).__name__),
            'tensor_info': {}
        }
        
        if isinstance(tensor, torch.Tensor):
            # Move to CPU for analysis to avoid device issues
            cpu_tensor = tensor.cpu()
            
            result['tensor_info'] = {
                'device': str(tensor.device),
                'dtype': str(tensor.dtype),
                'shape': list(tensor.shape),
                'requires_grad': tensor.requires_grad,
                'is_cuda': tensor.is_cuda,
                'element_count': tensor.numel(),
                'memory_size_bytes': tensor.element_size() * tensor.numel(),
                'data_hash': hash(cpu_tensor.numpy().tobytes()),
                'mean': float(cpu_tensor.float().mean()) if tensor.numel() > 0 else 0.0,
                'std': float(cpu_tensor.float().std()) if tensor.numel() > 0 else 0.0,
                'min': float(cpu_tensor.float().min()) if tensor.numel() > 0 else 0.0,
                'max': float(cpu_tensor.float().max()) if tensor.numel() > 0 else 0.0
            }
        elif isinstance(tensor, torch.nn.Module):
            result['tensor_info'] = {
                'model_type': str(type(tensor).__name__),
                'num_parameters': sum(p.numel() for p in tensor.parameters()),
                'trainable_parameters': sum(p.numel() for p in tensor.parameters() if p.requires_grad),
                'device': str(next(tensor.parameters()).device) if list(tensor.parameters()) else 'unknown'
            }
        else:
            # Handle other types
            result['tensor_info'] = {
                'value': str(tensor),
                'size_bytes': sys.getsizeof(tensor)
            }
            
        return result
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__,
            'traceback': traceback.format_exc()
        }


def main():
    """Main function to handle command line input"""
    if len(sys.argv) != 2:
        result = {
            'success': False,
            'error': 'Usage: python deserialize_helper.py <hex_data>',
            'error_type': 'UsageError'
        }
        print(json.dumps(result))
        sys.exit(1)
    
    serialized_hex = sys.argv[1]
    
    # Validate hex input
    try:
        bytes.fromhex(serialized_hex)
    except ValueError as e:
        result = {
            'success': False,
            'error': f'Invalid hex data: {e}',
            'error_type': 'ValueError'
        }
        print(json.dumps(result))
        sys.exit(1)
    
    # Perform deserialization and analysis
    result = deserialize_and_analyze(serialized_hex)
    
    # Output JSON result
    print(json.dumps(result, indent=2))
    
    # Exit with appropriate code
    sys.exit(0 if result['success'] else 1)


if __name__ == '__main__':
    main()
