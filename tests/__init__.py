#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shared Tensor Tests Package

This package contains all tests for the shared-tensor project.

Test Structure:
- unit/: Unit tests for individual components
- integration/: Integration tests for component interactions  
- torch/: PyTorch-specific tests including GPU tensor support
- examples/: Example usage tests and demonstrations
"""

import sys
import os

# Add the parent directory to the path for importing shared_tensor
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

__version__ = "1.0.0"
__author__ = "Shared Tensor Team"

# Test configuration
TEST_CONFIG = {
    'timeout': 30,  # Default test timeout in seconds
    'retry_count': 3,  # Default retry count for flaky tests
    'gpu_tests_enabled': True,  # Enable GPU tests if CUDA available
    'remote_tests_enabled': False,  # Enable remote tests (requires server)
    'verbose': True,  # Verbose test output
}

def configure_test_environment():
    """Configure the test environment"""
    # Set environment variables for testing
    os.environ.setdefault('SHARED_TENSOR_LOG_LEVEL', 'INFO')
    os.environ.setdefault('SHARED_TENSOR_TEST_MODE', '1')
    
    # Configure PyTorch for testing
    try:
        import torch
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            # Set CUDA device for consistent testing
            torch.cuda.set_device(0)
        else:
            print(f"🔧 Test environment: CUDA not available, using CPU only")
    except (ImportError, AttributeError):
        print(f"🔧 Test environment: PyTorch not available")

# Configure environment when module is imported
configure_test_environment()
