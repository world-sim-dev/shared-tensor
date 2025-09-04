#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example models module to demonstrate function path generation

This file shows how to create shareable PyTorch models using the provider pattern.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not available")
    exit(1)

from shared_tensor.provider import SharedTensorProvider

# Create a provider instance for this module
model_provider = SharedTensorProvider()


@model_provider.share(name="create_simple_model")
def create_simple_model(input_size: int, hidden_size: int, output_size: int):
    """Create a simple neural network model"""
    
    class SimpleNet(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(SimpleNet, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, output_size)
        
        def forward(self, x):
            out = self.fc1(x)
            out = self.relu(out)
            out = self.fc2(out)
            return out
    
    return SimpleNet(input_size, hidden_size, output_size)


@model_provider.share(name="create_resnet_block")  
def create_resnet_block(in_channels: int, out_channels: int):
    """Create a ResNet basic block"""
    
    class BasicBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super(BasicBlock, self).__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU()
            
            # Shortcut connection
            if in_channels != out_channels:
                self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
            else:
                self.shortcut = nn.Identity()
        
        def forward(self, x):
            residual = self.shortcut(x)
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += residual
            return self.relu(out)
    
    return BasicBlock(in_channels, out_channels)


class ModelFactory:
    """Example class with shared methods"""
    
    @staticmethod
    @model_provider.share(name="factory_create_linear")
    def create_linear_model(input_dim: int, output_dim: int):
        """Static method to create a linear model"""
        return nn.Linear(input_dim, output_dim)
    
    @model_provider.share(name="factory_create_conv")
    def create_conv_model(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        """Instance method to create a conv model"""
        return nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)


# Example usage and path inspection
if __name__ == "__main__":
    print("Function paths in shared_tensor.models module:")
    
    for func_name, func_info in model_provider._registered_functions.items():
        function_path = func_info['function_path']
        print(f"  {func_name} -> {function_path}")
    
    # Test creating an instance to see instance method paths
    factory = ModelFactory()
    
    print(f"\nThese paths can be imported remotely using:")
    print(f"  from shared_tensor.utils import import_function_from_path")
    print(f"  func = import_function_from_path('shared_tensor.models:create_simple_model')")
    print(f"  model = func(784, 128, 10)")
