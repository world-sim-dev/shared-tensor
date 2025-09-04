"""
Example usage of SharedTensor library

This example demonstrates how to use the share decorator to enable
remote execution of functions with automatic serialization/deserialization.
"""

import sys
import os
import torch
import torch.nn as nn

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

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared_tensor.provider import SharedTensorProvider
from shared_tensor.async_provider import AsyncSharedTensorProvider

provider = SharedTensorProvider()
#provider = AsyncSharedTensorProvider()


# Example 1: Simple function sharing
@provider.share(name="add_numbers")
def add_numbers(a, b):
    """Simple addition function"""
    return a + b

# Example 2: Function with external imports
@provider.share(name="create_tensor")  
def create_tensor(shape, device="cpu"):
    """Create a PyTorch tensor"""
    return torch.zeros(shape, device=device)

# Example 3: Complex function with class definition
@provider.share(name="get_model")
def get_model(input_size, hidden_size, output_size):
    """Create and return a neural network model"""
    return SimpleNet(input_size, hidden_size, output_size)

if __name__ == "__main__":
    # Show the function paths that would be sent to the remote server
    print("Function paths that will be sent to remote server:")
    
    for func_name, func_info in provider._registered_functions.items():
        function_path = func_info['function_path']
        print(f"  {func_name} -> {function_path}")
    
    print("\nExample function paths:")
    print(f"  add_numbers -> {provider._registered_functions.get('add_numbers', {}).get('function_path', 'N/A')}")
    print(f"  create_tensor -> {provider._registered_functions.get('create_tensor', {}).get('function_path', 'N/A')}")
    print(f"  get_model -> {provider._registered_functions.get('get_model', {}).get('function_path', 'N/A')}")
    
    print(f"\nRemote server can import these functions using:")
    print(f"  from utils import import_function_from_path")
    print(f"  func = import_function_from_path('__main__:add_numbers')")
    print(f"  result = func(10, 20)")
    
    print("\\nTesting remote function execution...")
    
    # Test simple function
    result = add_numbers(10, 20)
    print(f"10 + 20 = {result}")
    
    # Test tensor creation (requires remote PyTorch)
    tensor = create_tensor((3, 4), device="cuda:0")
    print(f"Created tensor shape: {tensor.shape}")
    
    # Test model creation
    model = get_model(784, 128, 10)
    print(f"Created model: {model}")
