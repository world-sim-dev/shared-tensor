#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script to demonstrate function path generation without external dependencies
"""

from shared_tensor.provider import SharedTensorProvider

provider = SharedTensorProvider()


@provider.share(name="math_add")
def add_numbers(a, b):
    """Simple addition function"""
    return a + b


@provider.share(name="string_process")
def process_string(text, uppercase=False):
    """Process a string"""
    result = text.strip()
    if uppercase:
        result = result.upper()
    return result


class Calculator:
    """Example class with shared methods"""
    
    def __init__(self, initial_value=0):
        self.value = initial_value
    
    @provider.share(name="calc_multiply")
    def multiply(self, factor):
        """Instance method to multiply value"""
        self.value *= factor
        return self.value
    
    @staticmethod
    @provider.share(name="calc_power")
    def power(base, exponent):
        """Static method to calculate power"""
        return base ** exponent


def nested_function_example():
    """Example of nested function"""
    
    @provider.share(name="nested_square")
    def square(x):
        return x * x
    
    return square


if __name__ == "__main__":
    print("Function paths that will be sent to remote server:")
    print("=" * 50)
    
    # Show all registered function paths
    for func_name, func_info in provider._registered_functions.items():
        function_path = func_info['function_path']
        module = func_info.get('module', 'Unknown')
        qualname = func_info.get('qualname', 'Unknown')
        
        print(f"Name: {func_name}")
        print(f"  Path: {function_path}")
        print(f"  Module: {module}")
        print(f"  Qualname: {qualname}")
        print()
    
    # Test nested function
    square_func = nested_function_example()
    
    print("After creating nested function:")
    if 'nested_square' in provider._registered_functions:
        nested_info = provider._registered_functions['nested_square']
        print(f"Nested function path: {nested_info['function_path']}")
    
    print("\nExample remote import commands:")
    print("from shared_tensor.utils import import_function_from_path")
    print()
    for func_name, func_info in provider._registered_functions.items():
        function_path = func_info['function_path']
        print(f"# Import {func_name}")
        print(f"func = import_function_from_path('{function_path}')")
        print()
