# Shared Tensor

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

A high-performance library for sharing GPU memory objects across processes using IPC mechanisms with JSON-RPC 2.0 protocol, enabling model and inference engine separation architecture.

## 🚀 Project Overview

Shared Tensor is a cross-process communication library designed specifically for deep learning and AI applications, utilizing IPC mechanisms and JSON-RPC protocol to achieve:

- **Efficient GPU Memory Sharing**: Cross-process sharing of PyTorch tensors and models
- **Remote Function Execution**: Easy remote function calls through decorators
- **Async/Sync Support**: Flexible execution modes for different scenarios
- **Model Serving**: Deploy machine learning models as independent services
- **Distributed Inference**: Support for distributed computing in multi-GPU environments

## 📋 Core Features

### 🔄 Cross-Process Communication
- **JSON-RPC 2.0 Protocol**: Standardized remote procedure calls
- **HTTP Transport**: Reliable HTTP-based communication mechanism
- **Serialization Optimization**: Efficient PyTorch object serialization/deserialization

### 🎯 Function Sharing
- **Decorator Pattern**: Easy function sharing using `@provider.share`
- **Auto Discovery**: Smart function path resolution and import
- **Parameter Passing**: Support for complex data type parameters

### ⚡ Async Support
- **Async Execution**: `AsyncSharedTensorProvider` supports non-blocking calls
- **Task Management**: Complete async task status tracking
- **Concurrent Processing**: Efficient concurrent request handling

### 🖥️ GPU Compatibility
- **CUDA Support**: Native CUDA tensor sharing support
- **Device Management**: Smart data migration between devices
- **Memory Optimization**: Efficient GPU memory usage

## 🛠️ Installation Guide

### Requirements

- **Python**: 3.8+
- **Operating System**: Linux (recommended)
- **PyTorch**: 1.12.0+
- **CUDA**: Optional, for GPU support

### Installation Methods

#### Install from Source

```bash
# Clone the repository
git clone https://github.com/world-sim-dev/shared-tensor.git
cd shared-tensor

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

#### Development Installation

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Install with test dependencies
pip install -e ".[test]"
```

### Verify Installation

```bash
# Check core functionality
python -c "import shared_tensor; print('✓ Shared Tensor installed successfully')"
```

## 🎯 Quick Start

### 1. Basic Function Sharing

```python
from shared_tensor.async_provider import AsyncSharedTensorProvider

# Create provider
provider = AsyncSharedTensorProvider()

# Share simple function
@provider.share()
def add_numbers(a, b):
    return a + b

# Share PyTorch function
@provider.share()
def create_tensor(shape):
    import torch
    return torch.zeros(shape)

# Load PyTorch model
@provider.share()
def load_model():
    ...

```

### 2. Start Server

```bash
# Method 1: Use command line tool, single server
shared-tensor-server

# Method 2: Use torchrun
torchrun --nproc_per_node=4 --no-python shared-tensor-server

# Method 3: Custom configuration
python shared_tensor/server.py
```


## 📖 Detailed Usage

### Model Sharing Example

```python
import torch
import torch.nn as nn

from shared_tensor.async_provider import AsyncSharedTensorProvider

# Create provider
provider = AsyncSharedTensorProvider()

# Define model
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Share model creation function
@provider.share(name="create_model")
def create_model(input_size=784, hidden_size=128, output_size=10):
    model = SimpleNet(input_size, hidden_size, output_size)
    return model

# Share inference function
model = create_model()
with torch.no_grad():
    model(input_data)
```

## 🔧 Configuration Options

### Server Configuration

```python
from shared_tensor.server import SharedTensorServer

server = SharedTensorServer(
    host="0.0.0.0",           # Listen address
    port=2537,                # Port number
    timeout=30,               # Request timeout
    max_workers=4,            # Maximum worker threads
    enable_cache=True,        # Enable result caching
    debug=False               # Debug mode
)
```

## 🧪 Testing

### Run Test Suite

```bash
# Run all tests
python tests/run_tests.py

# Run specific category tests
python tests/run_tests.py --category unit
python tests/run_tests.py --category integration
python tests/run_tests.py --category pytorch

# Run only PyTorch related tests
python tests/run_tests.py --torch-only

# Verbose output
python tests/run_tests.py --verbose
```

### Test Environment Info

```bash
# Check test environment
python tests/run_tests.py --env-info
```

### Individual Test Files

```bash
# Test tensor serialization
python tests/pytorch_tests/test_tensor_serialization.py

# Test async system
python tests/integration/test_async_system.py

# Test client
python tests/integration/test_client.py
```

## 🏗️ Architecture Design

### Core Components

```
shared-tensor/
├── shared_tensor/              # Core modules
│   ├── server.py              # JSON-RPC server
│   ├── client.py              # Sync client
│   ├── provider.py            # Sync provider
│   ├── async_client.py        # Async client
│   ├── async_provider.py      # Async provider
│   ├── async_task.py          # Async task management
│   ├── jsonrpc.py            # JSON-RPC protocol implementation
│   ├── utils.py              # Utility functions
│   └── errors.py             # Exception definitions
├── scripts/                   # Script tools
├── examples/                  # Usage examples
└── tests/                     # Test suite
```

### Communication Flow

```mermaid
sequenceDiagram
    participant CA as Client App
    participant SC as SharedTensorClient
    participant SS as SharedTensorServer
    participant FE as Function Executor
    
    Note over CA, FE: Client-Server Communication Flow
    
    CA->>SC: call_function("model_inference", args)
    SC->>SC: Serialize parameters
    SC->>SS: HTTP POST /jsonrpc<br/>JSON-RPC Request
    
    Note over SS: Server Processing
    SS->>SS: Parse JSON-RPC request
    SS->>SS: Resolve function path
    SS->>FE: Import & execute function
    FE->>FE: Deserialize parameters
    FE->>FE: Execute function logic
    FE->>SS: Return execution result
    
    Note over SS: Response Preparation
    SS->>SS: Serialize result
    SS->>SS: Create JSON-RPC response
    SS->>SC: HTTP Response<br/>JSON-RPC Result
    
    Note over SC: Client Processing
    SC->>SC: Parse response
    SC->>SC: Deserialize result
    SC->>CA: Return final result
    
    Note over CA, FE: End-to-End Process Complete
```

### Debug Tips

1. **Enable verbose logging**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

2. **Use debug mode**:
```python
provider = SharedTensorProvider(verbose_debug=True)
```

3. **Check function paths**:
```python
provider = SharedTensorProvider()
print(provider._registered_functions)
```

## 🤝 Contributing

We welcome community contributions! Please follow these steps:

### Development Environment Setup

```bash
# Clone repository
git clone https://github.com/world-sim-dev/shared-tensor.git
cd shared-tensor

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Code Standards

```bash
# Code formatting
black shared_tensor/ tests/ examples/

# Import sorting
isort shared_tensor/ tests/ examples/

# Static checking
flake8 shared_tensor/
mypy shared_tensor/
```

### Submission Process

1. Fork the project and create a feature branch
2. Write code and tests
3. Run the complete test suite
4. Submit a Pull Request

### Test Requirements

- New features must include tests
- Maintain test coverage > 90%
- All tests must pass

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details

## 🙏 Acknowledgments

- [PyTorch](https://pytorch.org/) - Deep learning framework
- [JSON-RPC 2.0](https://www.jsonrpc.org/) - Remote procedure call protocol

## 📞 Contact Us

- **Issues**: [GitHub Issues](https://github.com/world-sim-dev/shared-tensor/issues)
- **Documentation**: [Shared Tensor Documentation](https://github.com/world-sim-dev/shared-tensor/wiki)
- **Source**: [GitHub Repository](https://github.com/world-sim-dev/shared-tensor)

---

**Shared Tensor** - Making GPU memory sharing simple and efficient 🚀