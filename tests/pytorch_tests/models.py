"""
PyTorch Model Classes for Testing

This module contains model definitions used across various tests.
Models are defined at module level to ensure pickle compatibility.
"""

import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    """Simple test model for basic serialization testing"""
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 5)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(5, 1)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class DynamicTrainingModel(nn.Module):
    """Dynamic training model with configurable size"""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))


class GPUTrainingModel(nn.Module):
    """Training model optimized for GPU testing"""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class ComplexModel(nn.Module):
    """More complex model for advanced testing"""
    def __init__(self, input_size=784, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class ModelWithOptimizer:
    """Model wrapper with optimizer for testing training scenarios"""
    def __init__(self, model):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
    
    def train_step(self, x, y):
        self.optimizer.zero_grad()
        output = self.model(x)
        loss = self.loss_fn(output, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()


class CNNModel(nn.Module):
    """Simple CNN model for testing convolutional layers"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


# Model factory functions for testing
def create_simple_model():
    """Create a simple model for basic tests"""
    return SimpleModel()


def create_dynamic_model(input_size, hidden_size, output_size):
    """Create a dynamic model with specified dimensions"""
    return DynamicTrainingModel(input_size, hidden_size, output_size)


def create_gpu_model(input_size, hidden_size, output_size, device='cuda'):
    """Create a GPU model and move to specified device"""
    model = GPUTrainingModel(input_size, hidden_size, output_size)
    if torch.cuda.is_available() and device.startswith('cuda'):
        model = model.to(device)
    return model


def create_complex_model(input_size=784, num_classes=10):
    """Create a complex model for advanced testing"""
    return ComplexModel(input_size, num_classes)


def create_cnn_model(num_classes=10):
    """Create a CNN model for image processing tests"""
    return CNNModel(num_classes)


# Custom activation layer at module level to avoid pickling issues
class CustomActivation(nn.Module):
    """Custom activation function"""
    
    def __init__(self, alpha=0.2):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, x):
        return torch.where(x > 0, x, self.alpha * x)


class ModelWithCustomLayers(nn.Module):
    """Model with custom layer implementations for pickling tests"""
    
    def __init__(self):
        super().__init__()
        
        self.linear = nn.Linear(10, 5)
        self.custom_activation = CustomActivation(alpha=0.3)
        self.output = nn.Linear(5, 1)
    
    def forward(self, x):
        x = self.linear(x)
        x = self.custom_activation(x)
        x = self.output(x)
        return x


class ModelWithBuffers(nn.Module):
    """Model with registered buffers for advanced testing"""
    
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
        
        # Register some buffers
        self.register_buffer('running_mean', torch.zeros(5))
        self.register_buffer('running_var', torch.ones(5))
        self.register_buffer('num_batches_tracked', torch.tensor(0))
    
    def forward(self, x):
        x = self.linear(x)
        
        # Update buffers (like BatchNorm would do)
        if self.training:
            self.num_batches_tracked += 1
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            
            # Update running statistics
            momentum = 0.1
            self.running_mean = (1 - momentum) * self.running_mean + momentum * mean
            self.running_var = (1 - momentum) * self.running_var + momentum * var
        
        # Normalize using running statistics
        x = (x - self.running_mean) / torch.sqrt(self.running_var + 1e-5)
        return x


def create_model_with_custom_layers():
    """Create a model with custom layers"""
    return ModelWithCustomLayers()


def create_model_with_buffers():
    """Create a model with registered buffers"""
    return ModelWithBuffers()
