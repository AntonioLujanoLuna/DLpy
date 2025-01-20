# DLpy - A Deep Learning Library with DAG-based Autograd

DLpy is a PyTorch-like deep learning framework implemented in Python, featuring automatic differentiation through dynamic computation graphs. It provides a clean and educational implementation of core deep learning components while maintaining good performance.

## Features

- **Automatic Differentiation**: DAG-based autograd system supporting dynamic computational graphs
- **Neural Network Components**:
  - Basic layers (Linear, Conv2D, BatchNorm, LayerNorm)
  - Recurrent layers (LSTM, GRU)
  - Various activation functions
  - Advanced pooling and normalization layers
- **Optimizers**:
  - SGD with momentum
  - Adam
  - RMSprop
  - AdaGrad
  - AdaDelta
  - AdaMax
- **Data Handling**:
  - DataLoader with customizable sampling
  - Dataset abstractions
  - Batch processing utilities

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import dlpy
from dlpy.nn import Linear
from dlpy.optim import Adam

# Create a simple neural network
class SimpleNet(dlpy.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(10, 5)
        self.fc2 = Linear(5, 1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = x.relu()
        return self.fc2(x)

# Create model and optimizer
model = SimpleNet()
optimizer = Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(num_epochs):
    for batch_x, batch_y in dataloader:
        # Forward pass
        output = model(batch_x)
        loss = ((output - batch_y) ** 2).mean()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Dependencies

- NumPy >= 1.20.0
- Python >= 3.8

## Documentation

Detailed documentation is available in the docstrings and code comments.

Key components:

- `core/`: Core tensor and autograd functionality
- `nn/`: Neural network layers and utilities
- `optim/`: Optimization algorithms
- `data/`: Data loading and processing utilities

## Testing

Run the test suite with:

```bash
pytest tests/
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This implementation is inspired by PyTorch and serves both as a learning tool and a lightweight deep learning framework.