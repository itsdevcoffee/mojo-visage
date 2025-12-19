# Visage ML 🔥

> **Neural network library built from scratch in Mojo**

[![Mojo](https://img.shields.io/badge/Mojo-0.26.1-orange?logo=fire)](https://docs.modular.com/mojo/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-WIP-yellow)](https://github.com/itsdevcoffee/mojo-visage)

A foundational machine learning library written in Mojo, implementing neural networks and training algorithms from first principles. Combines Python's expressiveness with C-level performance.

⚠️ **Work in Progress** - APIs are subject to change as we explore the design space for ML in Mojo.

---

## Features

### ✅ Implemented

**Linear Algebra**
- Vector operations (add, dot product, elementwise multiply/divide)
- Matrix operations (multiply, transpose, matrix-vector, matrix-matrix)
- Scalar operations and shape validation

**Neural Networks**
- Dense (fully-connected) layers
- Activation functions: ReLU, Sigmoid, Tanh, Softmax
- Forward propagation through multi-layer networks

**Training**
- **Backpropagation** - Complete gradient computation using chain rule
- Loss functions: MSE, Binary Cross-Entropy
- Gradient descent optimizer
- Full training loop with real learning

### 🚧 In Progress

- Advanced optimizers (Adam, momentum, RMSprop)
- Regularization (dropout, L2, batch normalization)
- Convolutional layers
- SIMD optimization for performance
- Model save/load

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/itsdevcoffee/mojo-visage.git
cd mojo-visage

# Install dependencies (requires Mojo)
pixi install

# Run tests
pixi run test

# Watch a network learn XOR!
pixi run train-xor
```

### Basic Usage

```mojo
from visage import matrix_vector_multiply, vector_add

fn main() raises:
    var weights: List[List[Float64]] = [
        [0.5, -0.3],
        [0.2, 0.8]
    ]
    var inputs: List[Float64] = [1.0, 2.0]

    var output = matrix_vector_multiply(weights, inputs)
    print(output)  # Neural network layer computation!
```

### Train a Network

```bash
pixi run train-xor
```

```
Training Neural Network on XOR Problem
======================================

Epoch 500  | Loss: 0.254
Epoch 1000 | Loss: 0.114
Epoch 1500 | Loss: 0.018
...
Epoch 5000 | Loss: 0.001

Testing trained network:
Input: [0, 0] | Target: 0 | Prediction: 0.02 ✓
Input: [0, 1] | Target: 1 | Prediction: 0.96 ✓
Input: [1, 0] | Target: 1 | Prediction: 0.98 ✓
Input: [1, 1] | Target: 0 | Prediction: 0.04 ✓

✓ Training complete! Network learned XOR!
```

---

## Examples

```bash
# Linear algebra operations
pixi run example-basic

# Forward propagation demo
pixi run example-network

# Train on XOR (classic non-linear problem)
pixi run train-xor
```

---

## Project Structure

```
mojo-visage/
├── src/
│   ├── visage.mojo          # Core linear algebra
│   └── nn.mojo              # Neural network components
├── tests/                   # Test suite
├── examples/                # Usage examples & training demos
├── learn/                   # Educational content (separate from library)
└── docs/                    # Documentation
```

**Using the library:**
```bash
mojo -I src your_code.mojo
```

---

## Development Status

| Component | Status |
|-----------|--------|
| Linear Algebra | ✅ Complete |
| Activations | ✅ Complete |
| Forward Pass | ✅ Complete |
| Backpropagation | ✅ Complete |
| Loss Functions | ✅ Complete |
| Basic Training | ✅ Complete |
| Advanced Optimizers | 🚧 In Progress |
| Regularization | 📋 Planned |
| Conv Layers | 📋 Planned |
| SIMD Optimization | 📋 Planned |

---

## Why Mojo?

- **Python-like syntax** - Easy to read and write
- **C-level performance** - Fast execution for ML workloads
- **Built for AI** - First-class support for ML primitives
- **Zero dependencies** - Pure Mojo implementation

---

## Learning Resources

This library is built from scratch as a learning exercise. If you're interested in the educational journey:

- **[learn/](learn/)** - Step-by-step implementations (blocks 0-11)
- **[STRUCTURE.md](STRUCTURE.md)** - Repository organization
- **Learning tasks:** `pixi run learn-*`

---

## Contributing

Contributions welcome! This project is actively exploring ML library design in Mojo.

1. Fork the repository
2. Create a feature branch
3. Write tests for new functionality
4. Submit a pull request

Areas where contributions are especially welcome:
- SIMD optimizations
- Advanced optimizers (Adam, RMSprop)
- Regularization techniques
- Performance benchmarks

---

## Roadmap

**v0.1 (Current)** - Foundation
- [x] Core linear algebra
- [x] Basic neural network layers
- [x] Backpropagation
- [x] Training loop

**v0.2** - Optimization
- [ ] Advanced optimizers (Adam, momentum)
- [ ] Learning rate schedules
- [ ] SIMD acceleration
- [ ] Performance benchmarks

**v0.3** - Production Features
- [ ] Model serialization (save/load)
- [ ] Regularization (dropout, L2)
- [ ] Batch normalization
- [ ] Real dataset support

**v0.4+** - Advanced
- [ ] Convolutional layers
- [ ] Recurrent layers
- [ ] Custom autodiff
- [ ] GPU support

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

Built with [Mojo](https://docs.modular.com/mojo/) 🔥

Inspired by building ML from first principles to understand what's really happening under the hood.

---

**[View on GitHub](https://github.com/itsdevcoffee/mojo-visage)** | **[Report Issues](https://github.com/itsdevcoffee/mojo-visage/issues)**
