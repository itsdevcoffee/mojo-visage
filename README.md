# Visage ML 🔥

> **Machine learning library built in Mojo**

[![Mojo](https://img.shields.io/badge/Mojo-0.26.1-orange?logo=fire)](https://docs.modular.com/mojo/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A foundational ML/AI library written in Mojo for building neural network systems from the ground up.

---

## What is Visage?

Visage is a machine learning library that provides core primitives for building neural networks, optimizers, and training pipelines. Written entirely in Mojo, it combines Python's ease of use with performance suitable for production workloads.

**Current focus:**
- Core linear algebra operations (vectors, matrices)
- Neural network building blocks
- Optimization algorithms
- Training utilities

---

## Quick Start

### Prerequisites

- Linux or macOS (Mojo not available on Windows yet)
- [Pixi](https://pixi.sh/) package manager

### Installation

```bash
# Clone the repo
git clone https://github.com/itsdevcoffee/visage-ml.git
cd visage-ml

# Install dependencies
pixi install

# Run examples
pixi run vector-add
pixi run dot-product

# Run tests
pixi run test-all
```

---

## Example Usage

```mojo
from visage import vector_add, dot_product

fn main() raises:
    # Basic vector operations
    var a = List[Float64](1.0, 2.0, 3.0)
    var b = List[Float64](4.0, 5.0, 6.0)

    var sum = vector_add(a, b)      # [5.0, 7.0, 9.0]
    var dot = dot_product(a, b)     # 32.0

    print(sum)
    print(dot)
```

---

## Project Status

**Early Development** - APIs are subject to change. Currently implementing core math operations and foundational components.

Interested in the learning journey behind this library? Check out [LEARN.md](LEARN.md) for the full educational roadmap.

---

## Development

```bash
# List all available tasks
pixi task list

# Run specific implementations
pixi run vector-add
pixi run dot-product

# Enter Mojo REPL
pixi run repl
```

---

## Contributing

Contributions are welcome! This is an active project exploring the design space for ML libraries in Mojo.

1. Fork the repo
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Resources

- **Mojo Documentation:** [docs.modular.com/mojo](https://docs.modular.com/mojo/)
- **Learning Path:** [LEARN.md](LEARN.md) - Full educational curriculum
- **Mojo Community:** [Discord](https://discord.gg/modular)

---

**Built with Mojo 🔥**
