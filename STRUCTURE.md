# Repository Structure

This repository is organized into two main sections:

## Library Code (`visage/`)

Production-ready ML library implementation.

```
visage/
├── __init__.mojo          # Main package
├── linalg/                # Linear algebra primitives
│   ├── __init__.mojo
│   └── ops.mojo          # vector_add, dot_product, matrix ops
├── nn/                    # Neural network modules (coming soon)
├── optim/                 # Optimizers (coming soon)
└── utils/                 # Utilities (coming soon)
```

**Usage:**
```bash
# Run tests
pixi run test

# Run examples
pixi run example-basic
pixi run example-nn
```

## Learning Content (`learn/`)

Educational materials and step-by-step ML implementations.

```
learn/
├── README.md              # Full learning curriculum
├── src/                   # Block 0-11 learning projects
├── tests/                 # Learning exercise tests
├── scripts/               # Visualization scripts
├── results/               # Learning outputs
└── docs/                  # Learning guides, cheatsheets
```

**Usage:**
```bash
# Run learning projects
pixi run learn-vector-add
pixi run learn-dot-product

# Run learning tests
pixi run learn-test-all

# View visualizations
pixi run learn-viz-dot-product
```

## Examples & Tests

- `examples/` - Library usage examples
- `tests/` - Library test suite
- `docs/` - Library API documentation

## Quick Commands

```bash
# Library
pixi run test              # Run library tests
pixi run example-basic     # Basic operations demo
pixi run example-nn        # Neural network layer demo

# Learning
pixi run learn-vector-add  # Educational implementation
pixi run learn-test-all    # Learning exercise tests
```
