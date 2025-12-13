# Mojo ML From Scratch 🔥

> **Build Machine Learning from fundamentals to LLMs in Mojo**
> Learn by implementing everything yourself: Linear Algebra → Neural Networks → Transformers → Language Models

[![Mojo](https://img.shields.io/badge/Mojo-0.26.1-orange?logo=fire)](https://docs.modular.com/mojo/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Progress](https://img.shields.io/badge/progress-2%2F40%20projects-yellow)](src/block0/README.md)

---

## 🎯 What is This?

This is a **40+ project learning journey** where I build ML/AI systems from scratch in Mojo—no PyTorch, no TensorFlow, just pure understanding.

**The Philosophy:** You don't truly understand neural networks until you've implemented backpropagation by hand. You don't understand transformers until you've built attention from scratch. This repo is about **deep understanding**, not just using libraries.

**Why Mojo?** 🔥
- Python-like syntax (easy to read)
- C-like performance (fast to run)
- Built for AI/ML from the ground up
- Perfect for learning how things *really* work

---

## 📚 Learning Path

```
Block 0: Math Fundamentals        [████░░░░░░] 18%
Block 1: Optimization              [░░░░░░░░░░]  0%
Block 2: Classification            [░░░░░░░░░░]  0%
Block 3: Backpropagation           [░░░░░░░░░░]  0%
Block 4: Modern Training           [░░░░░░░░░░]  0%
Block 5: Embeddings                [░░░░░░░░░░]  0%
Block 6: Language Modeling         [░░░░░░░░░░]  0%
Block 7: RNNs                      [░░░░░░░░░░]  0%
Block 8: Attention                 [░░░░░░░░░░]  0%
Block 9: Transformers              [░░░░░░░░░░]  0%
Block 10: LLM Reality              [░░░░░░░░░░]  0%
Block 11: Chat + Tool Use          [░░░░░░░░░░]  0%
```

**Full learning outline:** [docs/learning-outline.md](docs/learning-outline.md)

---

## 🚀 Quick Start

### Prerequisites

- Linux or macOS (Mojo not available on Windows yet)
- [Pixi](https://pixi.sh/) package manager

### Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/mojo-ml-from-scratch.git
cd mojo-ml-from-scratch

# Install dependencies (Mojo, Python, NumPy, matplotlib)
pixi install

# Run your first Mojo ML code!
pixi run vector-add
pixi run dot-product

# Run tests
pixi run test-all
```

**New to Mojo?** Check out the [Mojo Cheatsheet](docs/cheatsheets/mojo-cheatsheet.md) for quick syntax reference!

---

## 💡 What You'll Build

### Block 0: Core Math Tools (In Progress)
- ✅ **Vector addition** - Element-wise operations and shape validation
- ✅ **Dot product** - The foundation of neural networks
- 🚧 Matrix-vector multiply
- 🚧 Matrix-matrix multiply
- 🔜 Random number generation with seeds
- 🔜 Dataset splitting and batching
- 🔜 Plotting and visualization

[View Block 0 Progress →](src/block0/README.md)

### Block 1: Optimization + Regression (Coming Next)
- Linear regression (closed-form solution)
- Gradient descent from scratch
- Loss functions (MSE, MAE)
- Feature scaling and regularization

### Blocks 2-11: The Journey to LLMs
- **Block 2:** Logistic regression, softmax, cross-entropy
- **Block 3:** Backpropagation without autograd (the hard way!)
- **Block 4:** Adam optimizer, dropout, batch norm
- **Block 5:** Word embeddings, similarity search
- **Block 6:** Character-level language models
- **Block 7:** RNNs and LSTMs from scratch
- **Block 8:** Attention mechanisms
- **Block 9:** Build a tiny transformer
- **Block 10:** LLM inference, fine-tuning, KV cache
- **Block 11:** Chat model with tool use

---

## 📂 Project Structure

```
mojo-ml-from-scratch/
├── src/
│   └── block0/
│       ├── 01_vector_matrix_ops/
│       │   ├── 01_vector_add.mojo       ✅ Complete
│       │   ├── 02_dot_product.mojo      ✅ Complete
│       │   └── README.md
│       ├── 02_random_utils/             🚧 Next
│       ├── 03_dataset_utils/
│       └── 04_plotting_utils/
├── tests/                                ✅ 7/7 passing
├── docs/
│   ├── learning-outline.md              📖 Full curriculum
│   └── cheatsheets/
│       └── mojo-cheatsheet.md           🔥 Mojo syntax reference
└── scripts/
```

**Organization Philosophy:** Each block → sections → numbered projects. Follow the numbers to learn in the right order!

---

## 🧪 Testing

Every implementation has comprehensive tests:

```bash
# Run all tests
pixi run test-all

# Run specific tests
pixi run test-vector-add
pixi run test-dot-product
```

**Test-Driven Learning:** Each project includes edge cases, error handling, and mathematical properties to ensure deep understanding.

---

## 📖 Key Learnings So Far

### Vector Addition
```mojo
fn vector_add(a: List[Float64], b: List[Float64]) raises -> List[Float64]:
    if len(a) != len(b):
        raise Error("Shape mismatch!")  # 80% of ML bugs!
    var result = List[Float64]()
    for i in range(len(a)):
        result.append(a[i] + b[i])
    return result^
```
**Lesson:** Always validate shapes. Shape bugs are the #1 source of ML errors.

### Dot Product
```mojo
fn dot_product(a: List[Float64], b: List[Float64]) raises -> Float64:
    var result: Float64 = 0.0
    for i in range(len(a)):
        result += a[i] * b[i]  # Multiply and accumulate
    return result
```
**Lesson:** Dot product is EVERYTHING in ML. Neural nets, attention, loss functions—all built on this.

---

## 🎓 Learning Resources

- **Mojo Documentation:** [docs.modular.com/mojo](https://docs.modular.com/mojo/)
- **Mojo Cheatsheet:** [Quick syntax reference](docs/cheatsheets/mojo-cheatsheet.md)
- **Learning Outline:** [Full curriculum breakdown](docs/learning-outline.md)
- **Mojo Community:** [Discord](https://discord.gg/modular)

---

## 🛠️ Development

### Running Individual Projects

```bash
# List all available commands
pixi task list

# Run specific implementations
pixi run vector-add
pixi run dot-product

# Enter Mojo REPL for experimentation
pixi run repl
```

### Project Guidelines

1. **Build from scratch** - No ML libraries, implement everything
2. **Optimize for understanding** - Readable code > clever code
3. **Test everything** - Edge cases, error cases, mathematical properties
4. **Document learnings** - README per block with key insights

---

## 🗺️ Roadmap

- [x] Set up Mojo environment with pixi
- [x] Implement vector addition with shape validation
- [x] Implement dot product (foundation of neural nets)
- [x] Build comprehensive test suite
- [ ] Complete Block 0 (matrix operations, random utils, datasets)
- [ ] Block 1: Gradient descent and linear regression
- [ ] Block 2: Classification with logistic regression
- [ ] Block 3: Manual backpropagation (the enlightenment moment!)
- [ ] Blocks 4-11: The journey to LLMs continues...

---

## 🤝 Contributing

This is primarily a **personal learning journey**, but if you're also learning ML from scratch in Mojo:

1. **Star the repo** if you find it helpful! ⭐
2. **Follow along** and build your own implementations
3. **Share your learnings** - Open issues with questions or insights
4. **Suggest improvements** - PRs welcome for bug fixes or clarity

---

## 📝 License

MIT License - Feel free to use this for your own learning!

---

## 🔍 Topics

`mojo` `machine-learning` `deep-learning` `neural-networks` `transformers` `llm` `from-scratch` `educational` `ai` `ml-fundamentals` `backpropagation` `gradient-descent` `attention-mechanism` `learn-by-building` `build-in-public`

---

## ⭐ Star History

If you find this helpful, consider giving it a star! It helps others discover this learning resource.

---

**Built with Mojo 🔥 | Learning in Public | [Follow the Journey](https://github.com/yourusername/mojo-ml-from-scratch)**
