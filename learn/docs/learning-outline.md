# Build-From-Scratch ML → LLMs Learning Path (Project Checklist)

A practical, bottom-up build sequence that prioritizes intuition and “I can explain this” mastery.
Each block is something you **build**, not just read about.

---

## How to use this

- Treat each block as a mini-project.
- Don’t optimize for performance; optimize for **understanding + observability**.
- For every build: log loss curves, sanity-check gradients, and write a short README of what you learned.

---

# Block 0 — Setup + Core Math Tools (1 project)

### Build

- A tiny “numerical toolbox”:
  - vector/matrix ops (or use NumPy, but understand shapes)
  - random seed control + reproducibility utilities
  - dataset utilities: train/val split, batching, shuffling
  - plotting utilities: loss curves, decision boundaries, histograms

### You should be able to explain

- What a tensor shape means and why bugs are often “shape bugs”
- Why reproducibility matters (seeds, determinism limits)

---

# Block 1 — Optimization + Regression Fundamentals (4 projects)

### Build

- Linear regression (closed-form / normal equation)
- Linear regression (gradient descent)
- Loss functions from scratch:
  - MSE
  - MAE (and why it behaves differently)
- Gradient descent variants (simple):
  - batch GD
  - mini-batch GD
  - SGD

### Add-ons (recommended)

- Feature scaling / standardization (and show why it matters)
- L2 regularization (ridge) implemented manually

### You should be able to explain

- Convex vs non-convex (at a practical level)
- Learning rate failure modes (too big explodes, too small stalls)
- Overfitting vs underfitting with a simple example

---

# Block 2 — Classification + Probabilistic Thinking (4 projects)

### Build

- Logistic regression from scratch (binary)
- Softmax regression from scratch (multiclass)
- Cross-entropy loss from scratch (binary + multiclass)
- Metrics from scratch:
  - accuracy, precision/recall/F1
  - confusion matrix

### Add-ons (recommended)

- Decision boundary visualization (2D toy datasets)
- Class imbalance handling (weighted loss)

### You should be able to explain

- Why softmax + cross-entropy “fits together”
- Calibration (probabilities vs confident wrong answers)
- Why accuracy can be misleading

---

# Block 3 — Backprop Intuition: Tiny Neural Net Without Autograd (3 projects)

### Build

- A 2-layer MLP for classification:
  - forward pass
  - manual backprop (derive gradients)
  - train loop with SGD
- Activation functions from scratch:
  - ReLU, tanh, sigmoid
- Initialization experiments:
  - show bad init vs decent init behavior

### Add-ons (recommended)

- Gradient checking with finite differences (sanity test)

### You should be able to explain

- Chain rule as “credit assignment”
- Vanishing/exploding gradients (and what causes them)
- Why ReLU often trains easier than sigmoid/tanh

---

# Block 4 — Modern Training Ingredients (5 projects)

_(You can now “earn” using an autodiff framework, but still implement the ideas yourself.)_

### Build

- A mini autograd engine (micrograd-style):
  - scalar ops first, then small tensors (optional)
- Optimizers:
  - Momentum
  - RMSProp
  - Adam
- Regularization:
  - L2 weight decay
  - dropout (forward pass + training behavior)
- Normalization:
  - batch norm (conceptually) or layer norm (more relevant for transformers)
- Training utilities:
  - early stopping
  - learning rate schedules (step, cosine)

### You should be able to explain

- What Adam is doing differently than SGD
- Why normalization stabilizes training
- Why dropout helps generalization (and when it doesn’t)

---

# Block 5 — Embeddings + Representation Learning (3 projects)

### Build

- Word embeddings (toy):
  - co-occurrence matrix + SVD OR a simple skip-gram-like approach
- Similarity search:
  - cosine similarity
  - nearest neighbors retrieval
- Analogy / clustering demos:
  - show embedding space has structure

### You should be able to explain

- What an embedding “means” operationally
- Why cosine similarity is commonly used
- How embeddings connect to retrieval (RAG foundations)

---

# Block 6 — Language Modeling Basics (2 projects)

### Build

- Character-level language model (n-gram baseline)
- Neural language model (MLP over context window):
  - predict next token
  - evaluate perplexity

### You should be able to explain

- What “next-token prediction” really trains
- Why perplexity is used (and what it implies)
- Why longer context helps but adds complexity

---

# Block 7 — Sequence Models (RNNs) (3 projects)

### Build

- Simple RNN from scratch (tiny)
- LSTM or GRU (pick one) from scratch (even tinier)
- Train on a toy sequence task:
  - copy task
  - parity
  - next-char on small text

### You should be able to explain

- Why RNNs struggle with long dependencies
- How gates help (high-level, no handwaving)
- Teacher forcing and why generation differs from training

---

# Block 8 — Attention (The Bridge to Transformers) (2 projects)

### Build

- Scaled dot-product attention from scratch
- “Attention as weighted average” visual demo:
  - show attention weights over tokens

### You should be able to explain

- What attention computes in one sentence
- Why scaling by sqrt(d_k) exists
- Why attention can be parallelized better than RNNs

---

# Block 9 — A Tiny Transformer (Encoder/Decoder or Decoder-only) (5 projects)

### Build

- Tokenization basics:
  - implement a simple BPE tokenizer (toy) OR use one and explain it
- Positional encodings:
  - sinusoidal or learned
- Multi-head self-attention from scratch
- Transformer block:
  - attention + MLP + residuals + layer norm
- Train a tiny decoder-only transformer:
  - tiny dataset (small text)
  - generate samples
  - track loss/perplexity

### Add-ons (recommended)

- Causal masking (and prove you understand it by breaking it)
- Weight tying (input/output embeddings)

### You should be able to explain

- Residual connections + layer norm role in stability
- What causal masking prevents
- Why multi-head attention exists

---

# Block 10 — “LLM Reality” (What Makes It Production) (6 projects)

### Build

- Inference essentials:
  - greedy vs sampling
  - temperature
  - top-k / nucleus (top-p)
- KV cache concept demo:
  - show why it speeds up autoregressive generation
- Fine-tuning basics:
  - supervised fine-tune on a tiny instruction dataset
- Evaluation harness:
  - holdout perplexity
  - a few task-style evals (toy)
- Safety/robustness experiments (toy):
  - prompt sensitivity
  - format-following failures
- Data pipeline basics:
  - dedupe
  - filtering
  - train/val leakage checks

### You should be able to explain

- Why decoding strategy changes output character
- Why evals are hard and easy to game
- Why data quality often beats clever tricks

---

# Block 11 — Capstone: Build a Mini Chat Model + Tool Use (2 projects)

### Build

- A small “chat-style” dataset format + SFT (toy)
- Minimal tool-use / function calling loop:
  - model outputs structured JSON
  - your runtime executes a tool
  - tool output is appended back into context

### You should be able to explain

- What instruction tuning changes vs base LM
- Why tool use is mostly “prompt + parsing + guardrails”
- Failure modes: tool hallucination, schema drift, looping

---

# “You’re solid” milestone (what good looks like)

You’ve reached strong working understanding when you can:

- Train a tiny transformer end-to-end and debug when it fails
- Explain attention, layer norm, residuals, masking, and tokenization clearly
- Predict common failure modes from symptoms (loss spikes, mode collapse, repetitive outputs)
- Reason about tradeoffs: compute vs data vs architecture vs decoding
- Build a minimal fine-tune + evaluation loop and trust your results

---

# Optional Advanced Blocks (pick based on interest)

### Systems + scaling

- Mixed precision (fp16/bf16) concepts
- Gradient accumulation
- Distributed data parallel concept demo

### Alignment & preference tuning (toy)

- Reward model basics
- DPO-style preference learning (conceptual build)

### Retrieval-Augmented Generation (RAG)

- Embed → retrieve → re-rank → generate
- Basic chunking + citations + evals for retrieval

---

## Suggested pacing (rough)

- Blocks 1–3: fundamentals (core intuition)
- Blocks 4–6: modern DL + LM basics
- Blocks 7–9: transformers
- Blocks 10–11: “LLM engineering reality”

---

## Deliverables per block (do this every time)

- `README.md` (what you built, what broke, what you learned)
- `results/` with:
  - loss curves
  - a few model outputs (before/after fixes)
- “I can explain it” notes:
  - 5–10 bullet explanations in your own words
