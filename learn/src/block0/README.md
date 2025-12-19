# Block 0 — Setup + Core Math Tools

**Goal:** Build a tiny numerical toolbox to get comfortable with Mojo before diving into ML.

## Structure

This block is organized into 4 sections, each containing focused projects:

### ✅ Section 1: Vector/Matrix Operations (In Progress)
**Location:** `01_vector_matrix_ops/`

Projects:
- ✅ `01_vector_add.mojo` - Element-wise addition
- ✅ `02_dot_product.mojo` - Multiply and sum (foundation of NNs)
- 🚧 `03_matrix_vector_multiply.mojo` - Combining dot products
- 🚧 `04_matrix_matrix_multiply.mojo` - Full matrix operations

**Progress:** 2/4 complete

---

### 🔜 Section 2: Random Utilities
**Location:** `02_random_utils/`

Projects:
- `01_seed_control.mojo` - Reproducible randomness
- `02_random_generation.mojo` - Generate random vectors/matrices

**Progress:** 0/2 complete

---

### 🔜 Section 3: Dataset Utilities
**Location:** `03_dataset_utils/`

Projects:
- `01_train_val_split.mojo` - Split data for training
- `02_batching.mojo` - Create mini-batches
- `03_shuffling.mojo` - Randomize order

**Progress:** 0/3 complete

---

### 🔜 Section 4: Plotting Utilities
**Location:** `04_plotting_utils/`

Projects:
- `01_export_csv.mojo` - Export data for plotting
- `02_plot_integration.mojo` - Python matplotlib integration

**Progress:** 0/2 complete

---

## Overall Progress: 2/11 Projects Complete (18%)

## Quick Commands

```bash
# Run Section 1 projects
pixi run mojo src/block0/01_vector_matrix_ops/01_vector_add.mojo
pixi run mojo src/block0/01_vector_matrix_ops/02_dot_product.mojo

# Run all tests
pixi run test-all

# Run specific section tests
pixi run mojo tests/block0/01_vector_matrix_ops/test_vector_ops.mojo
```

## Key Learnings So Far

1. **Shape validation is critical** - Catches 80% of ML bugs
2. **Dot product is everything** - Neural nets, attention, loss functions
3. **Element-wise vs reduction** - Returns vector vs scalar
4. **Mojo ownership** - Use `^` to transfer ownership

## Next Up

Complete Section 1 by implementing:
- Matrix-vector multiply
- Matrix-matrix multiply

Then move to Section 2 (Random Utilities).
