# Quick Reference: Buffer/Tensor/SIMD in Mojo

**One-page cheat sheet for high-performance DSP**

---

## When to Use What

| Use Case | Data Structure | Speedup | Complexity |
|----------|---------------|---------|------------|
| Element-wise ops | `UnsafePointer + SIMD` | 8-16x | Medium |
| 1D arrays (FFT) | `NDBuffer[dtype, 1]` | 5-7x | Low |
| 2D arrays (spectrogram) | `NDBuffer[dtype, 2]` | 4-8x | Low |
| Matrix multiply | `MAX Tensor + ops.matmul` | 10-100x | Low |
| Prototyping | `List[Float64]` | 1x | Very Low |

---

## UnsafePointer + SIMD Pattern

**Use for:** Element-wise multiply, add, power spectrum

```mojo
from memory import UnsafePointer

fn simd_operation(data: List[Float64]) -> List[Float64]:
    var N = len(data)
    var ptr = UnsafePointer[Float64].alloc(N)
    var result_ptr = UnsafePointer[Float64].alloc(N)

    # Copy to pointer
    for i in range(N):
        ptr[i] = data[i]

    # SIMD loop
    alias width = 8
    var i = 0
    while i + width <= N:
        var vec = ptr.load[width=width](i)
        var res = vec * 2.0  # Your operation
        result_ptr.store(i, res)
        i += width

    # Remainder
    while i < N:
        result_ptr[i] = ptr[i] * 2.0
        i += 1

    # Convert back
    var result = List[Float64]()
    for i in range(N):
        result.append(result_ptr[i])

    ptr.free()
    result_ptr.free()
    return result^
```

---

## NDBuffer Pattern

**Use for:** FFT, spectrograms, contiguous arrays

```mojo
from buffer import NDBuffer

# 1D buffer
var fft_data = NDBuffer[DType.float64, 1](shape=(512,))

# 2D buffer (spectrogram)
var spec = NDBuffer[DType.float64, 2](shape=(3000, 201))

# Access
fft_data[i] = value
spec[frame, freq] = value

# Flatten for SIMD
if spec.is_contiguous():
    var flat = spec.flatten()
```

---

## MAX Tensor Pattern

**Use for:** Matrix operations, neural networks

```mojo
from max.tensor import Tensor
from max.graph import ops

# Create tensors
var filterbank = Tensor[DType.float64](shape=(80, 201))
var spec = Tensor[DType.float64](shape=(201, 3000))

# Matrix multiply (GEMM)
var mel_spec = ops.matmul(filterbank, spec)
# Result: (80, 3000)
```

---

## Performance Rules

### ✅ DO

- Use contiguous memory (UnsafePointer, NDBuffer, Tensor)
- Direct SIMD load/store: `ptr.load[width=8](i)`
- Process remainder after SIMD loop
- Benchmark before and after
- Choose SIMD width: 4-8 for Float64

### ❌ DON'T

- Manual SIMD load loops: `for j in range(w): vec[j] = list[i+j]`
- Use List for performance-critical code
- Assume SIMD is faster (measure!)
- Forget to free() UnsafePointer
- Mix memory models without conversion

---

## Common Operations

### Element-wise multiply
```mojo
var result = a_ptr.load[width=8](i) * b_ptr.load[width=8](i)
```

### Power spectrum (real² + imag²)
```mojo
var power = real_vec * real_vec + imag_vec * imag_vec
```

### Matrix multiply
```mojo
var C = ops.matmul(A, B)  # MAX Tensor
```

---

## Conversion Helpers

```mojo
# List → UnsafePointer
var ptr = UnsafePointer[Float64].alloc(len(list))
for i in range(len(list)):
    ptr[i] = list[i]

# UnsafePointer → List
var list = List[Float64]()
for i in range(size):
    list.append(ptr[i])

# List → NDBuffer
var buf = NDBuffer[DType.float64, 1](shape=(len(list),))
for i in range(len(list)):
    buf[i] = list[i]
```

---

## Memory Safety

### RAII Wrapper (Recommended)
```mojo
struct Buffer:
    var data: UnsafePointer[Float64]
    var size: Int

    fn __init__(out self, size: Int):
        self.data = UnsafePointer[Float64].alloc(size)
        self.size = size

    fn __del__(deinit self):
        self.data.free()  # Automatic!
```

---

## Migration Priority

1. **Week 1:** UnsafePointer for window, power spectrum
2. **Week 2:** NDBuffer for FFT, spectrograms
3. **Week 3:** MAX Tensor for mel filterbank
4. **Week 4:** Optimization and tuning

---

## Performance Targets

| Current (List) | After Migration | Speedup |
|---------------|-----------------|---------|
| 165ms | 20-25ms | 6.6-8.3x |
| vs librosa: 11x slower | vs librosa: competitive | ✅ |

---

## Red Flags (SLOW Code)

```mojo
# ❌ Manual SIMD load
for j in range(simd_width):
    vec[j] = list[i + j]

# ❌ Nested Lists
var spec = List[List[Float64]]()

# ❌ Triple loop for matrix multiply
for i in range(m):
    for j in range(n):
        for k in range(p):
            C[i][j] += A[i][k] * B[k][j]
```

---

## Green Flags (FAST Code)

```mojo
# ✅ Direct SIMD load
var vec = ptr.load[width=8](i)

# ✅ Contiguous 2D
var spec = NDBuffer[DType.float64, 2](shape=(m, n))

# ✅ Optimized GEMM
var C = ops.matmul(A, B)
```

---

## SIMD Width Selection

```mojo
# Auto-select
alias width = simdwidthof[DType.float64]()

# Explicit (typical for Float64)
alias width = 8

# Always handle remainder!
while i + width <= N:
    # SIMD
    i += width
while i < N:
    # Scalar
    i += 1
```

---

## Benchmarking

```mojo
from benchmark import Benchmark

@always_inline
@parameter
fn test_fn():
    var result = my_function(data)

var report = Benchmark().run[test_fn]()
print("Time:", report.mean(), "ms")
```

---

## Key Learnings from mojo-audio

1. **Iterative FFT > Recursive FFT:** 3x speedup (algorithmic)
2. **List SIMD failed:** Manual load/store overhead
3. **Need contiguous memory:** For real SIMD benefits
4. **Measure everything:** Optimization can make things slower

---

## Decision Tree

```
Is it matrix multiply?
  YES → MAX Tensor + ops.matmul
  NO  ↓

Is it multi-dimensional array?
  YES → NDBuffer[dtype, rank]
  NO  ↓

Is it element-wise operation?
  YES → UnsafePointer + SIMD
  NO  ↓

Is performance critical?
  NO  → List[Float64] (easiest)
  YES → Reconsider above options
```

---

## Documentation

- **Full Research:** `/home/maskkiller/dev-coffee/repos/visage-ml/packages/mojo-audio/docs/BUFFER_TENSOR_RESEARCH.md`
- **Code Examples:** `/home/maskkiller/dev-coffee/repos/visage-ml/packages/mojo-audio/docs/MIGRATION_EXAMPLES.md`
- **Mojo SIMD Docs:** https://docs.modular.com/mojo/stdlib/builtin/simd/
- **Internal Learnings:** `/home/maskkiller/dev-coffee/repos/visage-ml/packages/mojo-audio/docs/SIMD_LEARNINGS.md`

---

## Current Status (mojo-audio)

**List-based implementation:**
- 165ms for 30s mel spectrogram
- 200x realtime (acceptable for MVP)
- Correct output, all tests passing

**Next optimization target:**
- UnsafePointer for hot loops → ~130ms
- NDBuffer for arrays → ~75ms
- MAX Tensor for matrix ops → ~20-25ms
- **Final: Competitive with librosa (15ms)**

---

**Remember:** Algorithm > Data structure > SIMD micro-optimizations
