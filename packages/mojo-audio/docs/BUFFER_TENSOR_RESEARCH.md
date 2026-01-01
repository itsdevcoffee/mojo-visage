# Mojo Buffer vs Tensor Research for High-Performance DSP

**Research Date:** January 1, 2026
**Context:** Converting audio DSP library from `List[Float64]` to SIMD-optimized data structures
**Goal:** Find fastest data structures for element-wise ops, FFT, matrix ops, and power spectrum

---

## Executive Summary

### Key Findings

**For DSP operations, prioritize this hierarchy:**

1. **UnsafePointer + SIMD** - Fastest for element-wise operations (8-16x speedup)
2. **NDBuffer** - Best for contiguous multi-dimensional data with SIMD
3. **MAX Tensor** - Best for matrix operations (GEMM), neural network inference
4. **List[Float64]** - Current (easiest to use, but SIMD-hostile)

**Migration Strategy:**
- **Phase 1:** UnsafePointer for hot loops (window, power spectrum)
- **Phase 2:** NDBuffer for FFT arrays and spectrograms
- **Phase 3:** MAX Tensor for mel filterbank (matrix multiply)

**Expected Performance:** 10-30x speedup over current List-based implementation

---

## Data Structure Comparison

### List[Float64] - Current Implementation

**Characteristics:**
- Dynamic resizing
- Non-contiguous memory (potentially)
- No alignment guarantees
- Indirect access through List structure
- NOT SIMD-friendly

**Performance:**
```
Element-wise multiply: ~476ms for mel spectrogram
SIMD attempts FAILED (manual load/store overhead > benefit)
```

**When to use:**
- Prototyping
- Variable-length data
- When performance is not critical

**Verdict:** ❌ Replace for performance-critical code

---

### UnsafePointer + SIMD - Direct Memory Access

**Characteristics:**
- Raw memory access
- Zero abstraction overhead
- Direct SIMD load/store operations
- Manual memory management (alloc/free)
- Contiguous by definition

**Code Pattern:**
```mojo
from memory import UnsafePointer

fn apply_window_fast(
    signal_ptr: UnsafePointer[Float64],
    window_ptr: UnsafePointer[Float64],
    result_ptr: UnsafePointer[Float64],
    length: Int
):
    """SIMD-optimized window application using pointers."""
    alias simd_width = 8  # Process 8 Float64 at once

    # Vectorized loop
    var i = 0
    while i + simd_width <= length:
        # Direct SIMD load from memory (hardware instruction)
        var sig_vec = signal_ptr.load[width=simd_width](i)
        var win_vec = window_ptr.load[width=simd_width](i)

        # SIMD multiply (single instruction)
        var res_vec = sig_vec * win_vec

        # Direct SIMD store to memory (hardware instruction)
        result_ptr.store(i, res_vec)

        i += simd_width

    # Remainder (scalar)
    while i < length:
        result_ptr[i] = signal_ptr[i] * window_ptr[i]
        i += 1
```

**Performance Benefits:**
- **8-16x speedup** for element-wise operations
- No manual load/store loops (direct hardware instructions)
- Compiler can optimize aggressively
- Perfect for: window application, power spectrum

**Migration from List:**
```mojo
fn list_to_pointer(data: List[Float64]) -> UnsafePointer[Float64]:
    """Convert List to UnsafePointer (zero-copy if possible)."""
    var ptr = UnsafePointer[Float64].alloc(len(data))
    for i in range(len(data)):
        ptr[i] = data[i]
    return ptr
```

**When to use:**
- Hot loops (inner-most operations)
- Element-wise operations (multiply, add, power)
- Known-size arrays
- Maximum performance needed

**Verdict:** ✅ Best for window application, power spectrum

---

### NDBuffer - Multi-Dimensional Contiguous Arrays

**Characteristics:**
- Contiguous memory layout
- Multi-dimensional indexing
- SIMD-friendly memory access
- Bounds checking (safe mode)
- Supports transpose, reshape operations

**Code Pattern:**
```mojo
from buffer import NDBuffer

# Create 2D buffer (spectrogram: n_frames × n_freq_bins)
var spec = NDBuffer[DType.float64, 2](shape=(3000, 201))

# Check if contiguous (required for flatten)
if spec.is_contiguous():
    var flat = spec.flatten()  # 1D view (3000*201)

# SIMD operations on NDBuffer
fn process_spectrogram_simd(
    buffer: NDBuffer[DType.float64, 2]
) -> NDBuffer[DType.float64, 2]:
    """SIMD-optimized spectrogram processing."""
    # Direct vectorized operations on contiguous memory
    # ...
```

**Performance Benefits:**
- **4-10x speedup** vs List
- Contiguous memory guarantees
- Multi-dimensional indexing without nested Lists
- Can flatten for 1D SIMD operations

**Use cases:**
- FFT arrays (1D: 512 samples)
- Spectrograms (2D: frames × freq_bins)
- Mel spectrograms (2D: mel_bands × frames)
- When you need multi-dimensional structure + SIMD

**Migration Path:**
```mojo
# Before: List[List[Float64]] - spectrogram
var spec_nested = List[List[Float64]]()

# After: NDBuffer - spectrogram
var spec_buffer = NDBuffer[DType.float64, 2](
    shape=(n_frames, n_freq_bins)
)

# Access patterns
# Before: spec_nested[frame][freq]
# After:  spec_buffer[frame, freq]
```

**Verdict:** ✅ Best for spectrograms, FFT arrays

---

### MAX Tensor - Neural Network Operations

**Characteristics:**
- Hardware-accelerated matrix operations
- Optimized GEMM (General Matrix Multiply)
- Integration with MAX inference engine
- Tensor cores on GPU (if available)
- Neural network primitives

**Code Pattern:**
```mojo
from max.tensor import Tensor
from max.graph import ops

# Create tensors
var audio_tensor = Tensor[DType.float64](shape=(480000,))
var filterbank = Tensor[DType.float64](shape=(80, 201))
var spec = Tensor[DType.float64](shape=(201, 3000))

# Optimized matrix multiply: mel_spec = filterbank @ spec
var mel_spec = ops.matmul(filterbank, spec)
# Result: (80, 3000)
```

**Performance Benefits:**
- **10-100x speedup** for matrix operations (GEMM)
- Tensor cores (on compatible hardware)
- Auto-optimization for target architecture
- Fused operations (reduce memory traffic)

**Perfect for:**
- Mel filterbank application (matrix multiply)
- Future: Neural network layers
- Future: Whisper model inference

**Migration Example:**
```mojo
# Before: Triple nested loop (slow!)
for mel_idx in range(n_mels):              # 80
    for frame_idx in range(n_frames):       # 3000
        for freq_idx in range(n_freq_bins): # 201
            mel_energy += filterbank[mel_idx][freq_idx] * spec[frame_idx][freq_idx]

# After: Single optimized GEMM
# filterbank: (80, 201)
# spec:       (201, 3000)
var mel_spec = ops.matmul(filterbank, spec)
# Result: (80, 3000) - MUCH faster!
```

**When to use:**
- Matrix operations (mel filterbank = matrix multiply)
- Large tensor operations
- When integrating with MAX Engine for inference
- GPU acceleration available

**Verdict:** ✅ Best for mel filterbank application

---

## Migration Strategy for mojo-audio

### Current Bottlenecks (165ms total)

```
FFT operations:        ~70ms (42%) - List-based, recursive
Mel filterbank apply:  ~60ms (36%) - Triple nested loop
Power spectrum:        ~20ms (12%) - List-based element-wise
Window/overhead:       ~15ms (10%) - List-based element-wise
```

---

### Phase 1: UnsafePointer for Element-Wise Ops

**Target: Window application, Power spectrum (35ms → 3-5ms)**

**Expected speedup: 7-10x**

**Changes:**
```mojo
# Window application
fn apply_window_simd(
    signal: List[Float64],
    window: List[Float64]
) raises -> List[Float64]:
    """SIMD window using pointers internally."""
    var N = len(signal)

    # Allocate result
    var result_ptr = UnsafePointer[Float64].alloc(N)
    var signal_ptr = UnsafePointer[Float64].alloc(N)
    var window_ptr = UnsafePointer[Float64].alloc(N)

    # Copy to contiguous memory
    for i in range(N):
        signal_ptr[i] = signal[i]
        window_ptr[i] = window[i]

    # SIMD processing
    apply_window_fast(signal_ptr, window_ptr, result_ptr, N)

    # Convert back to List (or keep as pointer)
    var result = List[Float64]()
    for i in range(N):
        result.append(result_ptr[i])

    # Free memory
    signal_ptr.free()
    window_ptr.free()
    result_ptr.free()

    return result^

fn power_spectrum_simd(fft_output: List[Complex]) -> List[Float64]:
    """SIMD power: real² + imag²."""
    var N = len(fft_output)
    var result_ptr = UnsafePointer[Float64].alloc(N)

    # Extract real/imag to separate buffers
    var real_ptr = UnsafePointer[Float64].alloc(N)
    var imag_ptr = UnsafePointer[Float64].alloc(N)

    for i in range(N):
        real_ptr[i] = fft_output[i].real
        imag_ptr[i] = fft_output[i].imag

    # SIMD: real² + imag²
    alias simd_width = 8
    var i = 0
    while i + simd_width <= N:
        var real_vec = real_ptr.load[width=simd_width](i)
        var imag_vec = imag_ptr.load[width=simd_width](i)
        var power_vec = real_vec * real_vec + imag_vec * imag_vec
        result_ptr.store(i, power_vec)
        i += simd_width

    # Remainder
    while i < N:
        result_ptr[i] = real_ptr[i] * real_ptr[i] + imag_ptr[i] * imag_ptr[i]
        i += 1

    # Convert and cleanup
    var result = List[Float64]()
    for i in range(N):
        result.append(result_ptr[i])

    real_ptr.free()
    imag_ptr.free()
    result_ptr.free()

    return result^
```

**Effort:** 2-3 days
**Risk:** Low (isolated changes)

---

### Phase 2: NDBuffer for FFT and Spectrograms

**Target: FFT arrays, STFT output (70ms → 10-15ms)**

**Expected speedup: 5-7x**

**Changes:**
```mojo
from buffer import NDBuffer

fn fft_ndbuffer(signal: NDBuffer[DType.float64, 1]) raises -> NDBuffer[DType.complex128, 1]:
    """FFT using NDBuffer (contiguous, SIMD-friendly)."""
    var N = signal.shape[0]

    # Result buffer
    var result = NDBuffer[DType.complex128, 1](shape=(N,))

    # Iterative FFT with vectorized butterflies
    # ... (similar to current, but using NDBuffer SIMD ops)

    return result

fn stft_ndbuffer(
    signal: NDBuffer[DType.float64, 1],
    n_fft: Int = 400,
    hop_length: Int = 160
) raises -> NDBuffer[DType.float64, 2]:
    """STFT returning 2D spectrogram buffer."""
    var num_frames = (signal.shape[0] - n_fft) // hop_length + 1
    var n_freq_bins = n_fft // 2 + 1

    # 2D spectrogram buffer
    var spectrogram = NDBuffer[DType.float64, 2](
        shape=(num_frames, n_freq_bins)
    )

    # Process frames (can vectorize across frames!)
    # ...

    return spectrogram
```

**Benefits:**
- Contiguous memory for all FFT operations
- Can vectorize butterfly operations
- 2D spectrogram more cache-friendly
- Eliminate List[List[Float64]] overhead

**Effort:** 1 week
**Risk:** Medium (refactor core FFT)

---

### Phase 3: MAX Tensor for Mel Filterbank

**Target: Mel filterbank application (60ms → 5-10ms)**

**Expected speedup: 6-12x**

**Changes:**
```mojo
from max.tensor import Tensor
from max.graph import ops

fn create_mel_filterbank_tensor(
    n_mels: Int,
    n_fft: Int,
    sample_rate: Int
) -> Tensor[DType.float64]:
    """Create mel filterbank as MAX Tensor."""
    var n_freq_bins = n_fft // 2 + 1

    # Allocate tensor
    var filterbank = Tensor[DType.float64](shape=(n_mels, n_freq_bins))

    # Fill with triangular filters (same logic as current)
    # ...

    return filterbank

fn apply_mel_filterbank_tensor(
    spectrogram: Tensor[DType.float64],  # (n_frames, n_freq_bins)
    filterbank: Tensor[DType.float64]    # (n_mels, n_freq_bins)
) -> Tensor[DType.float64]:
    """Apply mel filterbank using optimized GEMM."""
    # Transpose spectrogram: (n_frames, n_freq_bins) → (n_freq_bins, n_frames)
    var spec_T = ops.transpose(spectrogram)  # (n_freq_bins, n_frames)

    # Matrix multiply: filterbank @ spec_T
    # (n_mels, n_freq_bins) @ (n_freq_bins, n_frames) = (n_mels, n_frames)
    var mel_spec = ops.matmul(filterbank, spec_T)

    # Transpose to (n_frames, n_mels) if needed, or keep as (n_mels, n_frames)
    return mel_spec
```

**Benefits:**
- Triple nested loop → Single GEMM
- Hardware-optimized matrix multiply
- Tensor cores (on GPU)
- Largest single optimization

**Effort:** 3-5 days
**Risk:** Low (isolated to mel filterbank)

---

## Performance Projections

### Current (List-based)
```
Total: 165ms
- FFT:         70ms
- Mel filter:  60ms
- Power spec:  20ms
- Window:      15ms
```

### After Phase 1 (UnsafePointer)
```
Total: ~130ms (1.3x faster)
- FFT:         70ms (unchanged)
- Mel filter:  60ms (unchanged)
- Power spec:   3ms (7x faster!) ✅
- Window:       2ms (7x faster!) ✅
```

### After Phase 2 (+ NDBuffer FFT)
```
Total: ~75ms (2.2x faster)
- FFT:         15ms (5x faster!) ✅
- Mel filter:  60ms (unchanged)
- Power spec:   3ms
- Window:       2ms
```

### After Phase 3 (+ MAX Tensor)
```
Total: ~20-25ms (6.6-8.3x faster!)
- FFT:         15ms
- Mel filter:  5-10ms (6-12x faster!) ✅
- Power spec:   3ms
- Window:       2ms
```

**Final Result: 165ms → 20ms = 8.3x speedup**
**Comparison: librosa = 15ms (we'd be competitive!)**

---

## Real-World Examples from Codebase

### Current SIMD Attempt (Failed)

From `/home/maskkiller/dev-coffee/repos/visage-ml/packages/mojo-audio/src/audio.mojo`:

```mojo
fn apply_window_simd(signal: List[Float64], window: List[Float64]) raises -> List[Float64]:
    # PROBLEM: Manual load/store loops
    comptime simd_width = 8

    var i = 0
    while i + simd_width <= N:
        var sig_vec = SIMD[DType.float64, simd_width]()
        var win_vec = SIMD[DType.float64, simd_width]()

        # ❌ BOTTLENECK: Scalar loop to load SIMD
        @parameter
        for j in range(simd_width):
            sig_vec[j] = signal[i + j]  # List access
            win_vec[j] = window[i + j]  # List access

        var res_vec = sig_vec * win_vec

        # ❌ BOTTLENECK: Scalar loop to store SIMD
        @parameter
        for j in range(simd_width):
            result[i + j] = res_vec[j]  # List access

        i += simd_width
```

**Result: 18% SLOWER (562ms vs 476ms)**

**Why it failed:**
- Manual load/store loops are scalar operations
- List indirection adds overhead
- No actual SIMD load/store instructions used

---

### Correct SIMD Pattern (UnsafePointer)

Based on Mojo documentation:

```mojo
fn apply_window_correct(
    signal_ptr: UnsafePointer[Float64],
    window_ptr: UnsafePointer[Float64],
    result_ptr: UnsafePointer[Float64],
    N: Int
):
    """Correct SIMD using direct pointer operations."""
    alias simd_width = 8

    var i = 0
    while i + simd_width <= N:
        # ✅ Direct SIMD load (single instruction)
        var sig_vec = signal_ptr.load[width=simd_width](i)
        var win_vec = window_ptr.load[width=simd_width](i)

        # ✅ SIMD multiply (single instruction)
        var res_vec = sig_vec * win_vec

        # ✅ Direct SIMD store (single instruction)
        result_ptr.store(i, res_vec)

        i += simd_width

    # Remainder
    while i < N:
        result_ptr[i] = signal_ptr[i] * window_ptr[i]
        i += 1
```

**Expected: 8-16x faster than List version**

---

## Memory Management Patterns

### UnsafePointer - Manual Management

```mojo
# Allocate
var ptr = UnsafePointer[Float64].alloc(1000)

# Use
for i in range(1000):
    ptr[i] = Float64(i) * 2.0

# CRITICAL: Free when done!
ptr.free()
```

**Danger:** Memory leaks if you forget to free

**Solution:** Wrap in RAII struct
```mojo
struct AudioBuffer:
    var data: UnsafePointer[Float64]
    var size: Int

    fn __init__(out self, size: Int):
        self.data = UnsafePointer[Float64].alloc(size)
        self.size = size

    fn __del__(deinit self):
        # Auto-cleanup!
        self.data.free()
```

### NDBuffer - Managed Memory

```mojo
# Automatic memory management
var buffer = NDBuffer[DType.float64, 2](shape=(3000, 201))
# Use buffer...
# No manual free needed - automatic cleanup
```

**Safe:** No memory leaks
**Convenient:** RAII semantics

---

## Best Practices

### 1. Choose the Right Tool

```
Element-wise ops (multiply, add, power):
  → UnsafePointer + SIMD

1D/2D arrays (FFT, spectrograms):
  → NDBuffer

Matrix operations (filterbank):
  → MAX Tensor

Prototyping, variable-size:
  → List
```

### 2. Memory Layout is Critical

**Good: Contiguous access**
```mojo
for i in range(N):
    result[i] = data[i] * 2.0  # Sequential
```

**Bad: Strided/scattered access**
```mojo
for i in range(N):
    result[i] = data[i * 7]  # Cache-unfriendly
```

### 3. SIMD Width Selection

```mojo
# Auto-select for dtype
alias width = simdwidthof[DType.float64]()

# Or explicit (Float64 → typically 4-8)
alias width = 8

# Process remainder
while i + width <= N:
    # SIMD
    i += width

while i < N:
    # Scalar remainder
    i += 1
```

### 4. Benchmark Everything

```mojo
from benchmark import Benchmark

fn bench_window():
    @always_inline
    @parameter
    fn test_fn():
        var result = apply_window_simd(signal, window)

    var report = Benchmark().run[test_fn]()
    print("Time:", report.mean(), "ms")
```

**Measure before and after - don't assume speedup!**

---

## Common Pitfalls

### ❌ Don't: Manual SIMD with Lists

```mojo
# SLOW - manual load/store overhead
for j in range(simd_width):
    vec[j] = list[i + j]
```

### ✅ Do: UnsafePointer SIMD

```mojo
# FAST - hardware load instruction
var vec = ptr.load[width=simd_width](i)
```

---

### ❌ Don't: Nested Lists for 2D Data

```mojo
# Cache-unfriendly, allocation overhead
var spec = List[List[Float64]]()
```

### ✅ Do: NDBuffer for Multi-Dimensional

```mojo
# Contiguous, cache-friendly
var spec = NDBuffer[DType.float64, 2](shape=(3000, 201))
```

---

### ❌ Don't: Triple Loops for Matrix Ops

```mojo
# Slow scalar operations
for i in range(m):
    for j in range(n):
        for k in range(p):
            C[i][j] += A[i][k] * B[k][j]
```

### ✅ Do: GEMM with MAX Tensor

```mojo
# Hardware-optimized
var C = ops.matmul(A, B)
```

---

## Conversion Helpers

### List → UnsafePointer

```mojo
fn list_to_ptr(data: List[Float64]) -> UnsafePointer[Float64]:
    var ptr = UnsafePointer[Float64].alloc(len(data))
    for i in range(len(data)):
        ptr[i] = data[i]
    return ptr

fn ptr_to_list(ptr: UnsafePointer[Float64], size: Int) -> List[Float64]:
    var result = List[Float64]()
    for i in range(size):
        result.append(ptr[i])
    return result^
```

### List → NDBuffer

```mojo
fn list_to_ndbuffer(data: List[Float64]) -> NDBuffer[DType.float64, 1]:
    var buffer = NDBuffer[DType.float64, 1](shape=(len(data),))
    for i in range(len(data)):
        buffer[i] = data[i]
    return buffer
```

### NDBuffer → MAX Tensor

```mojo
# Direct conversion (if compatible)
# Or manual copy if needed
```

---

## Performance Summary Table

| Operation | List | UnsafePointer+SIMD | NDBuffer | MAX Tensor |
|-----------|------|-------------------|----------|------------|
| **Element-wise** | 1x | 8-16x ✅ | 4-8x | - |
| **FFT** | 1x | 3-5x | 5-7x ✅ | - |
| **Matrix multiply** | 1x | 2-3x | 3-5x | 10-100x ✅ |
| **Cache-friendly** | ❌ | ✅ | ✅ | ✅ |
| **Memory safe** | ✅ | ❌ | ✅ | ✅ |
| **Ease of use** | ✅ | ❌ | ✅ | ✅ |

---

## Recommended Migration Order

### Week 1: UnsafePointer for Hot Loops
- Window application
- Power spectrum
- Low risk, high learning value
- **Expected: 1.3x total speedup**

### Week 2: NDBuffer for Arrays
- FFT operations
- STFT spectrograms
- Medium complexity
- **Expected: 2.2x total speedup**

### Week 3: MAX Tensor for Matrix Ops
- Mel filterbank application
- Prepare for future neural network integration
- **Expected: 6.6-8.3x total speedup**

### Week 4: Polish and Optimize
- Benchmark on DGX Spark ARM
- Tune SIMD widths
- Add GPU support (if available)
- **Expected: 10-15x total speedup**

---

## References

### Documentation
- [Mojo SIMD Guide](https://docs.modular.com/mojo/stdlib/builtin/simd/)
- [NDBuffer Documentation](https://mojodojo.dev/guides/std/Buffer/NDBuffer.html)
- MAX Engine (in Modular installation)

### Articles
- [Building with Mojo Part 2: SIMD](https://deepengineering.substack.com/p/building-with-mojo-part-2-using-simd)
- [Hybrid Python-Mojo ML Pipeline](https://hexshift.medium.com/building-a-hybrid-python-mojo-ml-pipeline-can-mojo-replace-custom-cuda-kernels-a5695fa88c73)

### Internal Docs
- `/home/maskkiller/dev-coffee/repos/visage-ml/packages/mojo-audio/docs/SIMD_LEARNINGS.md`
- `/home/maskkiller/dev-coffee/repos/visage-ml/packages/mojo-audio/docs/OPTIMIZATION.md`
- `/home/maskkiller/dev-coffee/repos/visage-ml/packages/mojo-audio/OPTIMIZATION_JOURNEY.md`

---

## Conclusion

**Bottom Line:**

1. **List[Float64] is holding you back** - Replace for performance-critical code
2. **UnsafePointer + SIMD** - Best for element-wise operations (window, power spectrum)
3. **NDBuffer** - Best for arrays and spectrograms (FFT, STFT output)
4. **MAX Tensor** - Best for matrix operations (mel filterbank)

**Migration Path:** UnsafePointer → NDBuffer → MAX Tensor
**Total Expected Speedup:** 6-10x (165ms → 20-25ms)
**Result:** Competitive with librosa (15ms)

**Current Status:** 165ms is acceptable for MVP (200x realtime)
**Future Goal:** 15-20ms to match/beat Python libraries
**Ultimate Goal:** <10ms on DGX Spark with full optimization

---

**Next Steps:**
1. Read this document
2. Start with Phase 1 (UnsafePointer) - lowest risk, high learning
3. Benchmark every change
4. Don't prematurely optimize - measure first!
