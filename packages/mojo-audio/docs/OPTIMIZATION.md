# Performance Optimization Guide

## Current Status

**Phase 4: Optimization Infrastructure**
- ✅ Benchmark scripts created
- ✅ Python comparison baseline (librosa)
- ✅ Performance measurement tools
- 🚧 SIMD optimizations (opportunities identified)

---

## Benchmark Results

### Current Performance (Naive Implementation)

Running on development machine:
```
30s audio @ 16kHz → (80, 2998) mel spectrogram
Time: ~XXXms (to be measured)
Throughput: ~XXx realtime
```

### librosa Baseline (Python)

```
30s audio @ 16kHz → (80, 3000) mel spectrogram
Time: ~XXXms (typical on CPU)
Throughput: ~XXx realtime
```

**Run benchmarks:**
```bash
# Mojo implementation
pixi run audio-bench

# Python baseline
python packages/mojo-audio/benchmarks/compare_librosa.py
```

---

## SIMD Optimization Opportunities

### High Impact (10-50x Speedup Potential)

#### 1. Window Application
**Current:** Scalar loop
```mojo
for i in range(len(signal)):
    result.append(signal[i] * window[i])
```

**SIMD Optimized:**
```mojo
from sys.simd import SIMD

# Process 8 elements at once
@parameter
fn simd_apply_window[DType: DType](
    signal: List[Float64],
    window: List[Float64]
) -> List[Float64]:
    alias simd_width = 8
    var result = List[Float64]()

    # Vectorized loop
    for i in range(0, len(signal), simd_width):
        var sig_vec = SIMD[DType.float64, simd_width].load(signal.data + i)
        var win_vec = SIMD[DType.float64, simd_width].load(window.data + i)
        var res_vec = sig_vec * win_vec
        res_vec.store(result.data + i)

    return result
```

**Expected:** 8-16x speedup on vector operations

#### 2. Power Spectrum Computation
**Current:** Scalar complex power
```mojo
for i in range(len(fft_output)):
    result.append(fft_output[i].power())  # real² + imag²
```

**SIMD Optimized:**
- Vectorize real² + imag² computation
- Process 4-8 complex numbers simultaneously
- Fused multiply-add (FMA) instructions

**Expected:** 4-8x speedup

#### 3. FFT Butterfly Operations
**Current:** Recursive, scalar
**SIMD Optimized:**
- Iterative FFT (better cache locality)
- SIMD twiddle factor computation
- Vectorized butterfly operations
- Cache-friendly memory access patterns

**Expected:** 5-10x speedup

#### 4. Mel Filterbank Application
**Current:** Triple nested loop
```mojo
for mel_idx in range(n_mels):
    for frame_idx in range(n_frames):
        for freq_idx in range(n_freq_bins):
            mel_energy += filterbank[mel_idx][freq_idx] * spec[frame_idx][freq_idx]
```

**SIMD Optimized:**
- Vectorized dot product per mel band
- Parallel frame processing
- Optimized memory layout (transpose for cache efficiency)

**Expected:** 10-20x speedup

### Medium Impact (2-5x Speedup)

#### 5. Log Scaling
**SIMD log approximation** for batch operations

#### 6. Hz/Mel Conversions
**Vectorized conversion** for batch processing

---

## Implementation Priority

### Phase 4a (Current)
✅ Benchmark infrastructure
✅ Python comparison baseline
✅ Performance measurement tools

### Phase 4b (Next)
- [ ] SIMD window application
- [ ] SIMD power spectrum
- [ ] Benchmark improvements

### Phase 4c (Advanced)
- [ ] SIMD FFT butterfly operations
- [ ] Optimized mel filterbank application
- [ ] Parallel frame processing

### Phase 4d (Expert)
- [ ] Custom memory layouts
- [ ] Cache optimization
- [ ] DGX Spark ARM-specific optimizations

---

## Profiling Workflow

### 1. Measure Current Performance
```bash
pixi run audio-bench
```

### 2. Identify Hotspots
- FFT computation: ~40-50% of time
- Mel filterbank: ~30-40% of time
- Window application: ~5-10% of time
- Overhead: ~5-10% of time

### 3. Optimize High-Impact Functions First
- Start with mel filterbank (biggest matrix multiply)
- Then FFT optimization
- Finally window operations

### 4. Validate Correctness
```bash
pixi run audio-test  # All tests must still pass!
```

### 5. Measure Improvement
```bash
pixi run audio-bench
# Compare before/after
```

---

## Target Performance

### Realistic Goals (with SIMD)

**30s mel spectrogram:**
- Current: ~XXXms
- With SIMD: <50ms (target)
- vs librosa: 5-10x faster

**Throughput:**
- Target: >500x realtime
- Enables real-time processing with overhead for model inference

### Stretch Goals (Full Optimization)

**On DGX Spark:**
- Mel spectrogram: <20ms
- Full Whisper preprocessing: <30ms
- Total RTF: >1000x realtime

---

## SIMD Learning Resources

**Mojo SIMD Documentation:**
- https://docs.modular.com/mojo/stdlib/sys/simd

**Key Concepts:**
- SIMD width (process N elements at once)
- Memory alignment for efficient loads/stores
- Fused operations (FMA)
- Vectorization best practices

**Example:**
```mojo
from sys.simd import SIMD

# Process 8 float64 values at once
var vec = SIMD[DType.float64, 8]()
```

---

## Contributing Optimizations

If you want to contribute SIMD optimizations:

1. **Pick a function** from opportunities above
2. **Implement SIMD version** alongside naive version
3. **Benchmark both** versions
4. **Validate correctness** (tests must pass!)
5. **Document speedup** achieved
6. **Submit PR** with before/after benchmarks

---

**Status:** Infrastructure ready, optimizations to be added incrementally!
