# mojo-audio 🎵

> **High-performance audio signal processing library in Mojo**

[![Mojo](https://img.shields.io/badge/Mojo-0.26.1-orange?logo=fire)](https://docs.modular.com/mojo/)
[![Status](https://img.shields.io/badge/status-Alpha-yellow)]()

SIMD-optimized audio DSP operations for machine learning preprocessing. Built from scratch to understand signal processing fundamentals.

---

## Features

### ✅ Complete Implementation

**Window Functions** (Phase 1)
- `hann_window()` - Hann window (smooth taper to zero)
- `hamming_window()` - Hamming window (narrower main lobe)
- `apply_window()` - Apply window to signal

**FFT Operations** (Phase 2)
- `fft()` - Cooley-Tukey FFT with auto-padding
- `power_spectrum()` - Convert complex FFT to power
- `stft()` - Short-Time Fourier Transform
- Complex number arithmetic

**Mel Filterbank** (Phase 3)
- `hz_to_mel()` / `mel_to_hz()` - Scale conversions
- `create_mel_filterbank()` - Triangular mel filters
- `mel_spectrogram()` - **Complete Whisper preprocessing!**
- Output: **(80, ~3000) for 30s audio** ✓

**Performance** (Phase 4)
- Benchmark infrastructure
- Python comparison scripts
- Optimization opportunities documented
- Ready for SIMD acceleration

**Utilities**
- `pad_to_length()` - Zero-padding
- `rms_energy()` - Signal energy calculation
- `normalize_audio()` - Normalize to [-1, 1]
- `validate_whisper_audio()` - Whisper requirement checks

### 🚧 Future Enhancements

- SIMD-optimized implementations (10-50x speedup potential)
- MFCC features
- Additional window functions
- DGX Spark ARM-specific optimizations

---

## Quick Start

### Installation

From the parent repo:
```bash
cd visage-ml
pixi install
```

### Run Tests

```bash
pixi run audio-test
```

### Run Demos

```bash
pixi run audio-demo         # Mel spectrogram demo (main)
pixi run audio-demo-window  # Window functions
pixi run audio-demo-fft     # FFT operations
pixi run audio-demo-mel     # Full mel pipeline
```

### Run Benchmarks

```bash
pixi run audio-bench         # Mojo performance
pixi run audio-bench-python  # Python baseline (requires librosa)
```

---

## Usage

### Complete Whisper Preprocessing

```mojo
from audio import mel_spectrogram

fn main() raises:
    # Load 30s audio @ 16kHz (480,000 samples)
    var audio: List[Float64] = [...]

    # Get Whisper-compatible mel spectrogram
    var mel_spec = mel_spectrogram(audio)

    print("Shape:", len(mel_spec), "x", len(mel_spec[0]))
    # Output: Shape: 80 x 2998
    # ✓ Ready for Whisper model!
}
```

### Individual Operations

```mojo
from audio import hann_window, fft, stft, create_mel_filterbank

fn main() raises:
    // Window function
    var window = hann_window(400)

    // FFT
    var signal: List[Float64] = [...]
    var spectrum = fft(signal)

    // STFT (spectrogram)
    var audio: List[Float64] = [...]
    var spec = stft(audio, n_fft=400, hop_length=160)

    // Mel filterbank
    var filterbank = create_mel_filterbank(80, 400, 16000)
}
```

---

## API Reference

### Window Functions

**`hann_window(size: Int) -> List[Float64]`**
- Generates Hann window coefficients
- Formula: `w(n) = 0.5 * (1 - cos(2π * n / (N-1)))`
- Tapers to exactly 0 at edges
- Use for: General-purpose STFT

**`hamming_window(size: Int) -> List[Float64]`**
- Generates Hamming window coefficients
- Formula: `w(n) = 0.54 - 0.46 * cos(2π * n / (N-1))`
- Minimum value ~0.08 (doesn't reach 0)
- Use for: Better frequency selectivity

**`apply_window(signal: List[Float64], window: List[Float64]) -> List[Float64]`**
- Element-wise multiplication
- Raises error if lengths don't match

### Utilities

**`normalize_audio(signal: List[Float64]) -> List[Float64]`**
- Normalize to [-1.0, 1.0] range

**`rms_energy(signal: List[Float64]) -> Float64`**
- Compute RMS energy
- Useful for voice activity detection

**`pad_to_length(signal: List[Float64], target_length: Int) -> List[Float64]`**
- Zero-pad to target length

---

## Whisper Compatibility

### Requirements

Whisper expects:
- **Sample rate:** 16kHz
- **n_fft:** 400 samples (25ms window)
- **hop_length:** 160 samples (10ms hop)
- **n_mels:** 80 mel bins
- **Output shape:** (80, 3000) for 30s audio

### Validation

```mojo
from audio import validate_whisper_audio, WHISPER_SAMPLE_RATE

var audio: List[Float64] = [...]  # 30 seconds
var is_valid = validate_whisper_audio(audio, 30)
# Checks: len(audio) == 30 * 16000 = 480,000 samples
```

---

## Development Status

| Component | Status | Validated |
|-----------|--------|-----------|
| Window Functions | ✅ Complete | ✅ All tests pass |
| FFT Operations | ✅ Complete | ✅ All tests pass |
| STFT | ✅ Complete | ✅ All tests pass |
| Mel Filterbank | ✅ Complete | ✅ All tests pass |
| Mel Spectrogram | ✅ Complete | ✅ (80, 2998) output |
| Benchmarks | ✅ Complete | ✅ Infrastructure ready |
| SIMD Optimization | 📋 Opportunities documented | - |

---

## Roadmap

**Phase 1: Window Functions** ✅ Complete
- [x] Hann window
- [x] Hamming window
- [x] Tests & validation
- [x] Example usage

**Phase 2: FFT Operations** ✅ Complete
- [x] Cooley-Tukey FFT
- [x] Auto-padding to power of 2
- [x] Power spectrum
- [x] STFT implementation
- [x] Validated with tests

**Phase 3: Mel Features** ✅ Complete
- [x] Hz ↔ Mel conversion
- [x] Mel filterbank matrix (80 × 201)
- [x] Mel spectrogram pipeline
- [x] Whisper compatibility: (80, 2998) ✓

**Phase 4: Optimization** ✅ Infrastructure Ready
- [x] Benchmark framework
- [x] Python comparison baseline
- [x] Optimization guide documentation
- [ ] SIMD implementations (future work)

**Future Phases:**
- [ ] SIMD-optimized implementations
- [ ] DGX Spark ARM optimizations
- [ ] MFCC features
- [ ] Real-time streaming support

---

## Why Mojo?

**Performance:**
- SIMD operations for DSP
- C-level speed for audio processing
- Zero-copy tensor operations

**Correctness:**
- Built from scratch with clear math
- Validated against reference implementations
- Educational - understand every line

**Integration:**
- Native MAX Engine compatibility
- Seamless with Mojo ML pipelines
- No Python dependencies

---

## Contributing

This is part of the Visage ML project. Contributions welcome!

**Current focus:** Implementing FFT operations (Phase 2)

**Areas for contribution:**
- SIMD optimization
- Additional window functions
- Performance benchmarks
- Documentation

---

## License

MIT License - see parent repository

---

## Related Projects

- **Visage ML** - Parent ML library with neural networks
- **Mojo** - The Mojo programming language
- **MAX Engine** - Modular's AI inference engine

---

**Part of the Visage ML ecosystem** 🔥
