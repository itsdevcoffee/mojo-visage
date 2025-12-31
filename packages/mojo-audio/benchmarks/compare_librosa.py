"""
Compare mojo-audio performance against librosa (Python standard).

This script benchmarks librosa's mel spectrogram computation
to provide a performance baseline for comparison.
"""

import numpy as np
import time

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("⚠️  librosa not installed - install with: pip install librosa")


def benchmark_librosa_mel(duration_seconds, iterations=5):
    """Benchmark librosa mel spectrogram."""
    if not LIBROSA_AVAILABLE:
        return

    # Create test audio
    sr = 16000
    audio = np.random.rand(duration_seconds * sr).astype(np.float32) * 0.1

    # Whisper parameters
    n_fft = 400
    hop_length = 160
    n_mels = 80

    # Warmup
    _ = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()

        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )

        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    avg_time = np.mean(times)
    std_time = np.std(times)
    throughput = duration_seconds / (avg_time / 1000.0)

    print(f"Benchmarking {duration_seconds}s audio (librosa)...")
    print(f"  Iterations: {iterations}")
    print(f"  Avg time:   {avg_time:.2f}ms ± {std_time:.2f}ms")
    print(f"  Throughput: {throughput:.1f}x realtime")
    print(f"  Output shape: {mel_spec.shape}")
    print()

    return avg_time, mel_spec.shape


def main():
    print("="*70)
    print("librosa Mel Spectrogram Benchmark (Python Baseline)")
    print("="*70)
    print()

    if not LIBROSA_AVAILABLE:
        print("Please install librosa to run benchmarks:")
        print("  pip install librosa")
        return

    print("Testing librosa mel spectrogram performance:")
    print()

    # Benchmark different lengths
    benchmark_librosa_mel(1, iterations=10)
    benchmark_librosa_mel(10, iterations=5)
    benchmark_librosa_mel(30, iterations=3)  # Whisper length

    print("="*70)
    print("Baseline established!")
    print()
    print("Compare with mojo-audio:")
    print("  pixi run audio-bench")
    print()
    print("Expected Mojo speedup:")
    print("  - Current (naive): ~1-2x faster")
    print("  - With SIMD: 10-50x faster potential")
    print("="*70)
    print()


if __name__ == "__main__":
    main()
