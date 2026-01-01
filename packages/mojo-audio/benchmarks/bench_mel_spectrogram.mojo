"""
Benchmark mel spectrogram performance.

Measures time to compute mel spectrogram for various audio lengths.
"""

from audio import mel_spectrogram
from time import perf_counter_ns


fn benchmark_mel_spec(audio_seconds: Int, iterations: Int) raises:
    """Benchmark mel spectrogram for given audio length."""
    print("Benchmarking", audio_seconds, "seconds of audio...")

    # Create test audio
    var audio = List[Float64]()
    for _ in range(audio_seconds * 16000):
        audio.append(0.1)

    # Warmup
    _ = mel_spectrogram(audio)

    # Benchmark
    var start = perf_counter_ns()

    for _ in range(iterations):
        _ = mel_spectrogram(audio)

    var end = perf_counter_ns()

    var total_ns = end - start
    var avg_ms = Float64(total_ns) / Float64(iterations) / 1_000_000.0

    print("  Iterations:", iterations)
    print("  Avg time:  ", avg_ms, "ms")
    print("  Throughput:", Float64(audio_seconds) / (avg_ms / 1000.0), "x realtime")
    print()


fn main() raises:
    print("\n" + "="*70)
    print("Mel Spectrogram Performance Benchmark")
    print("="*70 + "\n")

    print("Testing mel_spectrogram() performance on various audio lengths:")
    print()

    # Benchmark different lengths
    benchmark_mel_spec(1, 10)    # 1s audio, 10 iterations
    benchmark_mel_spec(10, 5)    # 10s audio, 5 iterations
    benchmark_mel_spec(30, 3)    # 30s audio (Whisper), 3 iterations

    print("="*70)
    print("Benchmark complete!")
    print()
    print("Next steps:")
    print("  - Compare against librosa (Python)")
    print("  - Add SIMD optimizations")
    print("  - Profile hotspots")
    print("="*70 + "\n")
