"""Tests for FFT operations."""

from audio import Complex, fft, power_spectrum, stft


fn abs(x: Float64) -> Float64:
    """Absolute value."""
    if x < 0:
        return -x
    return x


fn test_complex_operations() raises:
    """Test Complex number operations."""
    print("Testing Complex number operations...")

    var a = Complex(3.0, 4.0)
    var b = Complex(1.0, 2.0)

    # Test addition
    var sum = a + b
    assert_close(sum.real, 4.0, 1e-10, "Complex addition real part")
    assert_close(sum.imag, 6.0, 1e-10, "Complex addition imag part")

    # Test subtraction
    var diff = a - b
    assert_close(diff.real, 2.0, 1e-10, "Complex subtraction real part")
    assert_close(diff.imag, 2.0, 1e-10, "Complex subtraction imag part")

    # Test multiplication
    var prod = a * b
    # (3+4i)(1+2i) = 3 + 6i + 4i + 8i² = 3 + 10i - 8 = -5 + 10i
    assert_close(prod.real, -5.0, 1e-10, "Complex multiplication real part")
    assert_close(prod.imag, 10.0, 1e-10, "Complex multiplication imag part")

    # Test magnitude: |3+4i| = sqrt(9+16) = 5
    assert_close(a.magnitude(), 5.0, 1e-10, "Complex magnitude")

    # Test power: 3²+4² = 25
    assert_close(a.power(), 25.0, 1e-10, "Complex power")

    print("  ✓ Complex operations validated")


fn test_fft_simple() raises:
    """Test FFT on simple known cases."""
    print("Testing FFT simple cases...")

    # Test 1: DC signal (constant)
    var dc: List[Float64] = [1.0, 1.0, 1.0, 1.0]
    var fft_dc = fft(dc)

    # DC signal: all energy should be in bin 0
    assert_true(fft_dc[0].power() > 10.0, "DC energy in bin 0")
    assert_true(fft_dc[1].power() < 0.1, "Minimal energy in other bins")

    # Test 2: Impulse (delta function)
    var impulse: List[Float64] = [1.0, 0.0, 0.0, 0.0]
    var fft_impulse = fft(impulse)

    # Impulse: flat spectrum (all bins have equal magnitude)
    var mag0 = fft_impulse[0].magnitude()
    var mag1 = fft_impulse[1].magnitude()
    assert_close(mag0, mag1, 0.01, "Impulse has flat spectrum")

    print("  ✓ FFT simple cases validated")


fn test_fft_auto_padding() raises:
    """Test that FFT auto-pads to power of 2."""
    print("Testing FFT auto-padding...")

    # Power of 2: should work as-is
    var valid: List[Float64] = [1.0, 2.0, 3.0, 4.0]  # Length 4 = 2²
    var result1 = fft(valid)
    assert_equal(len(result1), 4, "Should keep power-of-2 length")

    # Not power of 2: should auto-pad
    var not_pow2: List[Float64] = [1.0, 2.0, 3.0]  # Length 3
    var result2 = fft(not_pow2)
    # Padded to next power of 2 = 4
    assert_equal(len(result2), 4, "Should pad to next power of 2")

    # Whisper n_fft=400: should pad to 512
    var whisper_size = List[Float64]()
    for _ in range(400):
        whisper_size.append(0.5)
    var result3 = fft(whisper_size)
    assert_equal(len(result3), 512, "Should pad 400 to 512")

    print("  ✓ Auto-padding works correctly")


fn test_power_spectrum() raises:
    """Test power spectrum computation."""
    print("Testing power spectrum...")

    var signal: List[Float64] = [1.0, 0.0, 1.0, 0.0]
    var fft_result = fft(signal)
    var power = power_spectrum(fft_result)

    # Power should be non-negative
    for i in range(len(power)):
        assert_true(power[i] >= 0.0, "Power should be non-negative")

    # Length should match FFT output
    assert_equal(len(power), len(fft_result), "Power spectrum length")

    print("  ✓ Power spectrum validated")


fn test_stft_dimensions() raises:
    """Test STFT output dimensions (critical for Whisper!)."""
    print("Testing STFT dimensions...")

    # Create 30s of audio at 16kHz
    var audio_30s = List[Float64]()
    var samples_30s = 30 * 16000  # 480,000 samples

    for i in range(samples_30s):
        audio_30s.append(0.1)  # Dummy audio

    # Compute STFT with Whisper parameters
    var spectrogram = stft(audio_30s, n_fft=400, hop_length=160)

    # Check number of frames
    var n_frames = len(spectrogram)
    var expected_frames = 3000

    print("  Computed frames:", n_frames)
    print("  Expected frames:", expected_frames)

    # Allow small tolerance (2998-3000 is acceptable)
    var frame_diff = n_frames - expected_frames
    if frame_diff < 0:
        frame_diff = -frame_diff

    assert_true(frame_diff <= 2, "Should have ~3000 frames for 30s audio")

    # Check frequency bins (n_fft/2 + 1 = 201)
    var n_freq_bins = len(spectrogram[0])
    var expected_bins = 400 // 2 + 1  # 201

    assert_equal(n_freq_bins, expected_bins, "Should have 201 frequency bins")

    print("  Spectrogram shape: (", n_freq_bins, ",", n_frames, ")")
    print("  Expected shape:    ( 201 , 3000 )")
    print("  ✓ STFT dimensions validated!")


fn test_stft_basic() raises:
    """Test basic STFT functionality."""
    print("Testing STFT basic functionality...")

    # Short signal for testing
    var signal = List[Float64]()
    for i in range(1024):  # 1024 samples
        signal.append(0.5)

    var spec = stft(signal, n_fft=256, hop_length=128)

    # Should produce some frames
    assert_true(len(spec) > 0, "STFT should produce frames")

    # Each frame should have n_fft/2+1 bins
    assert_equal(len(spec[0]), 129, "Each frame should have 129 bins (256/2+1)")

    print("  ✓ STFT basic functionality validated")


# ==============================================================================
# Test Helpers
# ==============================================================================

fn assert_equal(value: Int, expected: Int, message: String) raises:
    """Assert integer equality."""
    if value != expected:
        raise Error(message + " (got " + String(value) + ", expected " + String(expected) + ")")


fn assert_close(value: Float64, expected: Float64, tolerance: Float64, message: String) raises:
    """Assert float values are close."""
    if abs(value - expected) > tolerance:
        raise Error(message + " (got " + String(value) + ", expected " + String(expected) + ")")


fn assert_true(condition: Bool, message: String) raises:
    """Assert condition is true."""
    if not condition:
        raise Error(message)


# ==============================================================================
# Test Runner
# ==============================================================================

fn main() raises:
    """Run all FFT tests."""
    print("\n" + "="*60)
    print("mojo-audio: FFT Operations Tests")
    print("="*60 + "\n")

    test_complex_operations()
    test_fft_simple()
    test_fft_auto_padding()
    test_power_spectrum()
    test_stft_basic()
    test_stft_dimensions()

    print("\n" + "="*60)
    print("✓ All FFT tests passed!")
    print("="*60 + "\n")
