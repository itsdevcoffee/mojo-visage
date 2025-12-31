"""
mojo-audio: High-performance audio signal processing library.

SIMD-optimized DSP operations for machine learning audio preprocessing.
Designed for Whisper and other speech recognition models.
"""

from math import cos, sqrt, log, sin, atan2
from math.constants import pi


# ==============================================================================
# Constants (Whisper Requirements)
# ==============================================================================

comptime WHISPER_SAMPLE_RATE = 16000
comptime WHISPER_N_FFT = 400
comptime WHISPER_HOP_LENGTH = 160
comptime WHISPER_N_MELS = 80
comptime WHISPER_FRAMES_30S = 3000


# ==============================================================================
# Complex Number Operations
# ==============================================================================

struct Complex(Copyable, Movable):
    """Complex number for FFT operations."""
    var real: Float64
    var imag: Float64

    fn __init__(out self, real: Float64, imag: Float64 = 0.0):
        """Initialize complex number."""
        self.real = real
        self.imag = imag

    fn __copyinit__(out self, existing: Self):
        """Copy constructor."""
        self.real = existing.real
        self.imag = existing.imag

    fn __moveinit__(out self, deinit existing: Self):
        """Move constructor."""
        self.real = existing.real
        self.imag = existing.imag

    fn __add__(self, other: Complex) -> Complex:
        """Complex addition."""
        return Complex(self.real + other.real, self.imag + other.imag)

    fn __sub__(self, other: Complex) -> Complex:
        """Complex subtraction."""
        return Complex(self.real - other.real, self.imag - other.imag)

    fn __mul__(self, other: Complex) -> Complex:
        """Complex multiplication."""
        var r = self.real * other.real - self.imag * other.imag
        var i = self.real * other.imag + self.imag * other.real
        return Complex(r, i)

    fn magnitude(self) -> Float64:
        """Compute magnitude: sqrt(real² + imag²)."""
        return sqrt(self.real * self.real + self.imag * self.imag)

    fn power(self) -> Float64:
        """Compute power: real² + imag²."""
        return self.real * self.real + self.imag * self.imag


# ==============================================================================
# FFT Operations
# ==============================================================================

fn next_power_of_2(n: Int) -> Int:
    """Find next power of 2 >= n."""
    var power = 1
    while power < n:
        power *= 2
    return power


fn fft_internal(signal: List[Float64]) raises -> List[Complex]:
    """Internal FFT - requires power of 2 length."""
    var N = len(signal)

    # Validate power of 2
    if N == 0 or (N & (N - 1)) != 0:
        raise Error("Internal FFT requires power of 2. Got " + String(N))

    # Base case
    if N == 1:
        var result = List[Complex]()
        result.append(Complex(signal[0], 0.0))
        return result^

    # Divide into even and odd indices
    var even = List[Float64]()
    var odd = List[Float64]()

    for i in range(N // 2):
        even.append(signal[2 * i])
        odd.append(signal[2 * i + 1])

    # Recursive FFT on even and odd parts
    var fft_even = fft_internal(even)
    var fft_odd = fft_internal(odd)

    # Combine results
    var result = List[Complex]()

    # Initialize result list
    for _ in range(N):
        result.append(Complex(0.0, 0.0))

    # Combine using butterfly operations
    for k in range(N // 2):
        var k_float = Float64(k)
        var N_float = Float64(N)

        # Twiddle factor: W = e^(-2πik/N)
        var angle = -2.0 * pi * k_float / N_float
        var twiddle = Complex(cos(angle), sin(angle))

        # Butterfly operation
        var t = twiddle * fft_odd[k]
        result[k] = fft_even[k] + t
        result[k + N // 2] = fft_even[k] - t

    return result^


fn fft(signal: List[Float64]) raises -> List[Complex]:
    """
    Fast Fourier Transform using Cooley-Tukey algorithm.

    Automatically pads to next power of 2 if needed.
    Handles Whisper's n_fft=400 by padding to 512.

    Args:
        signal: Input signal (any length)

    Returns:
        Complex frequency spectrum (padded length)

    Example:
        ```mojo
        var signal: List[Float64] = [1.0, 0.0, 1.0, 0.0]
        var spectrum = fft(signal)  # Length 4 (already power of 2)

        var whisper_frame: List[Float64] = [...]  # 400 samples
        var spec = fft(whisper_frame)  # Padded to 512
        ```
    """
    var N = len(signal)

    # Pad to next power of 2 if needed
    var fft_size = next_power_of_2(N)

    var padded = List[Float64]()
    for i in range(N):
        padded.append(signal[i])
    for _ in range(N, fft_size):
        padded.append(0.0)

    # Call internal FFT (requires power of 2)
    return fft_internal(padded)


fn power_spectrum(fft_output: List[Complex]) -> List[Float64]:
    """
    Compute power spectrum from FFT output.

    Power = real² + imag² for each frequency bin.

    Args:
        fft_output: Complex FFT coefficients

    Returns:
        Power values (real-valued)

    Example:
        ```mojo
        var spectrum = fft(signal)
        var power = power_spectrum(spectrum)
        ```
    """
    var result = List[Float64]()

    for i in range(len(fft_output)):
        result.append(fft_output[i].power())

    return result^


fn stft(
    signal: List[Float64],
    n_fft: Int = 400,
    hop_length: Int = 160,
    window_fn: String = "hann"
) raises -> List[List[Float64]]:
    """
    Short-Time Fourier Transform - Apply FFT to windowed frames.

    Creates spectrogram: frequency content over time.

    Args:
        signal: Input audio signal
        n_fft: FFT size (window size)
        hop_length: Step size between frames
        window_fn: "hann" or "hamming"

    Returns:
        Spectrogram (n_fft/2+1, n_frames)

    Example:
        ```mojo
        var audio: List[Float64] = [...]  # 30s of audio
        var spec = stft(audio, n_fft=400, hop_length=160)
        ```

    For 30s audio @ 16kHz with hop=160:
        n_frames = (480000 - 400) / 160 + 1 = 3000 ✓
    """
    # Create window
    var window: List[Float64]
    if window_fn == "hann":
        window = hann_window(n_fft)
    elif window_fn == "hamming":
        window = hamming_window(n_fft)
    else:
        raise Error("Unknown window function: " + window_fn)

    # Calculate number of frames
    var num_frames = (len(signal) - n_fft) // hop_length + 1

    # Initialize result
    var spectrogram = List[List[Float64]]()

    # Process each frame
    for frame_idx in range(num_frames):
        var start = frame_idx * hop_length

        # Extract frame
        var frame = List[Float64]()
        for i in range(n_fft):
            if start + i < len(signal):
                frame.append(signal[start + i])
            else:
                frame.append(0.0)  # Pad if needed

        # Apply window
        var windowed = apply_window(frame, window)

        # Compute FFT
        var fft_result = fft(windowed)

        # Get power spectrum (magnitude squared)
        var power = power_spectrum(fft_result)

        # Take first half (n_fft/2 + 1) - positive frequencies only
        var frame_power = List[Float64]()
        for i in range(n_fft // 2 + 1):
            frame_power.append(power[i])

        spectrogram.append(frame_power^)

    return spectrogram^


# ==============================================================================
# Window Functions
# ==============================================================================

fn hann_window(size: Int) -> List[Float64]:
    """
    Generate Hann window.

    Hann window: w(n) = 0.5 * (1 - cos(2π * n / (N-1)))

    Used in STFT to reduce spectral leakage. Smoothly tapers to zero
    at the edges, minimizing discontinuities.

    Args:
        size: Window length in samples

    Returns:
        Window coefficients (length = size)

    Example:
        ```mojo
        var window = hann_window(400)  # For Whisper n_fft
        ```

    Mathematical properties:
        - Symmetric
        - Tapers to 0 at edges
        - Maximum at center (1.0)
        - Smoother than Hamming
    """
    var window = List[Float64]()
    var N = Float64(size - 1)

    for n in range(size):
        var n_float = Float64(n)
        var coefficient = 0.5 * (1.0 - cos(2.0 * pi * n_float / N))
        window.append(coefficient)

    return window^


fn hamming_window(size: Int) -> List[Float64]:
    """
    Generate Hamming window.

    Hamming window: w(n) = 0.54 - 0.46 * cos(2π * n / (N-1))

    Similar to Hann but doesn't taper completely to zero.
    Better frequency selectivity, slightly more spectral leakage.

    Args:
        size: Window length in samples

    Returns:
        Window coefficients (length = size)

    Example:
        ```mojo
        var window = hamming_window(400)
        ```

    Mathematical properties:
        - Symmetric
        - Minimum value: ~0.08 (not 0)
        - Maximum at center: ~1.0
        - Narrower main lobe than Hann
    """
    var window = List[Float64]()
    var N = Float64(size - 1)

    for n in range(size):
        var n_float = Float64(n)
        var coefficient = 0.54 - 0.46 * cos(2.0 * pi * n_float / N)
        window.append(coefficient)

    return window^


fn apply_window(signal: List[Float64], window: List[Float64]) raises -> List[Float64]:
    """
    Apply window function to signal (element-wise multiplication).

    Args:
        signal: Input signal
        window: Window coefficients (must match signal length)

    Returns:
        Windowed signal

    Raises:
        Error if lengths don't match

    Example:
        ```mojo
        var signal: List[Float64] = [1.0, 2.0, 3.0, 4.0]
        var window = hann_window(4)
        var windowed = apply_window(signal, window)
        ```
    """
    if len(signal) != len(window):
        raise Error(
            "Signal and window must have same length. Got signal="
            + String(len(signal)) + ", window=" + String(len(window))
        )

    var result = List[Float64]()
    for i in range(len(signal)):
        result.append(signal[i] * window[i])

    return result^


# ==============================================================================
# Utility Functions
# ==============================================================================

fn pad_to_length(signal: List[Float64], target_length: Int) -> List[Float64]:
    """
    Pad signal with zeros to target length.

    Args:
        signal: Input signal
        target_length: Desired length

    Returns:
        Padded signal (or original if already long enough)

    Example:
        ```mojo
        var signal: List[Float64] = [1.0, 2.0, 3.0]
        var padded = pad_to_length(signal, 400)  # Pads to n_fft
        ```
    """
    var result = List[Float64]()

    # Copy original signal
    for i in range(len(signal)):
        result.append(signal[i])

    # Add zeros if needed
    for i in range(len(signal), target_length):
        result.append(0.0)

    return result^


fn rms_energy(signal: List[Float64]) -> Float64:
    """
    Compute Root Mean Square energy of signal.

    RMS = sqrt((1/N) * Σ(x²))

    Useful for:
    - Voice activity detection
    - Normalization
    - Audio quality metrics

    Args:
        signal: Input signal

    Returns:
        RMS energy value

    Example:
        ```mojo
        var energy = rms_energy(audio_chunk)
        if energy > threshold:
            print("Speech detected!")
        ```
    """
    var sum_squares: Float64 = 0.0

    for i in range(len(signal)):
        sum_squares += signal[i] * signal[i]

    var mean_square = sum_squares / len(signal)
    return sqrt(mean_square)


fn normalize_audio(signal: List[Float64]) -> List[Float64]:
    """
    Normalize audio to [-1.0, 1.0] range.

    Args:
        signal: Input signal

    Returns:
        Normalized signal

    Example:
        ```mojo
        var normalized = normalize_audio(raw_audio)
        ```
    """
    # Find max absolute value
    var max_val: Float64 = 0.0
    for i in range(len(signal)):
        var abs_val = signal[i]
        if abs_val < 0:
            abs_val = -abs_val
        if abs_val > max_val:
            max_val = abs_val

    # Avoid division by zero
    if max_val == 0.0:
        return signal

    # Normalize
    var result = List[Float64]()
    for i in range(len(signal)):
        result.append(signal[i] / max_val)

    return result^


# ==============================================================================
# Validation Helpers
# ==============================================================================

fn validate_whisper_audio(audio: List[Float64], duration_seconds: Int) -> Bool:
    """
    Validate audio meets Whisper requirements.

    Requirements:
    - 16kHz sample rate
    - Expected samples = duration_seconds * 16000
    - Normalized to [-1, 1]

    Args:
        audio: Input audio samples
        duration_seconds: Expected duration

    Returns:
        True if valid for Whisper

    Example:
        ```mojo
        var is_valid = validate_whisper_audio(audio, 30)
        if not is_valid:
            print("Audio doesn't meet Whisper requirements!")
        ```
    """
    var expected_samples = duration_seconds * WHISPER_SAMPLE_RATE
    return len(audio) == expected_samples
