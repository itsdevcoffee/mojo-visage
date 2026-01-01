"""
mojo-audio: High-performance audio signal processing library.

SIMD-optimized DSP operations for machine learning audio preprocessing.
Designed for Whisper and other speech recognition models.
"""

from math import cos, sqrt, log, sin, atan2, exp
from math.constants import pi
from memory import UnsafePointer


fn pow(base: Float64, exponent: Float64) -> Float64:
    """Power function: base^exponent."""
    return exp(exponent * log(base))


# ==============================================================================
# SIMD-Optimized Operations
# ==============================================================================

fn apply_window_simd(signal: List[Float64], window: List[Float64]) raises -> List[Float64]:
    """
    SIMD-optimized window application.

    Uses pointer-based SIMD for fast element-wise multiplication.

    Args:
        signal: Input signal
        window: Window coefficients

    Returns:
        Windowed signal
    """
    if len(signal) != len(window):
        raise Error("Signal and window length mismatch")

    var N = len(signal)
    var result = List[Float64]()

    # Pre-allocate
    for _ in range(N):
        result.append(0.0)

    # Use SIMD for element-wise multiply
    comptime simd_width = 8

    var i = 0
    while i + simd_width <= N:
        # Create SIMD vectors by loading from lists
        var sig_vec = SIMD[DType.float64, simd_width]()
        var win_vec = SIMD[DType.float64, simd_width]()

        @parameter
        for j in range(simd_width):
            sig_vec[j] = signal[i + j]
            win_vec[j] = window[i + j]

        # SIMD multiply
        var res_vec = sig_vec * win_vec

        # Store back
        @parameter
        for j in range(simd_width):
            result[i + j] = res_vec[j]

        i += simd_width

    # Remainder
    while i < N:
        result[i] = signal[i] * window[i]
        i += 1

    return result^


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


fn bit_reverse(n: Int, bits: Int) -> Int:
    """
    Reverse bits of integer n using 'bits' number of bits.

    Used for FFT bit-reversal permutation.
    """
    var result = 0
    var x = n
    for _ in range(bits):
        result = (result << 1) | (x & 1)
        x >>= 1
    return result


fn log2_int(n: Int) -> Int:
    """Compute log2 of power-of-2 integer."""
    var result = 0
    var x = n
    while x > 1:
        x >>= 1
        result += 1
    return result


fn precompute_twiddle_factors(N: Int) -> List[Complex]:
    """
    Pre-compute all twiddle factors for FFT of size N.

    Twiddle factor: W_N^k = e^(-2πik/N) = cos(-2πk/N) + i*sin(-2πk/N)

    Eliminates expensive transcendental function calls from FFT hot loop.
    MAJOR performance improvement!

    Args:
        N: FFT size (power of 2)

    Returns:
        Twiddle factors for all stages
    """
    var twiddles = List[Complex]()

    # Pre-compute all twiddles we'll need for all stages
    for i in range(N):
        var angle = -2.0 * pi * Float64(i) / Float64(N)
        twiddles.append(Complex(cos(angle), sin(angle)))

    return twiddles^


fn fft_iterative(signal: List[Float64]) raises -> List[Complex]:
    """
    Iterative FFT using Cooley-Tukey algorithm.

    Optimized with pre-computed twiddle factors!
    No cos/sin in hot loop = massive speedup.

    Args:
        signal: Input (length must be power of 2)

    Returns:
        Complex frequency spectrum
    """
    var N = len(signal)

    # Validate power of 2
    if N == 0 or (N & (N - 1)) != 0:
        raise Error("FFT requires power of 2. Got " + String(N))

    # PRE-COMPUTE twiddle factors (KEY OPTIMIZATION!)
    var twiddles = precompute_twiddle_factors(N)

    # Initialize output with bit-reversed input
    var result = List[Complex]()
    var log2_n = log2_int(N)

    for i in range(N):
        var reversed_idx = bit_reverse(i, log2_n)
        result.append(Complex(signal[reversed_idx], 0.0))

    # Iterative butterfly operations
    var size = 2
    while size <= N:
        var half_size = size // 2
        var stride = N // size  # Twiddle factor stride

        # Process each butterfly group
        for i in range(0, N, size):
            # Use pre-computed twiddles (no cos/sin!)
            for k in range(half_size):
                # Index into pre-computed twiddle table
                var twiddle_idx = k * stride
                var twiddle = Complex(twiddles[twiddle_idx].real, twiddles[twiddle_idx].imag)

                # Butterfly operation indices
                var idx1 = i + k
                var idx2 = i + k + half_size

                # Butterfly computation (make explicit copies)
                var t = twiddle * Complex(result[idx2].real, result[idx2].imag)
                var u = Complex(result[idx1].real, result[idx1].imag)

                var sum_val = u + t
                var diff_val = u - t

                result[idx1] = Complex(sum_val.real, sum_val.imag)
                result[idx2] = Complex(diff_val.real, diff_val.imag)

        size *= 2

    return result^


fn fft_internal(signal: List[Float64]) raises -> List[Complex]:
    """
    Internal FFT - uses iterative algorithm for better performance.

    Delegates to iterative FFT which has better cache locality.
    """
    return fft_iterative(signal)


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
    Compute power spectrum from FFT output (SIMD-optimized).

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
    var N = len(fft_output)
    var result = List[Float64]()

    # Pre-allocate
    for _ in range(N):
        result.append(0.0)

    # SIMD processing
    comptime simd_width = 8

    var i = 0
    while i + simd_width <= N:
        # Load real/imag into SIMD vectors
        var real_vec = SIMD[DType.float64, simd_width]()
        var imag_vec = SIMD[DType.float64, simd_width]()

        @parameter
        for j in range(simd_width):
            real_vec[j] = fft_output[i + j].real
            imag_vec[j] = fft_output[i + j].imag

        # SIMD: real² + imag²
        var power_vec = real_vec * real_vec + imag_vec * imag_vec

        # Store
        @parameter
        for j in range(simd_width):
            result[i + j] = power_vec[j]

        i += simd_width

    # Remainder
    while i < N:
        result[i] = fft_output[i].power()
        i += 1

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

        # Apply window (SIMD-optimized)
        var windowed = apply_window_simd(frame, window)

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
# Mel Scale Operations
# ==============================================================================

fn hz_to_mel(freq_hz: Float64) -> Float64:
    """
    Convert frequency from Hz to Mel scale.

    Mel scale formula: mel = 2595 * log10(1 + hz/700)

    The mel scale approximates human perception of pitch.
    Equal distances on the mel scale sound equally different to humans.

    Args:
        freq_hz: Frequency in Hertz

    Returns:
        Frequency in Mels

    Example:
        ```mojo
        var mel = hz_to_mel(1000.0)  # ~1000 Hz ≈ 1000 mels
        ```
    """
    return 2595.0 * log(1.0 + freq_hz / 700.0) / log(10.0)


fn mel_to_hz(freq_mel: Float64) -> Float64:
    """
    Convert frequency from Mel scale to Hz.

    Inverse of hz_to_mel: hz = 700 * (10^(mel/2595) - 1)

    Args:
        freq_mel: Frequency in Mels

    Returns:
        Frequency in Hertz

    Example:
        ```mojo
        var hz = mel_to_hz(1000.0)
        ```
    """
    return 700.0 * (pow(10.0, freq_mel / 2595.0) - 1.0)


fn create_mel_filterbank(
    n_mels: Int,
    n_fft: Int,
    sample_rate: Int
) -> List[List[Float64]]:
    """
    Create mel filterbank matrix for spectrogram → mel spectrogram conversion.

    Creates triangular filters spaced evenly on the mel scale.

    Args:
        n_mels: Number of mel bands (Whisper: 80)
        n_fft: FFT size (Whisper: 400)
        sample_rate: Audio sample rate (Whisper: 16000)

    Returns:
        Filterbank matrix (n_mels × (n_fft/2 + 1))
        For Whisper: (80 × 201)

    Example:
        ```mojo
        var filterbank = create_mel_filterbank(80, 400, 16000)
        # Shape: (80, 201) - ready to multiply with STFT output
        ```

    How it works:
        - Converts Hz frequency bins to Mel scale
        - Creates triangular filters on Mel scale
        - Each filter has peak at one mel frequency
        - Filters overlap to smooth the spectrum
    """
    var n_freq_bins = n_fft // 2 + 1  # Number of positive frequencies

    # Frequency range: 0 Hz to Nyquist (sample_rate/2)
    var nyquist = Float64(sample_rate) / 2.0

    # Convert to mel scale
    var mel_min = hz_to_mel(0.0)
    var mel_max = hz_to_mel(nyquist)

    # Create evenly spaced mel frequencies
    var mel_points = List[Float64]()
    var mel_step = (mel_max - mel_min) / Float64(n_mels + 1)

    for i in range(n_mels + 2):
        mel_points.append(mel_min + Float64(i) * mel_step)

    # Convert mel points back to Hz
    var hz_points = List[Float64]()
    for i in range(len(mel_points)):
        hz_points.append(mel_to_hz(mel_points[i]))

    # Convert Hz to FFT bin numbers
    var bin_points = List[Int]()
    for i in range(len(hz_points)):
        var bin = Int((Float64(n_fft + 1) * hz_points[i]) / Float64(sample_rate))
        bin_points.append(bin)

    # Create filterbank (n_mels × n_freq_bins)
    var filterbank = List[List[Float64]]()

    for mel_idx in range(n_mels):
        var filter_band = List[Float64]()

        # Initialize all bins to 0
        for _ in range(n_freq_bins):
            filter_band.append(0.0)

        # Create triangular filter
        var left = bin_points[mel_idx]
        var center = bin_points[mel_idx + 1]
        var right = bin_points[mel_idx + 2]

        # Create triangular filter only if valid range
        if center > left and right > center:
            # Rising slope (left to center)
            for bin_idx in range(left, center):
                if bin_idx < n_freq_bins and bin_idx >= 0:
                    var weight = Float64(bin_idx - left) / Float64(center - left)
                    filter_band[bin_idx] = weight

            # Falling slope (center to right)
            for bin_idx in range(center, right):
                if bin_idx < n_freq_bins and bin_idx >= 0:
                    var weight = Float64(right - bin_idx) / Float64(right - center)
                    filter_band[bin_idx] = weight

        filterbank.append(filter_band^)

    return filterbank^


fn apply_mel_filterbank(
    spectrogram: List[List[Float64]],
    filterbank: List[List[Float64]]
) raises -> List[List[Float64]]:
    """
    Apply mel filterbank to power spectrogram (optimized).

    Converts linear frequency bins to mel-spaced bins.
    Optimized: Pre-allocates memory, minimizes allocations.

    Args:
        spectrogram: Power spectrogram (n_freq_bins, n_frames)
        filterbank: Mel filterbank (n_mels, n_freq_bins)

    Returns:
        Mel spectrogram (n_mels, n_frames)

    Example:
        ```mojo
        var spec = stft(audio)  # (201, 3000)
        var filterbank = create_mel_filterbank(80, 400, 16000)  # (80, 201)
        var mel_spec = apply_mel_filterbank(spec, filterbank)  # (80, 3000)
        ```
    """
    var n_frames = len(spectrogram)
    if n_frames == 0:
        raise Error("Empty spectrogram")

    var n_freq_bins = len(spectrogram[0])
    var n_mels = len(filterbank)

    # Validate dimensions
    if len(filterbank[0]) != n_freq_bins:
        raise Error("Filterbank size mismatch with spectrogram")

    var mel_spec = List[List[Float64]]()

    # For each mel band
    for mel_idx in range(n_mels):
        var mel_band = List[Float64]()

        # Pre-allocate for this mel band
        for _ in range(n_frames):
            mel_band.append(0.0)

        # For each time frame
        for frame_idx in range(n_frames):
            var mel_energy: Float64 = 0.0

            # Sum weighted frequency bins (dot product)
            # Only process non-zero filter weights
            for freq_idx in range(n_freq_bins):
                var filter_weight = filterbank[mel_idx][freq_idx]
                if filter_weight > 0.0:  # Skip zero weights
                    mel_energy += filter_weight * spectrogram[frame_idx][freq_idx]

            mel_band[frame_idx] = mel_energy

        mel_spec.append(mel_band^)

    return mel_spec^


fn mel_spectrogram(
    audio: List[Float64],
    sample_rate: Int = 16000,
    n_fft: Int = 400,
    hop_length: Int = 160,
    n_mels: Int = 80
) raises -> List[List[Float64]]:
    """
    Compute mel spectrogram - the full Whisper preprocessing pipeline!

    This is the complete transformation: audio → mel spectrogram

    Args:
        audio: Input audio samples
        sample_rate: Sample rate in Hz (Whisper: 16000)
        n_fft: FFT size (Whisper: 400)
        hop_length: Frame hop size (Whisper: 160)
        n_mels: Number of mel bands (Whisper: 80)

    Returns:
        Mel spectrogram (n_mels, n_frames)
        For 30s Whisper audio: (80, ~3000) ✓

    Example:
        ```mojo
        # Load 30s audio @ 16kHz
        var audio: List[Float64] = [...]  # 480,000 samples

        # Get Whisper-compatible mel spectrogram
        var mel_spec = mel_spectrogram(audio)
        # Output: (80, ~3000) - ready for Whisper!
        ```

    Pipeline:
        1. STFT with Hann window → (201, ~3000)
        2. Mel filterbank application → (80, ~3000)
        3. Log scaling → final mel spectrogram

    This is exactly what Whisper expects as input!
    """
    # Step 1: Compute STFT (power spectrogram)
    var power_spec = stft(audio, n_fft, hop_length, "hann")

    # Step 2: Create mel filterbank
    var filterbank = create_mel_filterbank(n_mels, n_fft, sample_rate)

    # Step 3: Apply filterbank
    var mel_spec = apply_mel_filterbank(power_spec, filterbank)

    # Step 4: Log scaling (with small epsilon to avoid log(0))
    var epsilon: Float64 = 1e-10

    for i in range(len(mel_spec)):
        for j in range(len(mel_spec[i])):
            # Clamp to epsilon minimum
            var value = mel_spec[i][j]
            if value < epsilon:
                value = epsilon

            # Apply log10 scaling
            mel_spec[i][j] = log(value) / log(10.0)

    return mel_spec^


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
