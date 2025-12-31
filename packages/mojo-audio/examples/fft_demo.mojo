"""
FFT and STFT Demo

Demonstrates Fast Fourier Transform and Short-Time Fourier Transform.
Shows how to create spectrograms for audio ML preprocessing.
"""

from audio import fft, power_spectrum, stft, hann_window


fn main() raises:
    print("\n" + "="*70)
    print("FFT & STFT Demo - mojo-audio")
    print("="*70 + "\n")

    # Example 1: Simple FFT
    print("="*70)
    print("1. Fast Fourier Transform (FFT)")
    print("="*70 + "\n")

    var signal: List[Float64] = [1.0, 0.0, -1.0, 0.0]
    print("Signal:", signal)

    var spectrum = fft(signal)
    print("FFT output length:", len(spectrum))
    print()

    print("Frequency bins (complex):")
    for i in range(len(spectrum)):
        var mag = spectrum[i].magnitude()
        print("  Bin", i, ": magnitude =", mag)

    print()

    # Example 2: Power Spectrum
    print("="*70)
    print("2. Power Spectrum")
    print("="*70 + "\n")

    var power = power_spectrum(spectrum)
    print("Power spectrum (real-valued energy):")
    for i in range(len(power)):
        print("  Bin", i, ": power =", power[i])

    print()

    # Example 3: STFT - Create Spectrogram
    print("="*70)
    print("3. Short-Time Fourier Transform (STFT)")
    print("="*70 + "\n")

    # Create 1 second of test audio
    var audio_1s = List[Float64]()
    for i in range(16000):  # 1s @ 16kHz
        audio_1s.append(0.1)

    print("Input: 1 second of audio (16,000 samples)")
    print("Parameters:")
    print("  n_fft:      400 (Whisper standard)")
    print("  hop_length: 160 (Whisper standard)")
    print("  window:     Hann")
    print()

    var spec = stft(audio_1s, n_fft=400, hop_length=160)

    print("STFT Output:")
    print("  Frequency bins:", len(spec[0]))
    print("  Time frames:   ", len(spec))
    print("  Shape:         (", len(spec[0]), ",", len(spec), ")")
    print()

    print("Interpretation:")
    print("  - Each column = one time frame (10ms with hop=160)")
    print("  - Each row = one frequency bin")
    print("  - Values = power at that frequency and time")
    print()

    # Example 4: Whisper-Compatible Spectrogram
    print("="*70)
    print("4. Whisper-Compatible STFT (30 seconds)")
    print("="*70 + "\n")

    # Create 30 seconds of audio
    var audio_30s = List[Float64]()
    for _ in range(30 * 16000):  # 480,000 samples
        audio_30s.append(0.05)

    print("Input: 30 seconds of audio (480,000 samples)")
    print()

    var spec_30s = stft(audio_30s, n_fft=400, hop_length=160)

    print("Output spectrogram:")
    print("  Frequency bins:", len(spec_30s[0]), "(expected: 201)")
    print("  Time frames:   ", len(spec_30s), "(expected: ~3000)")
    print()

    print("Whisper requirements:")
    print("  ✓ Sample rate: 16kHz")
    print("  ✓ n_fft: 400")
    print("  ✓ hop_length: 160")
    print("  ✓ Shape: (201, ~3000)")
    print()

    print("="*70)
    print("✓ STFT complete! Ready for mel filterbank.")
    print("  Next: Apply mel filterbank → log scale → Whisper input")
    print("="*70 + "\n")
