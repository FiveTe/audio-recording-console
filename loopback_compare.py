"""
Quick comparison of a ClosedPalm recording with/without loopback subtraction.
Uses the saved loopback reference (data/loopback_ref.npy) if present.
Outputs plots to data/reports/loopback_compare.png
"""
import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

APP_ROOT = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(APP_ROOT, "data")
REPORT_DIR = os.path.join(DATA_DIR, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)
LOOPBACK_FILE = os.path.join(DATA_DIR, "loopback_ref.npy")


def load_loopback():
    if os.path.exists(LOOPBACK_FILE):
        try:
            return np.load(LOOPBACK_FILE).astype(np.float32)
        except Exception as e:
            print(f"Failed to load loopback ref: {e}")
    return None


def align_and_subtract(signal, ref):
    """
    Aligns ref to signal via cross-correlation, estimates gain, and subtracts.
    Returns cleaned signal, shift (samples), and gain.
    """
    if ref is None or len(ref) == 0:
        return signal, 0, 1.0

    sig = signal.astype(np.float32)
    ref = ref.astype(np.float32)

    # DC remove
    sig = sig - np.mean(sig)
    ref = ref - np.mean(ref)

    # Limit ref length to not exceed signal
    if len(ref) > len(sig):
        ref = ref[: len(sig)]

    # Cross-correlate to find best alignment (valid mode)
    corr = np.correlate(sig, ref, mode="valid")
    shift = int(np.argmax(corr))

    # Align ref
    ref_aligned = np.zeros_like(sig)
    end = min(len(sig), shift + len(ref))
    ref_aligned[shift:end] = ref[: end - shift]

    # Estimate gain via least squares on overlap
    overlap = ref_aligned != 0
    if not np.any(overlap):
        gain = 0.0
    else:
        gain = np.dot(sig[overlap], ref_aligned[overlap]) / (np.dot(ref_aligned[overlap], ref_aligned[overlap]) + 1e-9)
    cleaned = sig - gain * ref_aligned
    return cleaned, shift, gain


def main():
    parser = argparse.ArgumentParser(description="Compare raw vs loopback-subtracted for a ClosedPalm file.")
    parser.add_argument("--file", default=None, help="Path to a WAV file to use. If omitted, picks a ClosedPalm HEC.")
    args = parser.parse_args()

    if args.file:
        wav_path = args.file
        if not os.path.exists(wav_path):
            print(f"Provided file not found: {wav_path}")
            return
    else:
        # Find a ClosedPalm file (prefers SessionNine)
        candidates = glob.glob(os.path.join(DATA_DIR, "recordings", "SessionNine", "**", "ClosedPalm*HEC*.wav"), recursive=True)
        if not candidates:
            candidates = glob.glob(os.path.join(DATA_DIR, "recordings", "**", "ClosedPalm*HEC*.wav"), recursive=True)
        if not candidates:
            print("No ClosedPalm HEC recordings found.")
            return
        wav_path = sorted(candidates)[0]
    sr, audio = wavfile.read(wav_path)
    audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio = audio / (np.max(np.abs(audio)) or 1.0)

    loopback = load_loopback()
    audio_clean, shift, gain = align_and_subtract(audio, loopback)

    # Plots
    t = np.arange(len(audio)) / sr
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    # Waveforms (first 0.3 s)
    max_t = min(0.3, t[-1])
    N = int(max_t * sr)
    axes[0, 0].plot(t[:N], audio[:N], label="Raw", alpha=0.8)
    axes[0, 0].set_title("Raw waveform (first 0.3 s)")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Amp")
    axes[0, 1].plot(t[:N], audio_clean[:N], label="Loopback subtracted", color="C1", alpha=0.8)
    axes[0, 1].set_title("Loopback-subtracted waveform (first 0.3 s)")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Amp")
    # Spectra
    freqs = np.fft.rfftfreq(len(audio), 1 / sr)
    mag_raw = 20 * np.log10(np.abs(np.fft.rfft(audio * np.hanning(len(audio)))) + 1e-9)
    mag_clean = 20 * np.log10(np.abs(np.fft.rfft(audio_clean * np.hanning(len(audio_clean)))) + 1e-9)
    axes[1, 0].plot(freqs, mag_raw, alpha=0.8)
    axes[1, 0].set_title("Raw spectrum")
    axes[1, 0].set_xlabel("Freq (Hz)")
    axes[1, 0].set_ylabel("dB")
    axes[1, 0].set_xlim(20, sr / 2)
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_ylim(-120, 40)
    axes[1, 1].plot(freqs, mag_clean, color="C1", alpha=0.8)
    axes[1, 1].set_title("Loopback-subtracted spectrum")
    axes[1, 1].set_xlabel("Freq (Hz)")
    axes[1, 1].set_ylabel("dB")
    axes[1, 1].set_xlim(20, sr / 2)
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_ylim(-120, 40)
    fig.suptitle(f"ClosedPalm comparison\n{os.path.basename(wav_path)} | Loopback loaded: {loopback is not None} | shift={shift} samples | gain={gain:.3f}")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(REPORT_DIR, "loopback_compare.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved comparison plot to {out_path}")


if __name__ == "__main__":
    main()
