
import numpy as np
from scipy.io.wavfile import read as read_wav
from matplotlib.figure import Figure
import os

# This module is dynamically loaded by app.py
# It expects 'register_analyzer' to be in its global scope.
print(f"Loading analyzers from {os.path.basename(__file__)}...")

@register_analyzer("Comparison (Overlaid FFT)")
def analyze_comparison(all_data, selected_ids, chunk_size, rate):
    fig = Figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    
    freq_bins = np.fft.rfftfreq(chunk_size, 1 / rate)
    
    for uid in selected_ids:
        data = all_data[uid]
        ax.plot(freq_bins, data['spectral_data'], label=data['base_filename'])

    ax.set_title("Comparison of Averaged Spectrums")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude [dB]")
    ax.set_xlim(20, rate / 2)
    ax.set_ylim(-80, 20)
    ax.set_xscale('log')
    ax.grid(True, which="both", ls="--")
    ax.legend()
    fig.tight_layout()
    
    return {"Overlaid FFT": fig}

@register_analyzer("Detailed 5-Chirp Analysis")
def analyze_chirps_detailed(all_data, selected_ids, chunk_size, rate):
    if len(selected_ids) != 1:
        raise ValueError("This analyzer requires exactly one recording to be selected.")
    
    uid = selected_ids[0]
    data = all_data[uid]
    
    if not os.path.exists(data['wav_path']):
        raise FileNotFoundError(f"Missing audio file: {data['wav_path']}")
        
    sr, raw_audio = read_wav(data['wav_path'])

    # -- 1. Waveform & Segmentation --
    fig_wave = Figure(figsize=(8, 4))
    ax_wave = fig_wave.add_subplot(111)
    ax_wave.plot(np.arange(len(raw_audio)) / sr, raw_audio, color='lightblue', zorder=1)
    ax_wave.set_title(f"Waveform: {data['base_filename']}")
    ax_wave.set_xlabel("Time (s)")
    ax_wave.set_ylabel("Amplitude")
    ax_wave.grid(True, alpha=0.3)

    window_size, hop_size = 2048, 512
    rms = np.array([np.sqrt(np.mean(raw_audio[i:i+window_size]**2)) for i in range(0, len(raw_audio)-window_size, hop_size)])
    thresh = np.max(rms) * 0.5
    peaks = [i for i in range(1, len(rms)-1) if rms[i] > thresh and rms[i] > rms[i-1] and rms[i] > rms[i+1]]
    min_dist = (sr * 0.9) / hop_size
    debounced = [peaks[0]] if peaks else []
    for p in peaks[1:]:
        if p - debounced[-1] > min_dist: debounced.append(p)

    chirp_segments = []
    if len(debounced) == 5:
        for i, p in enumerate(debounced):
            start = p * hop_size
            end = start + int(sr * 0.8)
            chirp_segments.append(raw_audio[start:end])
            ax_wave.axvspan(start/sr, end/sr, color=f'C{i}', alpha=0.3, zorder=2, label=f"Chirp {i+1}")
        ax_wave.legend()
    
    fig_wave.tight_layout()

    # -- 2. Overlaid FFT of segments --
    fig_fft = Figure(figsize=(8, 4))
    ax_fft = fig_fft.add_subplot(111)
    if chirp_segments:
        for i, seg in enumerate(chirp_segments):
            N = len(seg)
            freqs = np.fft.rfftfreq(N, 1/sr)
            mag = np.abs(np.fft.rfft(seg * np.hanning(N)))
            ax_fft.plot(freqs, 20*np.log10(mag + 1e-9), label=f'Chirp {i+1}', color=f'C{i}', alpha=0.8)
    else:
        ax_fft.text(0.5, 0.5, "Could not automatically segment 5 chirps", ha='center')

    ax_fft.set_title("Spectral Analysis of Individual Chirps")
    ax_fft.set_xscale('log')
    ax_fft.set_xlim(20, sr/2)
    ax_fft.set_xlabel("Frequency (Hz)")
    ax_fft.set_ylabel("Amplitude [dB]")
    ax_fft.grid(True, which="both", ls="--")
    if chirp_segments: ax_fft.legend()
    fig_fft.tight_layout()

    # -- 3. Spectrogram --
    fig_spec = Figure(figsize=(8, 4))
    ax_spec = fig_spec.add_subplot(111)
    ax_spec.specgram(raw_audio, Fs=sr, NFFT=1024, cmap='viridis')
    ax_spec.set_title("Full Spectrogram")
    ax_spec.set_xlabel("Time (s)")
    ax_spec.set_ylabel("Frequency (Hz)")
    ax_spec.axis('tight')
    fig_spec.tight_layout()

    return {
        "Waveform Segmentation": fig_wave,
        "Chirp FFTs": fig_fft,
        "Full Spectrogram": fig_spec
    }
