import os
import numpy as np
from scipy.io.wavfile import read as read_wav
from scipy.signal import butter, sosfiltfilt, welch
from matplotlib.figure import Figure
try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
except Exception:
    PCA = None
    TSNE = None


def _butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    return butter(order, [lowcut / nyq, highcut / nyq], btype="band", output="sos")


@register_analyzer("Denoise & Feature Probe")
def analyze_denoise_feature(all_data, selected_ids, chunk_size, rate):
    """
    Denoise recordings and visualize spectra to check if grasp-related features are present.
    Runs PCA/t-SNE on PSD features before and after bandpass to see clustering shifts.
    """
    if PCA is None or TSNE is None:
        raise ImportError("scikit-learn is required for PCA/t-SNE.")

    if len(selected_ids) < 2:
        raise ValueError("Select at least two recordings for this analyzer.")

    raw_psds = []
    clean_psds = []
    labels = []
    sessions = []
    freqs_ref = None

    # Process each selected file
    for uid in selected_ids:
        entry = all_data[uid]
        wav_path = entry.get("wav_path")
        if not wav_path or not os.path.exists(wav_path):
            raise FileNotFoundError(f"Missing audio file: {wav_path}")

        fs, raw = read_wav(wav_path)
        raw = raw.astype(np.float32)
        peak = np.max(np.abs(raw)) or 1.0
        raw_norm = raw / peak

        bp_sos = _butter_bandpass(100.0, 6000.0, fs, order=4)
        clean = sosfiltfilt(bp_sos, raw_norm)

        freqs, psd_raw = welch(raw_norm, fs=fs, nperseg=2048, noverlap=1024)
        _, psd_clean = welch(clean, fs=fs, nperseg=2048, noverlap=1024)
        if freqs_ref is None:
            freqs_ref = freqs
        elif freqs.shape != freqs_ref.shape or not np.allclose(freqs, freqs_ref):
            raise ValueError("Frequency grid mismatch across recordings; ensure consistent sample rates.")

        raw_psds.append(psd_raw)
        clean_psds.append(psd_clean)
        labels.append(entry.get("label", "Unlabeled"))
        sessions.append(entry.get("session", ""))

    band_mask = (freqs_ref >= 20) & (freqs_ref <= rate / 2)

    # Make matrices
    X_raw = np.log10(np.vstack(raw_psds) + 1e-12)[:, band_mask]
    X_clean = np.log10(np.vstack(clean_psds) + 1e-12)[:, band_mask]

    def _embed(matrix):
        # Standardize
        Xc = matrix - matrix.mean(axis=0)
        std = matrix.std(axis=0)
        std[std == 0] = 1.0
        Xn = Xc / std
        pca = PCA(n_components=2)
        pca_coords = pca.fit_transform(Xn)
        perplexity = max(2, min(30, len(matrix) - 1))
        tsne = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=perplexity, random_state=42)
        tsne_coords = tsne.fit_transform(Xn)
        return pca_coords, tsne_coords, perplexity

    pca_raw, tsne_raw, perplex_raw = _embed(X_raw)
    pca_clean, tsne_clean, perplex_clean = _embed(X_clean)

    # PCA plots
    fig_pca_raw = Figure(figsize=(6, 5))
    ax_pr = fig_pca_raw.add_subplot(111)
    for lbl in sorted(set(labels)):
        idx = [i for i, l in enumerate(labels) if l == lbl]
        ax_pr.scatter(pca_raw[idx, 0], pca_raw[idx, 1], label=lbl, alpha=0.8)
    ax_pr.set_title("PCA (Raw PSD)")
    ax_pr.set_xlabel("PC1")
    ax_pr.set_ylabel("PC2")
    ax_pr.legend()
    ax_pr.grid(True, alpha=0.3)

    fig_pca_clean = Figure(figsize=(6, 5))
    ax_pc = fig_pca_clean.add_subplot(111)
    for lbl in sorted(set(labels)):
        idx = [i for i, l in enumerate(labels) if l == lbl]
        ax_pc.scatter(pca_clean[idx, 0], pca_clean[idx, 1], label=lbl, alpha=0.8)
    ax_pc.set_title("PCA (Denoised PSD)")
    ax_pc.set_xlabel("PC1")
    ax_pc.set_ylabel("PC2")
    ax_pc.legend()
    ax_pc.grid(True, alpha=0.3)

    # t-SNE plots
    fig_tsne_raw = Figure(figsize=(6, 5))
    ax_tr = fig_tsne_raw.add_subplot(111)
    for lbl in sorted(set(labels)):
        idx = [i for i, l in enumerate(labels) if l == lbl]
        ax_tr.scatter(tsne_raw[idx, 0], tsne_raw[idx, 1], label=lbl, alpha=0.8)
    ax_tr.set_title(f"t-SNE (Raw PSD, perplexity={perplex_raw})")
    ax_tr.set_xlabel("Dim 1")
    ax_tr.set_ylabel("Dim 2")
    ax_tr.legend()
    ax_tr.grid(True, alpha=0.3)

    fig_tsne_clean = Figure(figsize=(6, 5))
    ax_tc = fig_tsne_clean.add_subplot(111)
    for lbl in sorted(set(labels)):
        idx = [i for i, l in enumerate(labels) if l == lbl]
        ax_tc.scatter(tsne_clean[idx, 0], tsne_clean[idx, 1], label=lbl, alpha=0.8)
    ax_tc.set_title(f"t-SNE (Denoised PSD, perplexity={perplex_clean})")
    ax_tc.set_xlabel("Dim 1")
    ax_tc.set_ylabel("Dim 2")
    ax_tc.legend()
    ax_tc.grid(True, alpha=0.3)

    # Average PSD comparison
    fig_psd = Figure(figsize=(8, 5))
    ax_psd = fig_psd.add_subplot(111)
    ax_psd.semilogx(freqs_ref[band_mask], 10 * np.log10(np.mean(raw_psds, axis=0)[band_mask] + 1e-12), label="Avg Raw", alpha=0.7)
    ax_psd.semilogx(freqs_ref[band_mask], 10 * np.log10(np.mean(clean_psds, axis=0)[band_mask] + 1e-12), label="Avg Denoised", alpha=0.8)
    ax_psd.axvspan(100, 2000, color="green", alpha=0.1, label="Grasp-focus band (100-2000 Hz)")
    ax_psd.set_title("Power Spectral Density (Averages)")
    ax_psd.set_xlabel("Frequency (Hz)")
    ax_psd.set_ylabel("Power (dB/Hz)")
    ax_psd.grid(True, which="both", ls="--", alpha=0.4)
    ax_psd.legend()
    fig_psd.tight_layout()

    return {
        "PCA (Raw)": fig_pca_raw,
        "PCA (Denoised)": fig_pca_clean,
        "t-SNE (Raw)": fig_tsne_raw,
        "t-SNE (Denoised)": fig_tsne_clean,
        "PSD Averages": fig_psd,
    }
