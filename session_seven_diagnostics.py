"""
Diagnostics for SessionSeven:
- Plot raw waveforms per label (sampled)
- Plot spectrograms per label (sampled)
- PCA scatter on raw waveform segments
- PCA scatter on cepstral features
- PCA scatter on chirp cross-correlation features

Outputs are saved under data/reports/.
"""
import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, chirp, correlate
from sklearn.decomposition import PCA
import argparse

APP_ROOT = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(APP_ROOT, "data")
REPORT_DIR = os.path.join(DATA_DIR, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)


def load_session_wavs(session_name="SessionSeven"):
    session_dir = os.path.join(DATA_DIR, "recordings", session_name)
    json_files = glob.glob(os.path.join(session_dir, "**", "*_data.json"), recursive=True)
    data = []
    for jf in json_files:
        try:
            with open(jf, "r") as f:
                meta = json.load(f)
            wav_path = meta.get("wav_path")
            label = meta.get("label", "Unlabeled")
            if not wav_path or not os.path.exists(wav_path):
                continue
            sr, wav = read_wav_mono(wav_path)
            data.append({"label": label, "wav": wav, "sr": sr, "path": wav_path})
        except Exception as e:
            print(f"Skipping {jf}: {e}")
    return data


def read_wav_mono(path):
    from scipy.io import wavfile
    sr, audio = wavfile.read(path)
    audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio[:, 0]
    # normalize
    peak = np.max(np.abs(audio)) or 1.0
    audio = audio / peak
    return sr, audio


def sample_by_label(data, per_label=3, seed=42):
    rng = np.random.default_rng(seed)
    out = []
    labels = {}
    for item in data:
        labels.setdefault(item["label"], []).append(item)
    for lbl, arr in labels.items():
        if len(arr) > per_label:
            arr = rng.choice(arr, per_label, replace=False)
        out.extend(arr)
    return out, list(labels.keys())


def plot_waveforms(samples, out_path):
    fig, axes = plt.subplots(len(samples), 1, figsize=(10, 2 * len(samples)), sharex=False)
    if len(samples) == 1:
        axes = [axes]
    for ax, s in zip(axes, samples):
        t = np.arange(len(s["wav"])) / s["sr"]
        ax.plot(t, s["wav"], alpha=0.8)
        ax.set_title(f"{s['label']} | {os.path.basename(s['path'])}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amp")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_spectrograms(samples, out_path):
    fig, axes = plt.subplots(len(samples), 1, figsize=(10, 3 * len(samples)), sharex=False)
    if len(samples) == 1:
        axes = [axes]
    for ax, s in zip(axes, samples):
        f, t, Sxx = spectrogram(s["wav"], fs=s["sr"], nperseg=1024, noverlap=512)
        Sxx_db = 10 * np.log10(Sxx + 1e-12)
        ax.pcolormesh(t, f, Sxx_db, shading="gouraud", cmap="magma")
        ax.set_title(f"Spectrogram: {s['label']} | {os.path.basename(s['path'])}")
        ax.set_ylabel("Freq (Hz)")
        ax.set_xlabel("Time (s)")
        ax.set_ylim([50, s["sr"] / 2])
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def center_crop(wav, sr, seconds=1.5):
    target = int(seconds * sr)
    if len(wav) <= target:
        pad = target - len(wav)
        left = pad // 2
        right = pad - left
        return np.pad(wav, (left, right))
    start = (len(wav) - target) // 2
    return wav[start:start + target]


def pca_scatter(X, labels, out_path, title):
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(6, 5))
    for lbl in sorted(set(labels)):
        idx = [i for i, l in enumerate(labels) if l == lbl]
        ax.scatter(coords[idx, 0], coords[idx, 1], label=lbl, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def real_cepstrum(wav, n_coeff=30):
    spectrum = np.fft.rfft(wav)
    log_mag = np.log(np.abs(spectrum) + 1e-12)
    cep = np.fft.irfft(log_mag)
    return cep[:n_coeff]


def chirp_xcorr_features(wav, sr, ref_duration=0.8, f0=100, f1=6000, topk=3):
    t = np.linspace(0, ref_duration, int(sr * ref_duration), endpoint=False)
    ref = chirp(t, f0=f0, f1=f1, t1=ref_duration, method="linear")
    corr = correlate(wav, ref, mode="valid")
    corr = corr / (np.linalg.norm(ref) * (np.linalg.norm(wav) + 1e-9))
    peaks_idx = np.argpartition(-corr, topk)[:topk]
    peaks_idx = peaks_idx[np.argsort(-corr[peaks_idx])]
    feats = []
    for i in range(topk):
        if i < len(peaks_idx):
            feats.append(corr[peaks_idx[i]])
            feats.append(peaks_idx[i] / sr)
        else:
            feats.extend([0.0, 0.0])
    feats.append(np.max(corr))
    return np.array(feats)


def prepare_features(data, crop_seconds=1.5):
    # Raw crop
    X_raw = []
    labels = []
    srs = []
    for d in data:
        wav_c = center_crop(d["wav"], d["sr"], seconds=crop_seconds)
        X_raw.append(wav_c)
        labels.append(d["label"])
        srs.append(d["sr"])
    min_len = min(len(x) for x in X_raw)
    X_raw = np.stack([x[:min_len] for x in X_raw])

    # Cepstrum
    X_cep = np.vstack([real_cepstrum(x, n_coeff=30) for x in X_raw])

    # Chirp xcorr
    X_xcorr = []
    for wav, sr in zip(X_raw, srs):
        feats = chirp_xcorr_features(wav, sr, ref_duration=0.8, f0=100, f1=6000, topk=3)
        X_xcorr.append(feats)
    X_xcorr = np.vstack(X_xcorr)

    return X_raw, X_cep, X_xcorr, labels


def main():
    parser = argparse.ArgumentParser(description="Diagnostics for a session (waveforms, spectrograms, PCA).")
    parser.add_argument("--session", default="SessionSeven", help="Session folder name (default: SessionSeven)")
    args = parser.parse_args()

    data = load_session_wavs(args.session)
    if not data:
        print(f"No data found for {args.session}.")
        return

    # Sample 3 per label for visualization
    vis_samples, labels_list = sample_by_label(data, per_label=3)
    plot_waveforms(vis_samples, os.path.join(REPORT_DIR, f"{args.session}_waveforms.png"))
    plot_spectrograms(vis_samples, os.path.join(REPORT_DIR, f"{args.session}_spectrograms.png"))

    # Feature prep and PCA plots
    X_raw, X_cep, X_xcorr, labels = prepare_features(data, crop_seconds=1.5)
    pca_scatter(X_raw, labels, os.path.join(REPORT_DIR, f"{args.session}_pca_raw.png"), "PCA on Raw Waveform Crop")
    pca_scatter(X_cep, labels, os.path.join(REPORT_DIR, f"{args.session}_pca_cepstrum.png"), "PCA on Cepstrum")
    pca_scatter(X_xcorr, labels, os.path.join(REPORT_DIR, f"{args.session}_pca_xcorr.png"), "PCA on Chirp XCorr Feats")

    print(f"Saved diagnostics to data/reports/ for {args.session}")


if __name__ == "__main__":
    main()
