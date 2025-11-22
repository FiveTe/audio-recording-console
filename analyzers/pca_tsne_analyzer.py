import numpy as np
from matplotlib.figure import Figure

# Optional: these imports are only used by this analyzer
try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
except Exception as e:  # pragma: no cover - dependency guard
    PCA = None
    TSNE = None
try:
    import librosa
    LIBROSA_AVAILABLE = True
except Exception:
    LIBROSA_AVAILABLE = False
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False


@register_analyzer("PCA & t-SNE (Sessions)")
def analyze_pca_tsne(all_data, selected_ids, chunk_size, rate):
    """
    Runs PCA and t-SNE on averaged spectra of selected recordings (the stored, averaged FFT magnitude, not raw waveform).
    If librosa is available, also runs PCA/t-SNE on log-mel features.
    If TensorFlow is available, also runs a tiny 2D autoencoder to provide a nonlinear embedding.
    """
    if PCA is None or TSNE is None:
        raise ImportError("scikit-learn is required for PCA/t-SNE analyzer.")

    if len(selected_ids) < 2:
        raise ValueError("Select at least two recordings for PCA/t-SNE.")

    features = []
    labels = []
    mel_features = [] if LIBROSA_AVAILABLE else None

    for uid in selected_ids:
        entry = all_data[uid]
        spec = np.asarray(entry.get("spectral_data"))
        if spec.ndim != 1:
            raise ValueError(f"Unexpected spectrum shape for {entry.get('base_filename')}")
        if features and spec.shape != features[0].shape:
            raise ValueError("Spectrum length mismatch across recordings; ensure consistent recording settings.")
        features.append(spec)
        label = entry.get("label", "Unlabeled")
        labels.append(label)
        if LIBROSA_AVAILABLE:
            wav_path = entry.get("wav_path")
            if not wav_path:
                raise ValueError(f"Missing wav_path for {entry.get('base_filename')}")
            audio, sr = librosa.load(wav_path, sr=None, mono=True)
            mel = librosa.feature.melspectrogram(
                y=audio, sr=sr, n_fft=2048, hop_length=512, n_mels=64, fmin=50, fmax=sr / 2
            )
            mel_db = librosa.power_to_db(mel + 1e-9)
            mel_features.append(mel_db.mean(axis=1))

    X = np.vstack(features)

    # Normalize per feature to balance scale; keeps relative noise structure intact.
    X_centered = X - X.mean(axis=0)
    std = X_centered.std(axis=0)
    std[std == 0] = 1.0
    X_norm = X_centered / std

    fig_pca = Figure(figsize=(6, 5))
    ax_pca = fig_pca.add_subplot(111)

    pca = PCA(n_components=min(2, X_norm.shape[1]))
    coords_pca = pca.fit_transform(X_norm)
    for lbl in sorted(set(labels)):
        idx = [i for i, l in enumerate(labels) if l == lbl]
        ax_pca.scatter(coords_pca[idx, 0], coords_pca[idx, 1], label=lbl, alpha=0.8)
    ax_pca.set_title("PCA of Averaged Spectra")
    ax_pca.set_xlabel("PC1")
    ax_pca.set_ylabel("PC2")
    ax_pca.legend()
    ax_pca.grid(True, alpha=0.3)

    fig_tsne = Figure(figsize=(6, 5))
    ax_tsne = fig_tsne.add_subplot(111)

    perplexity = max(2, min(30, len(selected_ids) - 1))
    tsne = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=perplexity, random_state=42)
    coords_tsne = tsne.fit_transform(X_norm)
    for lbl in sorted(set(labels)):
        idx = [i for i, l in enumerate(labels) if l == lbl]
        ax_tsne.scatter(coords_tsne[idx, 0], coords_tsne[idx, 1], label=lbl, alpha=0.8)
    ax_tsne.set_title(f"t-SNE of Averaged Spectra (perplexity={perplexity})")
    ax_tsne.set_xlabel("Dim 1")
    ax_tsne.set_ylabel("Dim 2")
    ax_tsne.legend()
    ax_tsne.grid(True, alpha=0.3)

    results = {
        "PCA (2D)": fig_pca,
        "t-SNE (2D)": fig_tsne,
    }

    # Optional mel-space embeddings if librosa is present
    if LIBROSA_AVAILABLE and mel_features and len(mel_features) == len(labels):
        X_mel = np.vstack(mel_features)
        X_mel_c = X_mel - X_mel.mean(axis=0)
        std_mel = X_mel_c.std(axis=0)
        std_mel[std_mel == 0] = 1.0
        X_mel_n = X_mel_c / std_mel

        fig_pca_mel = Figure(figsize=(6, 5))
        ax_pca_mel = fig_pca_mel.add_subplot(111)
        pca_mel = PCA(n_components=min(2, X_mel_n.shape[1]))
        coords_pca_mel = pca_mel.fit_transform(X_mel_n)
        for lbl in sorted(set(labels)):
            idx = [i for i, l in enumerate(labels) if l == lbl]
            ax_pca_mel.scatter(coords_pca_mel[idx, 0], coords_pca_mel[idx, 1], label=lbl, alpha=0.8)
        ax_pca_mel.set_title("PCA of Log-Mel Features")
        ax_pca_mel.set_xlabel("PC1")
        ax_pca_mel.set_ylabel("PC2")
        ax_pca_mel.legend()
        ax_pca_mel.grid(True, alpha=0.3)

        fig_tsne_mel = Figure(figsize=(6, 5))
        ax_tsne_mel = fig_tsne_mel.add_subplot(111)
        perplexity_mel = max(2, min(30, len(selected_ids) - 1))
        tsne_mel = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=perplexity_mel, random_state=42)
        coords_tsne_mel = tsne_mel.fit_transform(X_mel_n)
        for lbl in sorted(set(labels)):
            idx = [i for i, l in enumerate(labels) if l == lbl]
            ax_tsne_mel.scatter(coords_tsne_mel[idx, 0], coords_tsne_mel[idx, 1], label=lbl, alpha=0.8)
        ax_tsne_mel.set_title(f"t-SNE of Log-Mel Features (perplexity={perplexity_mel})")
        ax_tsne_mel.set_xlabel("Dim 1")
        ax_tsne_mel.set_ylabel("Dim 2")
        ax_tsne_mel.legend()
        ax_tsne_mel.grid(True, alpha=0.3)

        results["PCA (Mel)"] = fig_pca_mel
        results["t-SNE (Mel)"] = fig_tsne_mel

    # Optional TensorFlow autoencoder embedding
    if TF_AVAILABLE:
        try:
            input_dim = X_norm.shape[1]
            inputs = tf.keras.Input(shape=(input_dim,))
            x = tf.keras.layers.Dense(64, activation="relu")(inputs)
            x = tf.keras.layers.Dense(16, activation="relu")(x)
            latent = tf.keras.layers.Dense(2, activation=None, name="latent")(x)
            x = tf.keras.layers.Dense(16, activation="relu")(latent)
            x = tf.keras.layers.Dense(64, activation="relu")(x)
            outputs = tf.keras.layers.Dense(input_dim, activation=None)(x)
            autoencoder = tf.keras.Model(inputs, outputs)
            autoencoder.compile(optimizer="adam", loss="mse")
            autoencoder.fit(X_norm, X_norm, epochs=80, batch_size=max(4, len(X_norm)//2), verbose=0)
            encoder = tf.keras.Model(inputs, latent)
            coords_tf = encoder.predict(X_norm, verbose=0)

            fig_tf = Figure(figsize=(6, 5))
            ax_tf = fig_tf.add_subplot(111)
            for lbl in sorted(set(labels)):
                idx = [i for i, l in enumerate(labels) if l == lbl]
                ax_tf.scatter(coords_tf[idx, 0], coords_tf[idx, 1], label=lbl, alpha=0.8)
            ax_tf.set_title("TensorFlow 2D Autoencoder Embedding")
            ax_tf.set_xlabel("Latent 1")
            ax_tf.set_ylabel("Latent 2")
            ax_tf.legend()
            ax_tf.grid(True, alpha=0.3)
            results["TF Autoencoder (2D)"] = fig_tf
        except Exception:
            pass

    return results
