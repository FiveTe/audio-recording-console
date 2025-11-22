import os
import numpy as np

try:
    import librosa
except Exception as e:
    librosa = None
try:
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import classification_report, confusion_matrix
except Exception as e:
    MLPClassifier = None


@register_model("Log-Mel MLP")
def train_logmel_mlp(data):
    """
    Train a small MLP on log-mel averaged features from recordings.
    Prints a quick validation report (train/hold-out split).
    """
    if librosa is None or MLPClassifier is None:
        print("Log-Mel MLP requires librosa and scikit-learn. Please install them first.")
        return None

    X = []
    y = []
    for entry in data.values():
        wav_path = entry.get("wav_path")
        label = entry.get("label", "Unlabeled")
        amp = entry.get("amplitude", 0.0)
        if not wav_path or not os.path.exists(wav_path):
            print(f"Skipping missing audio: {wav_path}")
            continue
        try:
            audio, sr = librosa.load(wav_path, sr=None, mono=True)
            mel = librosa.feature.melspectrogram(
                y=audio, sr=sr, n_fft=2048, hop_length=512, n_mels=64, fmin=50, fmax=sr / 2
            )
            mel_db = librosa.power_to_db(mel + 1e-9)
            feature = mel_db.mean(axis=1)  # average over time -> 64-dim
            # Append amplitude as an explicit feature (StandardScaler will normalize)
            feature = np.concatenate([feature, [amp]], axis=0)
            X.append(feature)
            y.append(label)
        except Exception as e:
            print(f"Failed to process {wav_path}: {e}")

    if not X:
        print("No training data available after preprocessing.")
        return None

    X = np.vstack(X)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_val, y_train, y_val = train_test_split(X, y_enc, test_size=0.25, random_state=42, stratify=y_enc)

    clf = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=300,
            random_state=42,
            early_stopping=True,
            n_iter_no_change=10,
        ),
    )

    print("Training Log-Mel MLP...")
    clf.fit(X_train, y_train)

    print("Validation report:")
    y_pred = clf.predict(X_val)
    print(classification_report(y_val, y_pred, target_names=le.classes_))
    cm = confusion_matrix(y_val, y_pred)
    print("Confusion matrix (rows=true, cols=pred):")
    print(le.classes_)
    print(cm)

    # Return trained pipeline and label encoder for downstream use if needed
    return {"model": clf, "label_encoder": le}
