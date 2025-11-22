import numpy as np

try:
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import classification_report, confusion_matrix
except Exception:
    MLPClassifier = None


@register_model("Averaged Spectrum MLP")
def train_spectrum_mlp(data):
    """
    Train an MLP on preprocessed (averaged) spectral_data stored in each recording JSON.
    Uses a simple train/val split and prints a classification report.
    """
    if MLPClassifier is None:
        print("Averaged Spectrum MLP requires scikit-learn. Please install it first.")
        return None

    X = []
    y = []
    expected_len = None

    for entry in data.values():
        spec = entry.get("spectral_data")
        label = entry.get("label", "Unlabeled")
        amp = entry.get("amplitude", 0.0)
        if spec is None:
            continue
        arr = np.asarray(spec)
        if arr.ndim != 1:
            print(f"Skipping {entry.get('base_filename')}: spectral_data not 1-D.")
            continue
        if expected_len is None:
            expected_len = arr.shape[0]
        if arr.shape[0] != expected_len:
            print(f"Skipping {entry.get('base_filename')}: inconsistent spectral length ({arr.shape[0]} vs {expected_len}).")
            continue
        # Append amplitude as a feature; scaler will normalize magnitude differences.
        feat = np.concatenate([arr, [amp]], axis=0)
        X.append(feat)
        y.append(label)

    if len(X) < 2:
        print("Not enough consistent spectral samples to train.")
        return None

    X = np.vstack(X)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_val, y_train, y_val = train_test_split(X, y_enc, test_size=0.25, random_state=42, stratify=y_enc)

    clf = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            max_iter=400,
            random_state=42,
            early_stopping=True,
            n_iter_no_change=15,
        ),
    )

    print("Training Averaged Spectrum MLP...")
    clf.fit(X_train, y_train)

    print("Validation report:")
    y_pred = clf.predict(X_val)
    print(classification_report(y_val, y_pred, target_names=le.classes_))
    cm = confusion_matrix(y_val, y_pred)
    print("Confusion matrix (rows=true, cols=pred):")
    print(le.classes_)
    print(cm)

    return {"model": clf, "label_encoder": le}
