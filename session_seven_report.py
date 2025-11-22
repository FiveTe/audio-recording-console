"""
Offline training/report script for SessionSeven recordings.
Uses averaged spectral_data (no app runtime needed).

Outputs:
- Console summary of cross-val accuracies.
- Saved plot: fold accuracies + aggregated confusion matrix.
"""
import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix

APP_ROOT = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(APP_ROOT, "data")
SESSION_NAME = "SessionSeven"
REPORT_DIR = os.path.join(DATA_DIR, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)


def load_session(session_name="SessionSeven"):
    session_dir = os.path.join(DATA_DIR, "recordings", session_name)
    json_files = glob.glob(os.path.join(session_dir, "**", "*_data.json"), recursive=True)
    X, y = [], []
    expected_len = None
    for jf in json_files:
        try:
            with open(jf, "r") as f:
                d = json.load(f)
            spec = d.get("spectral_data")
            label = d.get("label", "Unlabeled")
            amp = d.get("amplitude", 0.0)
            if spec is None:
                continue
            arr = np.asarray(spec)
            if arr.ndim != 1:
                continue
            if expected_len is None:
                expected_len = arr.shape[0]
            if arr.shape[0] != expected_len:
                print(f"Skipping {jf}: spectral length mismatch ({arr.shape[0]} vs {expected_len})")
                continue
            feat = np.concatenate([arr, [amp]], axis=0)
            X.append(feat)
            y.append(label)
        except Exception as e:
            print(f"Failed to load {jf}: {e}")
            continue
    if not X:
        raise RuntimeError(f"No usable data found under {session_dir}")
    return np.vstack(X), np.array(y)


def run_cv_report(X, y, n_splits=5, random_state=42):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    skf = StratifiedKFold(n_splits=min(n_splits, len(np.unique(y_enc))), shuffle=True, random_state=random_state)

    accuracies = []
    cm_sum = np.zeros((len(le.classes_), len(le.classes_)), dtype=np.float64)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_enc), 1):
        clf = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                solver="adam",
                max_iter=400,
                random_state=random_state + fold,
                early_stopping=True,
                n_iter_no_change=15,
            ),
        )
        clf.fit(X[train_idx], y_enc[train_idx])
        preds = clf.predict(X[test_idx])
        acc = accuracy_score(y_enc[test_idx], preds)
        accuracies.append(acc)
        cm_sum += confusion_matrix(y_enc[test_idx], preds, labels=range(len(le.classes_)))
        print(f"Fold {fold}: accuracy={acc:.3f}, size train={len(train_idx)}, test={len(test_idx)}")

    avg_acc = np.mean(accuracies)
    print(f"\nAccuracies per fold: {['{:.3f}'.format(a) for a in accuracies]}")
    print(f"Mean accuracy: {avg_acc:.3f} | Std: {np.std(accuracies):.3f}")
    return accuracies, cm_sum, le


def plot_report(accuracies, cm_sum, le, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy bar plot
    ax_acc = axes[0]
    ax_acc.bar(range(1, len(accuracies) + 1), accuracies, color="#4CAF50")
    ax_acc.axhline(np.mean(accuracies), color="red", linestyle="--", label=f"Mean={np.mean(accuracies):.2f}")
    ax_acc.set_ylim(0, 1)
    ax_acc.set_title("Cross-Validation Accuracies")
    ax_acc.set_xlabel("Fold")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend()

    # Confusion matrix (normalized rows)
    cm_norm = cm_sum / (cm_sum.sum(axis=1, keepdims=True) + 1e-9)
    ax_cm = axes[1]
    im = ax_cm.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax_cm.set_title("Aggregated Confusion Matrix (normalized rows)")
    ax_cm.set_xticks(range(len(le.classes_)))
    ax_cm.set_yticks(range(len(le.classes_)))
    ax_cm.set_xticklabels(le.classes_, rotation=45, ha="right")
    ax_cm.set_yticklabels(le.classes_)
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax_cm.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", color="black", fontsize=7)
    fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    print("Loading SessionSeven spectral data...")
    X, y = load_session(SESSION_NAME)
    print(f"Loaded {len(X)} samples; classes: {set(y)}")
    accuracies, cm_sum, le = run_cv_report(X, y)
    out_path = os.path.join(REPORT_DIR, f"session_seven_report.png")
    plot_report(accuracies, cm_sum, le, out_path)
    print(f"Saved report plot to {out_path}")
    print("Tip: add more balanced samples per label to improve performance.")


if __name__ == "__main__":
    main()
