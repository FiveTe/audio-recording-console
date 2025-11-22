"""
Offline log-mel training/report script for SessionSeven recordings.
Runs multiple models on the same data with cross-validation and saves accuracy + confusion matrices.

Models:
- Log-Mel MLP
- Linear SVM
- Random Forest
"""
import os
import glob
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import librosa
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

APP_ROOT = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(APP_ROOT, "data")
SESSION_NAME = "SessionSeven"
REPORT_DIR = os.path.join(DATA_DIR, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)


def load_logmel(session_name="SessionSeven"):
    session_dir = os.path.join(DATA_DIR, "recordings", session_name)
    json_files = glob.glob(os.path.join(session_dir, "**", "*_data.json"), recursive=True)
    X, y, groups = [], [], []
    for jf in json_files:
        try:
            with open(jf, "r") as f:
                d = json.load(f)
            wav_path = d.get("wav_path")
            label = d.get("label", "Unlabeled")
            session = d.get("session", session_name)
            amp = d.get("amplitude", 0.0)
            if not wav_path or not os.path.exists(wav_path):
                continue
            audio, sr = librosa.load(wav_path, sr=None, mono=True)
            mel = librosa.feature.melspectrogram(
                y=audio, sr=sr, n_fft=2048, hop_length=512, n_mels=64, fmin=50, fmax=sr / 2
            )
            mel_db = librosa.power_to_db(mel + 1e-9)
            feat = mel_db.mean(axis=1)  # 64-dim
            feat = np.concatenate([feat, [amp]], axis=0)
            X.append(feat)
            y.append(label)
            groups.append(session)
        except Exception as e:
            print(f"Skipping {jf}: {e}")
    if not X:
        raise RuntimeError(f"No usable data found under {session_dir}")
    return np.vstack(X), np.array(y), np.array(groups)


def run_cv_model(model_name, estimator, X, y_enc, n_classes, strategy="stratified", n_splits=5, seed=42, groups=None):
    if strategy == "group" and groups is not None:
        splitter = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
        splits = splitter.split(X, y_enc, groups)
    else:
        splitter = StratifiedKFold(n_splits=min(n_splits, len(np.unique(y_enc))), shuffle=True, random_state=seed)
        splits = splitter.split(X, y_enc)

    accs = []
    cm_sum = np.zeros((n_classes, n_classes), dtype=np.float64)
    for fold, (tr, te) in enumerate(splits, 1):
        estimator.fit(X[tr], y_enc[tr])
        preds = estimator.predict(X[te])
        acc = accuracy_score(y_enc[te], preds)
        accs.append(acc)
        cm_sum += confusion_matrix(y_enc[te], preds, labels=range(n_classes))
        print(f"[{model_name}] fold {fold}: acc={acc:.3f} train={len(tr)} test={len(te)}")
    return np.mean(accs), np.std(accs), accs, cm_sum


def plot_combined_report(model_stats, le, out_path):
    fig, axes = plt.subplots(1, len(model_stats) + 1, figsize=(5 * (len(model_stats) + 1), 5))

    # Accuracies bar (first axis)
    ax_acc = axes[0]
    names = [m["name"] for m in model_stats]
    means = [m["mean"] for m in model_stats]
    stds = [m["std"] for m in model_stats]
    ax_acc.bar(range(len(names)), means, yerr=stds, color="#4CAF50", alpha=0.8, capsize=4)
    ax_acc.set_xticks(range(len(names)))
    ax_acc.set_xticklabels(names, rotation=20, ha="right")
    ax_acc.set_ylim(0, 1)
    ax_acc.set_title("Cross-Validation Accuracy (mean ± std)")
    ax_acc.set_ylabel("Accuracy")

    # Confusion matrices
    for idx, m in enumerate(model_stats, 1):
        ax = axes[idx]
        cm_norm = m["cm"] / (m["cm"].sum(axis=1, keepdims=True) + 1e-9)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"{m['name']} CM")
        ax.set_xticks(range(len(le.classes_)))
        ax.set_yticks(range(len(le.classes_)))
        ax.set_xticklabels(le.classes_, rotation=45, ha="right")
        ax.set_yticklabels(le.classes_)
        for i in range(cm_norm.shape[0]):
            for j in range(cm_norm.shape[1]):
                ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", color="black", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_model(model_stat, le, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # Accuracy list
    ax_acc = axes[0]
    accs = model_stat["accs"]
    ax_acc.bar(range(1, len(accs) + 1), accs, color="#2196F3")
    ax_acc.axhline(np.mean(accs), color="red", linestyle="--", label=f"Mean={np.mean(accs):.2f}")
    ax_acc.set_ylim(0, 1)
    ax_acc.set_title(f"{model_stat['name']} Accuracies")
    ax_acc.set_xlabel("Fold")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend()

    cm_norm = model_stat["cm"] / (model_stat["cm"].sum(axis=1, keepdims=True) + 1e-9)
    ax_cm = axes[1]
    im = ax_cm.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax_cm.set_title(f"{model_stat['name']} Confusion Matrix")
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


def parse_args():
    p = argparse.ArgumentParser(description="SessionSeven log-mel report (multi-model CV).")
    p.add_argument("--session", default=SESSION_NAME, help="Session folder name (default: SessionSeven)")
    p.add_argument("--folds", type=int, default=5, help="Number of CV folds (default: 5)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for stratified CV")
    p.add_argument("--strategy", choices=["stratified", "group"], default="stratified",
                   help="CV split strategy: stratified or group (GroupKFold by session)")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Loading {args.session} log-mel features...")
    X, y, groups = load_logmel(args.session)
    print(f"Loaded {len(X)} samples; classes: {set(y)}")

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(le.classes_)

    models = [
        ("Log-Mel MLP", make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", solver="adam", max_iter=400, random_state=args.seed, early_stopping=True, n_iter_no_change=15))),
        ("Linear SVM", make_pipeline(StandardScaler(), LinearSVC(random_state=args.seed))),
        ("LogReg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=200, random_state=args.seed))),
        ("kNN-5", make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))),
        ("Random Forest", RandomForestClassifier(n_estimators=200, random_state=args.seed)),
        ("Extra Trees", ExtraTreesClassifier(n_estimators=200, random_state=args.seed)),
    ]

    stats = []
    for name, est in models:
        mean_acc, std_acc, accs, cm = run_cv_model(
            name, est, X, y_enc, n_classes, strategy=args.strategy, n_splits=args.folds, seed=args.seed, groups=groups
        )
        stats.append({"name": name, "mean": mean_acc, "std": std_acc, "cm": cm, "accs": accs})
        print(f"{name}: accs={['{:.3f}'.format(a) for a in accs]}, mean={mean_acc:.3f}, std={std_acc:.3f}")

    suffix = f"{args.strategy}_cv{args.folds}"
    out_path = os.path.join(REPORT_DIR, f"session_seven_logmel_report_{suffix}.png")
    plot_combined_report(stats, le, out_path)
    print(f"Saved combined report plot to {out_path}")

    # Save per-model reports
    for m in stats:
        indiv_path = os.path.join(REPORT_DIR, f"session_seven_logmel_{m['name'].replace(' ', '_').lower()}_{suffix}.png")
        plot_per_model(m, le, indiv_path)
        print(f"Saved {m['name']} report to {indiv_path}")
    print("Tip: add balanced samples per class to improve performance.")


if __name__ == "__main__":
    main()
