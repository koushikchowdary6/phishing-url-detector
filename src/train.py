"""Train and evaluate the phishing-URL classifier."""
from __future__ import annotations

import argparse

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from dataset import generate
from features import FEATURE_NAMES, features_vector


def build_dataset(n: int, seed: int):
    urls, labels = generate(n, seed)
    X = np.array([features_vector(u) for u in urls])
    y = np.array(labels)
    return X, y


def _feature_disjoint_split(X: np.ndarray, y: np.ndarray, seed: int):
    """Keep feature-identical URLs together to prevent exact-vector leakage.

    The synthetic generator can repeat the same URL, and different URLs can
    also produce identical lexical feature vectors. A row-wise random split
    would put these indistinguishable observations on both sides of the
    evaluation boundary, overstating the independence of the holdout.
    """
    _, groups = np.unique(X, axis=0, return_inverse=True)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    if set(np.unique(y[train_idx])) != {0, 1} or set(np.unique(y[test_idx])) != {0, 1}:
        raise ValueError("both holdout partitions must contain both labels; increase --samples")
    return train_idx, test_idx


def train(n: int = 2000, seed: int = 42):
    if isinstance(n, bool) or not isinstance(n, int) or n < 8 or n % 2:
        raise ValueError("samples must be an even integer of at least 8")
    X, y = build_dataset(n, seed)
    train_idx, test_idx = _feature_disjoint_split(X, y, seed)
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    clf = RandomForestClassifier(n_estimators=120, random_state=seed)
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)[:, 1]
    preds = (proba >= 0.5).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(y_te, proba)),
        "report": classification_report(y_te, preds, target_names=["legit", "phish"], output_dict=True),
        "split": {
            "method": "feature-disjoint GroupShuffleSplit",
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
            "duplicate_feature_rows": int(len(X) - len(np.unique(X, axis=0))),
        },
    }
    return clf, metrics, (X_te, y_te)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the phishing-URL classifier")
    parser.add_argument("-n", "--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    clf, metrics, _ = train(args.samples, args.seed)
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    rep = metrics["report"]
    print(f"Accuracy: {rep['accuracy']:.4f}")
    print(f"Phish precision/recall: {rep['phish']['precision']:.3f} / {rep['phish']['recall']:.3f}")
    split = metrics["split"]
    print(
        f"Holdout: {split['method']} ({split['train_rows']} train / "
        f"{split['test_rows']} test; {split['duplicate_feature_rows']} "
        "duplicate feature rows kept within groups)"
    )
    importances = sorted(zip(FEATURE_NAMES, clf.feature_importances_), key=lambda t: -t[1])
    print("\nTop features:")
    for name, imp in importances[:6]:
        print(f"  {name:22} {imp:.3f}")


if __name__ == "__main__":
    main()
