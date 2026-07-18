"""Train and evaluate the phishing-URL classifier."""
from __future__ import annotations

import argparse

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from dataset import generate
from features import FEATURE_NAMES, features_vector


def build_dataset(n: int, seed: int):
    urls, labels = generate(n, seed)
    X = np.array([features_vector(u) for u in urls])
    y = np.array(labels)
    return X, y


def train(n: int = 2000, seed: int = 42):
    X, y = build_dataset(n, seed)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    clf = RandomForestClassifier(n_estimators=120, random_state=seed)
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)[:, 1]
    preds = (proba >= 0.5).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(y_te, proba)),
        "report": classification_report(y_te, preds, target_names=["legit", "phish"], output_dict=True),
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
    importances = sorted(zip(FEATURE_NAMES, clf.feature_importances_), key=lambda t: -t[1])
    print("\nTop features:")
    for name, imp in importances[:6]:
        print(f"  {name:22} {imp:.3f}")


if __name__ == "__main__":
    main()
