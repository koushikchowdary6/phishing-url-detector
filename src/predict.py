"""Train once, then score arbitrary URLs from the command line."""
from __future__ import annotations

import sys

from features import extract_features, features_vector
from train import train


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python src/predict.py <url> [<url> ...]", file=sys.stderr)
        return 2
    clf, _, _ = train()
    for url in sys.argv[1:]:
        proba = clf.predict_proba([features_vector(url)])[0, 1]
        verdict = "PHISHING" if proba >= 0.5 else "legit"
        print(f"[{verdict:8}] p={proba:.2f}  {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
