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
    invalid = False
    for index, url in enumerate(sys.argv[1:], start=1):
        try:
            vector = features_vector(url)
        except ValueError as exc:
            # Report the argument position, not raw untrusted URL content.
            # A bad URL must not prevent scoring later valid arguments.
            print(f"URL argument {index} rejected: {exc}", file=sys.stderr)
            invalid = True
            continue
        proba = clf.predict_proba([vector])[0, 1]
        verdict = "PHISHING" if proba >= 0.5 else "legit"
        print(f"[{verdict:8}] p={proba:.2f}  {url}")
    return 2 if invalid else 0


if __name__ == "__main__":
    raise SystemExit(main())
