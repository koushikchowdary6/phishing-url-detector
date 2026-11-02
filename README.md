# Phishing URL Detector 🎣🤖

[![CI](https://github.com/koushikchowdary6/phishing-url-detector/actions/workflows/ci.yml/badge.svg)](https://github.com/koushikchowdary6/phishing-url-detector/actions/workflows/ci.yml)

A machine-learning classifier that flags **phishing URLs** from **lexical
features of the URL string alone** — no page fetch, no network calls — so it's
fast and safe to run on untrusted links. Built with scikit-learn
(`RandomForestClassifier`).

## What it looks at

14 well-known URL phishing signals, computed from the string: length, number of
dots/hyphens/digits, subdomain count, IP-as-host, the `user@host` trick,
HTTPS vs HTTP, URL-shortener domains, suspicious keywords (`login`, `verify`,
`secure`, …), and the Shannon entropy of the hostname. See `src/features.py`.

**Authority-aware feature integrity:** IP-host detection parses actual IPv4
and IPv6 addresses instead of treating any four numeric labels as an IP.
Shortener detection matches exact hostnames or DNS subdomains (not
`bit.ly.attacker.example` or URL-path mentions), and the `@` feature
requires URL userinfo rather than an ordinary email in a path or query.
IP address octets are no longer counted as DNS subdomains. Regression
tests cover these edge cases. These are lexical feature corrections,
not independently measured phishing-detection improvements.

## Quickstart

```bash
pip install -r requirements.txt

# Train + evaluate on the bundled synthetic dataset
PYTHONPATH=src python src/train.py

# Score your own URLs
PYTHONPATH=src python src/predict.py https://github.com/login http://paypal-secure-verify-88213.xyz/wp-admin/login
```

```
[legit   ] p=0.03  https://github.com/login
[PHISHING] p=0.95  http://paypal-secure-verify-88213.xyz/wp-admin/login
```

## Tests

```bash
PYTHONPATH=src python -m pytest tests/ -v
```

7 tests cover feature extraction (the `@`-trick, IP-host, and HTTPS signals),
dataset balance, and that the trained model beats chance and scores an obvious
phish as malicious. CI trains and evaluates on every push.

## ⚠️ Honesty about the data & metrics

The bundled dataset is **synthetic** (`src/dataset.py`) — generated
legitimate-looking and phishing-looking URLs. Real phishing feeds (PhishTank,
etc.) need live access and licensing, so this project ships a reproducible
offline generator instead. Because synthetic phishing URLs are cleanly
separable, the model scores near-perfectly here; **that is not a real-world
accuracy claim.** The pipeline is designed so you can drop in a real labeled
CSV and retrain — the feature extraction and model code don't change.

## Why this design

URL-only (lexical) detection is the standard fast first line: it needs no
sandbox and can't be evaded by cloaking the landing page. It's usually combined
with host-based and content-based signals in production — this repo implements
the lexical tier cleanly and testably.

## Author

Koushik Chowdary — [LinkedIn](https://linkedin.com/in/koushik-chowdary) · [GitHub](https://github.com/koushikchowdary6)
