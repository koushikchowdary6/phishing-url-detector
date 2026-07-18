"""Lexical feature extraction for URLs.

These features are computed purely from the URL string — no network calls,
no page fetch — which is what makes the classifier fast and safe to run on
untrusted URLs. Each feature is a well-known phishing signal from the URL
literature (long URLs, many subdomains, IP-as-host, '@' tricks, etc.).
"""
from __future__ import annotations

import math
import re
from urllib.parse import urlparse

SUSPICIOUS_WORDS = (
    "login", "secure", "account", "update", "verify", "bank", "confirm",
    "signin", "password", "ebayisapi", "webscr", "paypal", "wp-admin",
)
SHORTENERS = ("bit.ly", "tinyurl", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly")
FEATURE_NAMES = [
    "url_length", "host_length", "path_length", "num_dots", "num_hyphens",
    "num_subdomains", "num_digits", "num_special", "has_ip", "has_at",
    "has_https", "is_shortener", "num_suspicious_words", "host_entropy",
]


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = {c: s.count(c) for c in set(s)}
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def extract_features(url: str) -> dict[str, float]:
    url = url.strip()
    parsed = urlparse(url if "://" in url else "http://" + url)
    host = parsed.hostname or ""
    path = parsed.path or ""

    has_ip = 1.0 if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", host) else 0.0
    subdomains = max(0, host.count(".") - 1) if host else 0

    return {
        "url_length": float(len(url)),
        "host_length": float(len(host)),
        "path_length": float(len(path)),
        "num_dots": float(url.count(".")),
        "num_hyphens": float(url.count("-")),
        "num_subdomains": float(subdomains),
        "num_digits": float(sum(c.isdigit() for c in url)),
        "num_special": float(sum(c in "@?%=&_~" for c in url)),
        "has_ip": has_ip,
        "has_at": 1.0 if "@" in url else 0.0,
        "has_https": 1.0 if parsed.scheme == "https" else 0.0,
        "is_shortener": 1.0 if any(s in host for s in SHORTENERS) else 0.0,
        "num_suspicious_words": float(sum(w in url.lower() for w in SUSPICIOUS_WORDS)),
        "host_entropy": _shannon_entropy(host),
    }


def features_vector(url: str) -> list[float]:
    f = extract_features(url)
    return [f[name] for name in FEATURE_NAMES]
