"""Lexical feature extraction for URLs.

These features are computed purely from the URL string — no network calls,
no page fetch — which is what makes the classifier fast and safe to run on
untrusted URLs. Each feature is a well-known phishing signal from the URL
literature (long URLs, many subdomains, IP-as-host, '@' tricks, etc.).
"""
from __future__ import annotations

import math
from ipaddress import ip_address
from urllib.parse import urlparse

SUSPICIOUS_WORDS = (
    "login", "secure", "account", "update", "verify", "bank", "confirm",
    "signin", "password", "ebayisapi", "webscr", "paypal", "wp-admin",
)
SHORTENERS = ("bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly")
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
    # Reject malformed/untrusted input rather than silently scoring an empty
    # hostname or letting urlparse's IPv6/NFKC ValueError crash batch scoring.
    # Raw controls and whitespace also make CLI output unsafe to log.
    if not isinstance(url, str) or not url:
        raise ValueError("URL must be a nonempty string")
    if any(char.isspace() or ord(char) < 0x20 or ord(char) == 0x7F for char in url):
        raise ValueError("URL contains raw whitespace or control characters")
    try:
        parsed = urlparse(url if "://" in url else "http://" + url)
        host = parsed.hostname or ""
    except ValueError as exc:
        raise ValueError("URL has malformed authority") from exc
    if not host:
        raise ValueError("URL has no hostname")
    path = parsed.path or ""

    # Parse IP addresses rather than matching four numeric-looking labels:
    # 999.999.999.999 is not IPv4, and bracketed IPv6 is an IP host too.
    try:
        ip_address(host)
        is_ip = True
    except ValueError:
        is_ip = False
    # IP octets are not DNS subdomains. Match shorteners on DNS-label
    # boundaries, never as substrings of an attacker-controlled hostname.
    dns_host = host.rstrip(".")
    subdomains = max(0, dns_host.count(".") - 1) if dns_host and not is_ip else 0
    is_shortener = not is_ip and any(
        dns_host == domain or dns_host.endswith("." + domain)
        for domain in SHORTENERS
    )

    return {
        "url_length": float(len(url)),
        "host_length": float(len(host)),
        "path_length": float(len(path)),
        "num_dots": float(url.count(".")),
        "num_hyphens": float(url.count("-")),
        "num_subdomains": float(subdomains),
        "num_digits": float(sum(c.isdigit() for c in url)),
        "num_special": float(sum(c in "@?%=&_~" for c in url)),
        "has_ip": 1.0 if is_ip else 0.0,
        "has_at": 1.0 if parsed.username is not None else 0.0,
        "has_https": 1.0 if parsed.scheme == "https" else 0.0,
        "is_shortener": 1.0 if is_shortener else 0.0,
        "num_suspicious_words": float(sum(w in url.lower() for w in SUSPICIOUS_WORDS)),
        "host_entropy": _shannon_entropy(host),
    }


def features_vector(url: str) -> list[float]:
    f = extract_features(url)
    return [f[name] for name in FEATURE_NAMES]
