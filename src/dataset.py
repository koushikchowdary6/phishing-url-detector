"""Synthetic labeled URL dataset generator.

Real phishing URL corpora (e.g. PhishTank) require live feeds and have
licensing constraints, so this project ships a *synthetic* generator that
produces legitimate-looking and phishing-looking URLs with realistic
structure. It is clearly synthetic — the point is a self-contained,
reproducible pipeline you can run offline, not a claim of real-world
accuracy. Swap in a real labeled CSV to train on genuine data.
"""
from __future__ import annotations

import random

LEGIT_DOMAINS = [
    "google.com", "github.com", "wikipedia.org", "amazon.com", "microsoft.com",
    "apple.com", "cloudflare.com", "stackoverflow.com", "nytimes.com", "usf.edu",
    "python.org", "mozilla.org", "linkedin.com", "netflix.com", "reddit.com",
]
LEGIT_PATHS = ["", "/", "/about", "/products/item", "/blog/2025/post", "/help", "/search?q=news"]

PHISH_BRANDS = ["paypal", "apple", "amazon", "netflix", "bankofamerica", "microsoft", "chase"]
PHISH_WORDS = ["secure", "login", "verify", "account", "update", "confirm", "signin"]
PHISH_TLDS = [".xyz", ".top", ".info", ".online", ".click", ".tk"]


def make_legit(rng: random.Random) -> str:
    domain = rng.choice(LEGIT_DOMAINS)
    scheme = "https" if rng.random() < 0.9 else "http"
    path = rng.choice(LEGIT_PATHS)
    sub = rng.choice(["", "www.", "www.", "docs.", "mail."])
    return f"{scheme}://{sub}{domain}{path}"


def make_phish(rng: random.Random) -> str:
    brand = rng.choice(PHISH_BRANDS)
    word = rng.choice(PHISH_WORDS)
    style = rng.randint(0, 3)
    scheme = "http" if rng.random() < 0.7 else "https"
    if style == 0:  # IP host
        ip = ".".join(str(rng.randint(1, 254)) for _ in range(4))
        return f"{scheme}://{ip}/{brand}/{word}?id={rng.randint(1000,9999)}"
    if style == 1:  # @ trick
        return f"{scheme}://{brand}.{word}.com@{rng.choice(LEGIT_DOMAINS)}/{word}"
    if style == 2:  # many-subdomain lookalike
        return f"{scheme}://{brand}-{word}.{brand}.{rng.choice(['com','net'])}{rng.choice(PHISH_TLDS)}/{word}.php"
    # long hyphenated with suspicious words + digits
    tail = "-".join([brand, word, "secure", str(rng.randint(10000, 99999))])
    return f"{scheme}://{tail}{rng.choice(PHISH_TLDS)}/wp-admin/login"


def generate(n: int = 2000, seed: int = 42) -> tuple[list[str], list[int]]:
    """Return (urls, labels) with label 1 = phishing, 0 = legitimate."""
    rng = random.Random(seed)
    urls, labels = [], []
    for _ in range(n // 2):
        urls.append(make_legit(rng)); labels.append(0)
        urls.append(make_phish(rng)); labels.append(1)
    combined = list(zip(urls, labels))
    rng.shuffle(combined)
    urls, labels = zip(*combined)
    return list(urls), list(labels)
