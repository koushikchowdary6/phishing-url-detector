"""Malformed URLs must not corrupt lexical features or abort a prediction batch."""

import sys

import pytest

import predict
from features import extract_features, features_vector


@pytest.mark.parametrize("url", [
    "",
    "http://",
    "https://[2001:db8::1",
    "https://[not-ipv6]/login",
    "https://example.com/\nforged-log-line",
    "https://exa mple.com/login",
    "https://example.com/\x1b[31m",
])
def test_invalid_url_is_rejected_before_feature_extraction(url):
    with pytest.raises(ValueError, match="URL"):
        extract_features(url)
    with pytest.raises(ValueError, match="URL"):
        features_vector(url)


@pytest.mark.parametrize("url", [
    "https://example.com/login",
    "example.com/path",
    "http://192.0.2.1/",
    "https://[2001:db8::1]/",
])
def test_valid_url_forms_remain_supported(url):
    assert len(features_vector(url)) == 14


def test_predict_cli_skips_bad_urls_and_returns_nonzero(monkeypatch, capsys):
    class FakeClassifier:
        def predict_proba(self, features):
            assert len(features[0]) == 14
            return [[0.75, 0.25]]

    monkeypatch.setattr(predict, "train", lambda: (FakeClassifier(), {}, ()))
    monkeypatch.setattr(sys, "argv", [
        "predict.py",
        "https://example.com/",
        "https://[broken",
        "https://example.org/",
    ])

    assert predict.main() == 2
    captured = capsys.readouterr()
    assert captured.out.count("p=0.25") == 2
    assert "URL argument 2 rejected" in captured.err
    assert "[broken" not in captured.err


def test_predict_cli_success_with_only_valid_urls(monkeypatch, capsys):
    class FakeClassifier:
        def predict_proba(self, features):
            return [[0.25, 0.75]]

    monkeypatch.setattr(predict, "train", lambda: (FakeClassifier(), {}, ()))
    monkeypatch.setattr(sys, "argv", ["predict.py", "https://example.com/"])

    assert predict.main() == 0
    assert "PHISHING" in capsys.readouterr().out
