import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from features import extract_features, FEATURE_NAMES
from dataset import generate
from train import train


def test_feature_vector_shape():
    f = extract_features("https://example.com/path")
    assert set(f.keys()) == set(FEATURE_NAMES)


def test_at_trick_flagged():
    f = extract_features("http://bank.com@evil.example/login")
    assert f["has_at"] == 1.0


def test_ip_host_flagged():
    f = extract_features("http://192.168.1.10/login")
    assert f["has_ip"] == 1.0


def test_https_detected():
    assert extract_features("https://example.com")["has_https"] == 1.0
    assert extract_features("http://example.com")["has_https"] == 0.0


def test_dataset_is_balanced_and_labeled():
    urls, labels = generate(200, seed=1)
    assert len(urls) == len(labels) == 200
    assert sum(labels) == 100  # 50/50 phish/legit


def test_model_learns_signal():
    # On the synthetic dataset the model should be clearly better than chance.
    _, metrics, _ = train(n=800, seed=7)
    assert metrics["roc_auc"] > 0.9
    assert metrics["report"]["accuracy"] > 0.9


def test_obvious_phish_scores_high():
    from features import features_vector
    clf, _, _ = train(n=800, seed=3)
    p = clf.predict_proba([features_vector(
        "http://paypal-secure-verify-88213.xyz/wp-admin/login")])[0, 1]
    assert p > 0.5
