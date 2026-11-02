"""URL lexical signals must describe the parsed authority, not substrings."""

import pytest

from features import extract_features, features_vector, FEATURE_NAMES


@pytest.mark.parametrize("url", [
    "https://bit.ly/abc",
    "https://www.bit.ly/abc",
    "https://TINYURL.COM./abc",
])
def test_real_shortener_host_matches_dns_boundary(url):
    assert extract_features(url)["is_shortener"] == 1.0


@pytest.mark.parametrize("url", [
    "https://notbit.ly.example.com/",
    "https://bit.ly.attacker.example/",
    "https://example.com/path/bit.ly",
    "https://example.com/?next=bit.ly",
])
def test_shortener_substrings_outside_host_domain_do_not_match(url):
    assert extract_features(url)["is_shortener"] == 0.0


@pytest.mark.parametrize("url", [
    "http://192.168.1.10/login",
    "https://[2001:db8::1]/login",
    "http://[::1]:8080/",
])
def test_valid_ipv4_ipv6_hosts_are_ip_and_not_subdomains(url):
    features = extract_features(url)
    assert features["has_ip"] == 1.0
    assert features["num_subdomains"] == 0.0


@pytest.mark.parametrize("url", [
    "http://999.999.999.999/login",
    "https://192.168.1.999/",
    "https://example.com/",
])
def test_invalid_ip_looking_hosts_are_not_ip(url):
    assert extract_features(url)["has_ip"] == 0.0


def test_userinfo_at_trick_not_confused_with_path_or_query():
    assert extract_features("http://bank.com@evil.example/login")["has_at"] == 1.0
    assert extract_features("https://example.com/users/@alice")["has_at"] == 0.0
    assert extract_features("https://example.com/?contact=a@b.com")["has_at"] == 0.0


def test_feature_schema_and_vector_stay_stable():
    features = extract_features("https://[2001:db8::1]/login")
    assert list(features) == FEATURE_NAMES
    assert features_vector("https://[2001:db8::1]/login") == list(features.values())
