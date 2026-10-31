import pytest

from scripts.load_test import percentile, validate_target_url


def test_load_probe_only_accepts_local_health_endpoints() -> None:
    assert validate_target_url("http://127.0.0.1:8000/ready") == "http://127.0.0.1:8000/ready"

    with pytest.raises(ValueError, match="loopback"):
        validate_target_url("https://example.com/ready")


def test_load_probe_rejects_sensitive_or_mutating_targets() -> None:
    with pytest.raises(ValueError, match="without credentials"):
        validate_target_url("http://user:secret@localhost/ready")
    with pytest.raises(ValueError, match="health or /ready"):
        validate_target_url("http://localhost/api/v1/consultations")


def test_percentile_reports_nearest_rank_for_small_samples() -> None:
    assert percentile([30.0, 10.0, 20.0], 95) == 30.0
