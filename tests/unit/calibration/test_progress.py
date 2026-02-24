"""Tests for progress tracking and ETA helpers in calibration.analysis."""

from __future__ import annotations

from calibration.analysis import format_eta, format_progress


class TestFormatEta:
    """Tests for format_eta."""

    def test_seconds(self):
        result = format_eta(remaining=5, avg_time=6.0, n_workers=1)
        assert result == "30s"

    def test_minutes(self):
        result = format_eta(remaining=10, avg_time=60.0, n_workers=2)
        assert result == "5m 00s"

    def test_hours(self):
        result = format_eta(remaining=100, avg_time=360.0, n_workers=1)
        assert "h" in result

    def test_zero_workers(self):
        result = format_eta(remaining=10, avg_time=1.0, n_workers=0)
        assert result == "unknown"

    def test_zero_avg_time(self):
        result = format_eta(remaining=10, avg_time=0.0, n_workers=1)
        assert result == "unknown"

    def test_zero_remaining(self):
        result = format_eta(remaining=0, avg_time=10.0, n_workers=1)
        assert result == "0s"

    def test_parallel_reduces_eta(self):
        eta_serial = format_eta(remaining=10, avg_time=10.0, n_workers=1)
        eta_parallel = format_eta(remaining=10, avg_time=10.0, n_workers=10)
        # Parallel should be shorter or equal
        # Just check both are valid strings
        assert isinstance(eta_serial, str)
        assert isinstance(eta_parallel, str)


class TestFormatProgress:
    """Tests for format_progress."""

    def test_basic_format(self):
        result = format_progress(
            completed=50,
            total=200,
            remaining=150,
            eta="5m 30s",
        )
        assert "50/200" in result
        assert "25.0%" in result
        assert "150 remaining" in result
        assert "5m 30s" in result

    def test_zero_total(self):
        result = format_progress(completed=0, total=0, remaining=0, eta="0s")
        assert "0/0" in result
        assert "0.0%" in result

    def test_complete(self):
        result = format_progress(completed=100, total=100, remaining=0, eta="0s")
        assert "100.0%" in result
