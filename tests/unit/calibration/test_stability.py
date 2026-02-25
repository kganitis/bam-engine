"""Tests for calibration.stability module."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from calibration.analysis import CalibrationResult
from calibration.stability import (
    _rank_candidates,
    parse_stability_tiers,
    run_tiered_stability,
)
from tests.helpers.scores import make_score as _make_score


class TestParseStabilityTiers:
    """Tests for parse_stability_tiers."""

    def test_basic_parse(self):
        tiers = parse_stability_tiers("100:10,50:20,10:100")
        assert tiers == [(100, 10), (50, 20), (10, 100)]

    def test_single_tier(self):
        tiers = parse_stability_tiers("20:50")
        assert tiers == [(20, 50)]

    def test_spaces_handled(self):
        tiers = parse_stability_tiers("100 : 10 , 50 : 20")
        assert tiers == [(100, 10), (50, 20)]


class TestRankCandidates:
    """Tests for _rank_candidates."""

    def _make_candidates(self) -> list[CalibrationResult]:
        """Create test candidates with known scores."""
        return [
            CalibrationResult(
                {"id": "A"},
                0.8,
                1,
                0,
                0,
                mean_score=0.75,
                std_score=0.10,
                pass_rate=0.9,
                combined_score=0.75 * (1 - 0.10),
            ),
            CalibrationResult(
                {"id": "B"},
                0.7,
                1,
                0,
                0,
                mean_score=0.80,
                std_score=0.05,
                pass_rate=1.0,
                combined_score=0.80 * (1 - 0.05),
            ),
            CalibrationResult(
                {"id": "C"},
                0.6,
                0,
                0,
                1,
                mean_score=0.85,
                std_score=0.20,
                pass_rate=0.5,
                combined_score=0.85 * (1 - 0.20),
            ),
        ]

    def test_combined_ranking(self):
        """Combined: mean * pass_rate * (1 - k*std), k=1.0."""
        candidates = self._make_candidates()
        ranked = _rank_candidates(candidates, rank_by="combined", k_factor=1.0)
        # B: 0.80 * 1.0 * 0.95 = 0.760
        # A: 0.75 * 0.9 * 0.90 = 0.6075
        # C: 0.85 * 0.5 * 0.80 = 0.340
        assert ranked[0].params["id"] == "B"

    def test_stability_ranking(self):
        """Stability: pass_rate DESC, n_fail ASC, combined DESC."""
        candidates = self._make_candidates()
        ranked = _rank_candidates(candidates, rank_by="stability")
        # B has pass_rate=1.0 (highest)
        assert ranked[0].params["id"] == "B"
        # C has pass_rate=0.5 (lowest)
        assert ranked[-1].params["id"] == "C"

    def test_mean_ranking(self):
        """Mean: mean_score DESC only."""
        candidates = self._make_candidates()
        ranked = _rank_candidates(candidates, rank_by="mean")
        # C has highest mean (0.85)
        assert ranked[0].params["id"] == "C"

    def test_k_factor_affects_ranking(self):
        """Higher k penalizes variance more."""
        candidates = self._make_candidates()
        # With k=0 (ignore variance), mean * pass_rate wins
        ranked_k0 = _rank_candidates(
            [CalibrationResult(**{**vars(c)}) for c in candidates],
            rank_by="combined",
            k_factor=0.0,
        )
        # With k=3 (heavy variance penalty), low-std wins
        ranked_k3 = _rank_candidates(
            [CalibrationResult(**{**vars(c)}) for c in candidates],
            rank_by="combined",
            k_factor=3.0,
        )
        # k=0: B wins (0.80*1.0=0.80 > A: 0.75*0.9=0.675 > C: 0.85*0.5=0.425)
        assert ranked_k0[0].params["id"] == "B"
        # k=3: C's high std heavily penalized, still last
        assert ranked_k3[-1].params["id"] == "C"


class TestRunTieredStability:
    """Tests for run_tiered_stability."""

    @patch("calibration.stability.ProcessPoolExecutor", ThreadPoolExecutor)
    @patch("calibration.stability.get_validation_func")
    @patch("calibration.stability.save_checkpoint")
    @patch("calibration.stability.delete_checkpoint")
    def test_tiny_tiers(self, mock_del, mock_save, mock_get_func):
        """Test with tiny tiers [(3, 2), (2, 4)].

        Uses ThreadPoolExecutor so mocks propagate to worker threads.
        """

        def mock_validate(**kwargs):
            return _make_score(0.75)

        mock_get_func.return_value = mock_validate

        candidates = [
            CalibrationResult({"a": i}, 0.5 + i * 0.1, 1, 0, 0) for i in range(5)
        ]

        results = run_tiered_stability(
            candidates,
            "baseline",
            tiers=[(3, 2), (2, 4)],
            n_workers=1,
            n_periods=50,
        )

        # After tier 2, should have <= 2 configs
        assert len(results) <= 2
        # Each should have seed_scores
        for r in results:
            assert r.seed_scores is not None
            assert len(r.seed_scores) >= 2

    @patch("calibration.stability.ProcessPoolExecutor", ThreadPoolExecutor)
    @patch("calibration.stability.get_validation_func")
    @patch("calibration.stability.save_checkpoint")
    @patch("calibration.stability.delete_checkpoint")
    def test_tiers_monotonically_reduce(self, mock_del, mock_save, mock_get_func):
        """Property: each tier keeps <= previous tier's n_configs."""

        def mock_validate(**kwargs):
            return _make_score(0.75)

        mock_get_func.return_value = mock_validate

        candidates = [
            CalibrationResult({"a": i}, 0.5 + i * 0.01, 1, 0, 0) for i in range(10)
        ]

        results = run_tiered_stability(
            candidates,
            "baseline",
            tiers=[(8, 2), (5, 4), (3, 8)],
            n_workers=1,
            n_periods=50,
        )
        assert len(results) <= 3

    @patch("calibration.stability.ProcessPoolExecutor", ThreadPoolExecutor)
    @patch("calibration.stability.get_validation_func")
    @patch("calibration.stability.save_checkpoint")
    @patch("calibration.stability.delete_checkpoint")
    def test_pass_rate_populated(self, mock_del, mock_save, mock_get_func):
        """pass_rate and seed_fails are populated after tiered testing."""

        def mock_validate(**kwargs):
            return _make_score(0.75)

        mock_get_func.return_value = mock_validate

        candidates = [
            CalibrationResult({"a": i}, 0.5 + i * 0.1, 1, 0, 0) for i in range(3)
        ]

        results = run_tiered_stability(
            candidates,
            "baseline",
            tiers=[(3, 3)],
            n_workers=1,
            n_periods=50,
        )

        for r in results:
            assert r.pass_rate is not None
            assert r.seed_fails is not None
            assert len(r.seed_fails) == len(r.seed_scores or [])
